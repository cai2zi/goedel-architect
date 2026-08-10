from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


WORK = Path("/ssd/czx/czx_work/cot_blueprint_refine")
V7 = WORK / "qwen3_8b_397b_wrong10_step_v7_phase1b_plan_subgraph" / "robustpa" / "blueprint"
V6_REPORT = WORK / "qwen3_8b_397b_wrong76_step_v6_report" / "report.json"
OUT = WORK / "qwen3_8b_397b_wrong10_step_v7_phase1b_plan_subgraph_report"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def longest_repeat(history: list[dict]) -> int:
    longest = current = 0
    previous = None
    for row in history:
        value = tuple(sorted(row.get("plannedNodes") or ()))
        current = current + 1 if value == previous else 1
        previous = value
        longest = max(longest, current)
    return longest


def main() -> None:
    rows = read_jsonl(V7 / "results.jsonl")
    v6 = json.loads(V6_REPORT.read_text(encoding="utf-8"))
    v6_cases = {row["source_id"]: row for row in v6["cases"]}
    cases = []
    for row in rows:
        trace = read_jsonl(Path(row["trace_path"]))
        history = list(row.get("phase1b_edit_history") or ())
        usage = Counter()
        prompt_usage = Counter()
        completion_usage = Counter()
        length_finishes = 0
        for event in trace:
            if event.get("kind") != "llm_request_end" or not event.get("ok"):
                continue
            args = event.get("args") or {}
            phase = str(args.get("phase") or "unknown")
            total = int(args.get("total_tokens") or 0)
            usage[phase] += total
            prompt_usage[phase] += int(args.get("prompt_tokens") or 0)
            completion_usage[phase] += int(args.get("completion_tokens") or 0)
            length_finishes += str(args.get("finish_reason") or "").lower() == "length"
        updates = [
            event for event in trace
            if event.get("kind") == "phase1BSemanticObligationsUpdated"
        ]
        commits = [event for event in trace if event.get("kind") == "phase1BSubgraphCommit"]
        rollbacks = [event for event in trace if event.get("kind") == "phase1BSubgraphRollback"]
        rollback_hash_ok = all(
            (event.get("args") or {}).get("committedHashBefore")
            == (event.get("args") or {}).get("committedHashAfter")
            for event in rollbacks
        )
        baseline = v6_cases.get(row["source_id"], {})
        cases.append({
            "source_id": row["source_id"],
            "v6_status": baseline.get("v6_status"),
            "v7_status": row.get("status"),
            "error": row.get("error", ""),
            "rounds": len(history),
            "commits": len(commits),
            "rollbacks": len(rollbacks),
            "rollback_hash_ok": rollback_hash_ok,
            "accepted_edits": sum(len(item.get("accepted") or ()) for item in history),
            "max_subgraph_size": max(
                [len(item.get("accepted") or ()) for item in history] or [0]
            ),
            "used_multi_node_subgraph": any(
                len(item.get("accepted") or ()) >= 2 for item in history
            ),
            "longest_repeated_plan_nodes": longest_repeat(history),
            "search_calls": sum(
                event.get("kind") == "phase1BMathlibSearchResult" for event in trace
            ),
            "open_obligation_curve": [
                int((event.get("args") or {}).get("openObligationCount") or 0)
                for event in updates
            ],
            "final_open_obligations": len(
                (((row.get("phase1b_validation") or {}).get("semanticAudit") or {})
                 .get("openObligations") or ())
            ),
            "tokens_by_phase": dict(usage),
            "prompt_tokens_by_phase": dict(prompt_usage),
            "completion_tokens_by_phase": dict(completion_usage),
            "total_tokens": sum(usage.values()),
            "length_finishes": length_finishes,
            "plans": [{
                "round": item.get("round"),
                "nodes": item.get("plannedNodes") or [],
                "plan": item.get("plan") or "",
                "committed": bool(item.get("committed")),
                "rollback_reasons": item.get("rollbackReasons") or [],
            } for item in history],
        })

    statuses = Counter(item["v7_status"] for item in cases)
    controls = {
        "MATH-500/test/counting_and_probability/765.json",
        "MATH-500/test/prealgebra/378.json",
        "MATH-500/test/intermediate_algebra/662.json",
    }
    coupled = {item["source_id"] for item in cases} - controls
    recovered = sorted(
        item["source_id"] for item in cases
        if item["source_id"] in coupled and item["v7_status"] == "strictAccepted"
    )
    controls_kept = sorted(
        item["source_id"] for item in cases
        if item["source_id"] in controls and item["v7_status"] == "strictAccepted"
    )
    baseline_cases = [v6_cases[item["source_id"]] for item in cases]
    baseline_tokens = sum(int(item.get("all_llm_tokens") or 0) for item in baseline_cases)
    baseline_judge_tokens = sum(int(item.get("judge_tokens") or 0) for item in baseline_cases)
    judge_tokens = sum(
        item["tokens_by_phase"].get("phase1BFormalDecompiler", 0)
        + item["tokens_by_phase"].get("phase1BStrictComparator", 0)
        for item in cases
    )
    summary = {
        "total": len(cases),
        "status_counts": dict(statuses),
        "terminal_count": len(cases),
        "infra_errors": [item["source_id"] for item in cases if "infra" in item["error"].lower()],
        "format_errors": [item["source_id"] for item in cases if "invalid:" in item["error"].lower()],
        "length_finish_count": sum(item["length_finishes"] for item in cases),
        "controls_kept": controls_kept,
        "controls_regressed": sorted(controls - set(controls_kept)),
        "coupled_failures_recovered": recovered,
        "multi_node_subgraph_cases": sum(item["used_multi_node_subgraph"] for item in cases),
        "rollback_hash_all_valid": all(item["rollback_hash_ok"] for item in cases),
        "cases_with_five_repeated_plan_node_sets": [
            item["source_id"] for item in cases
            if item["longest_repeated_plan_nodes"] >= 5
        ],
        "mean_turns": sum(item["rounds"] for item in cases) / len(cases),
        "total_tokens": sum(item["total_tokens"] for item in cases),
        "planner_tokens": sum(
            item["tokens_by_phase"].get("phase1BPlanner", 0) for item in cases
        ),
        "editor_tokens": sum(
            item["tokens_by_phase"].get("phase1B", 0) for item in cases
        ),
        "judge_tokens": judge_tokens,
        "v6_total_tokens": baseline_tokens,
        "v6_judge_tokens": baseline_judge_tokens,
        "total_token_change_fraction": (
            sum(item["total_tokens"] for item in cases) / baseline_tokens - 1
        ),
    }
    payload = {"summary": summary, "cases": cases}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Phase 1B v7 Plan/Subgraph pilot report", "",
        f"- Status: {dict(statuses)}",
        f"- Controls kept: {len(controls_kept)}/3; regressions: {summary['controls_regressed']}",
        f"- New coupled recoveries: {len(recovered)}/7 ({recovered})",
        f"- Multi-node committed subgraphs: {summary['multi_node_subgraph_cases']}/10 cases",
        f"- Rollback committed hashes preserved: {summary['rollback_hash_all_valid']}",
        f"- Mean repair turns: {summary['mean_turns']:.1f}",
        f"- Planner tokens: {summary['planner_tokens']}; Editor tokens: {summary['editor_tokens']}; total LLM tokens: {summary['total_tokens']}",
        f"- v6 same-case tokens: {summary['v6_total_tokens']}; v7 change: {summary['total_token_change_fraction']:.1%}",
        f"- Judge tokens: v6={summary['v6_judge_tokens']}, v7={summary['judge_tokens']}",
        f"- Length finishes: {summary['length_finish_count']}; infra errors: {len(summary['infra_errors'])}",
        f"- Five-round repeated node sets: {summary['cases_with_five_repeated_plan_node_sets']}",
        "", "## Per-case comparison", "",
        "| source_id | v6 | v7 | turns | commits/rollbacks | edits | max subgraph | searches | open curve | tokens |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for item in cases:
        lines.append(
            f"| {item['source_id']} | {item['v6_status']} | {item['v7_status']} | "
            f"{item['rounds']} | {item['commits']}/{item['rollbacks']} | "
            f"{item['accepted_edits']} | {item['max_subgraph_size']} | "
            f"{item['search_calls']} | {item['open_obligation_curve']} | "
            f"{item['total_tokens']} |"
        )
    lines.extend([
        "", "## Acceptance result", "",
        "The pilot **fails the promotion gate**. All 10 cases terminated without "
        "infrastructure or length failures, all three controls stayed strictAccepted, "
        "8/10 cases committed a multi-node subgraph, and every deterministic rollback "
        "preserved the committed hash. However, none of the seven coupled failures "
        "recovered (required at least two), and five cases repeated the same planned "
        "node set for five or more consecutive turns.",
        "", "## Concrete case findings", "",
        "- `counting_and_probability/765` remains a clean positive result. One planned "
        "two-node edit replaces the vacuous `known_quantities` and "
        "`verification_breakdown`; the obligation curve is `4 → 0`.",
        "- `intermediate_algebra/662` remains strictAccepted. The first identical "
        "two-node batch is rolled back; the next Plan edits only the root dependency "
        "and closes the sole DAG obligation.",
        "- `prealgebra/378` improves over the first implementation attempt and retains "
        "the v6 success. Two Lean-invalid subgraphs are rolled back with concrete "
        "diagnostics; round 3 atomically edits four declarations and closes `5 → 0` "
        "obligations. This is the clearest evidence that compact rollback diagnostics help.",
        "- `precalculus/1056` makes formal changes but regresses semantically. It adds "
        "`volume_region_claim` with the interior inequality, yet the root still concludes "
        "the boundary equation and the Judge ends with six obligations (interior, answer "
        "formulation, root target/grounding and S012 DAG use). Planning coordinated nodes, "
        "but did not preserve one stable global target.",
        "- `cmimc_2025/23` commits six linked intersection/shoelace nodes, but the new "
        "`circle_intersection_*` lemmas do not replace or semantically bind the original "
        "coordinate definitions. Nine old/new obligations coexist; subsequent rounds are "
        "discarded because one planned declaration is identical, demonstrating that exact "
        "Plan coverage plus all-or-nothing validation can deadlock a partially-correct batch.",
        "- `counting_and_probability/430` eventually commits four nodes, but its root "
        "still contains the reflexive conjunct `(1/6) = (1/6)` and the probability process "
        "is represented by cardinality facts rather than a probability space. Static "
        "rollback blocks several worse roots, while the semantic obligations remain 5.",
        "- `aime_2024/81` compiles increasingly large three-node replacements, but formal "
        "conditions such as an existential `chord` for any pair are too weak to encode "
        "rectangle sides. The decompiler/Judge still sees a vacuous S002 node, missing "
        "rectangle construction, and ungrounded root despite five commits.",
        "- `geometry/434` is the nearest non-control improvement: 18 edits reduce the "
        "active inventory to three defects. The remaining `angle_at_B` type does not "
        "formally encode the consecutive-interior-angle theorem, and `x_interpretation` "
        "still fails to bind x to the diagram angle. Natural-language comments cannot "
        "satisfy the formal decompiler.",
        "- `counting_and_probability/731` commits five multi-node subgraphs but holds at "
        "eight obligations. Axis alignment, perpendicular-bisector rhombus construction, "
        "specific diagonals, and the probability target remain encoded only weakly or in "
        "comments; repeated rewrites do not change the Judge-visible semantics.",
        "- `hmmt_feb_2025/18` performs no commit: every five-node batch contains at least "
        "one identical replacement, so atomic validation rejects all useful co-edits too. "
        "Its original five obligations therefore remain unchanged.",
        "", "## Interpretation", "",
        "The Planner improves coordination and the subgraph interface can repair coupled "
        "controls, but the experiment does not show new repair capability on the seven "
        "hard cases. The dominant limiter is now the contract between Plan and Editor: "
        "the Planner names every node conceptually involved, while the Editor often leaves "
        "one unchanged; exact set equality plus `identicalReplacement` then rejects the "
        "whole atomic batch. Where commits occur, the second limiter is semantic target "
        "instability: successive globally valid candidates can add or reopen obligations "
        "because this iteration intentionally has no semantic rollback.",
    ])
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

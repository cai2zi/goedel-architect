from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from typing import Any


WORK = Path("/ssd/czx/czx_work/cot_blueprint_refine")
RUNS = {
    "planDirectTurn8": WORK / "qwen3_8b_397b_wrong10_step_v11_plan_direct_search_turn8_smoke",
    "directEditTurn8": WORK / "qwen3_8b_397b_wrong10_step_v11_direct_edit_search_turn8_smoke",
    "planDirectTurn16": WORK / "qwen3_8b_397b_wrong10_step_v11_plan_direct_search_turn16_smoke",
    "directEditTurn16": WORK / "qwen3_8b_397b_wrong10_step_v11_direct_edit_search_turn16_smoke",
}
OUT = WORK / "qwen3_8b_397b_wrong10_step_v11_search_comparison_report"
ACCEPTED = {"strictAccepted", "acceptedWithJustifiedSideBranches"}
CONTROLS = {
    "MATH-500/test/counting_and_probability/765.json",
    "MATH-500/test/prealgebra/378.json",
    "MATH-500/test/intermediate_algebra/662.json",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def elapsed_seconds(root: Path) -> float | None:
    path = root / "kimina" / "session.json"
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    start, end = row.get("ready_at") or row.get("started_at"), row.get("stopped_at")
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() if start and end else None


def comparator_defects(comparator: dict[str, Any]) -> int:
    total = 0
    for step in comparator.get("steps") or ():
        total += sum(len(step.get(field) or ()) for field in (
            "missing_clauses", "weakened_clauses", "unbound_objects", "wrong_relations"
        ))
    root = comparator.get("root") or {}
    total += int(not bool(root.get("target_object_preserved")))
    total += int(not bool(root.get("answer_grounded")))
    total += sum(not bool(x.get("justified_side_branch")) for x in comparator.get("unreachable_steps") or ())
    total += len(comparator.get("dependency_issues") or ())
    return total


def final_defects(row: dict[str, Any]) -> int | None:
    audit = ((row.get("phase1b_validation") or {}).get("semanticAudit") or {})
    decompiler = audit.get("formalDecompiler") or {}
    comparator = audit.get("strictComparator") or {}
    if not comparator:
        return None
    return comparator_defects(comparator) + sum(
        node.get("semantic_effect") == "vacuous" for node in decompiler.get("nodes") or ()
    )


def summarize_case(row: dict[str, Any]) -> dict[str, Any]:
    history = list(row.get("phase1b_edit_history") or ())
    attempts = [attempt for turn in history for attempt in turn.get("attempts") or ()]
    trace = read_jsonl(Path(row["trace_path"]))
    kinds = Counter(str(event.get("kind") or "") for event in trace)
    tokens: Counter[str] = Counter()
    calls: Counter[str] = Counter()
    search_queries = search_results = cache_hits = 0
    length_finishes = 0
    for event in trace:
        kind, args = str(event.get("kind") or ""), event.get("args") or {}
        if kind == "llm_request_end" and event.get("ok"):
            phase = str(args.get("phase") or "unknown")
            tokens[phase] += int(args.get("total_tokens") or 0)
            calls[phase] += 1
            length_finishes += int(args.get("finish_reason") == "length")
        if kind == "phase1BMathlibSearchStart":
            search_queries += int(args.get("queryCount") or 0)
        elif kind == "phase1BMathlibSearchResult":
            search_results += int(args.get("resultCount") or 0)
            cache_hits += int(bool(args.get("cacheHit")))
    commits = [turn for turn in history if turn.get("committed")]
    search_attempts = [attempt for attempt in attempts if attempt.get("mathlibSearchUsed")]
    search_commits = sum(
        bool(turn.get("committed")) and any(a.get("mathlibSearchUsed") for a in turn.get("attempts") or ())
        for turn in history
    )
    assessments = [
        attempt.get("commitAssessment") or {}
        for attempt in attempts if attempt.get("commitAssessment")
    ]
    first = assessments[0] if assessments else {}
    return {
        "source_id": row["source_id"],
        "status": row.get("status"),
        "turns": len(history),
        "attempts": len(attempts),
        "commits": len(commits),
        "hard_failures": sum(bool(a.get("hardErrors")) for a in attempts),
        "stable_failures": sum(not bool((a.get("stableGate") or {}).get("passed", True)) for a in attempts),
        "semantic_gate_failures": sum(
            (a.get("commitAssessment") or {}).get("decision") not in (None, "commit") for a in attempts
        ),
        "search_turns": kinds["phase1BMathlibSearchStart"],
        "search_queries": search_queries,
        "search_results": search_results,
        "search_cache_hits": cache_hits,
        "search_commits": search_commits,
        "search_attempts": len(search_attempts),
        "deferred_edits": kinds["phase1BEditDeferredUntilAfterSearch"],
        "length_finishes": length_finishes,
        "initial_strict_defects": first.get("strictDefectBefore"),
        "final_strict_defects": final_defects(row),
        "tokens": sum(tokens.values()),
        "tokens_by_phase": dict(tokens),
        "llm_calls": sum(calls.values()),
        "calls_by_phase": dict(calls),
        "trace_path": row.get("trace_path"),
        "blueprint_dir": row.get("blueprint_dir"),
    }


def summarize_run(label: str, root: Path) -> dict[str, Any]:
    rows = read_jsonl(root / "robustpa" / "blueprint" / "results.jsonl")
    cases = [summarize_case(row) for row in rows]
    accepted = {case["source_id"] for case in cases if case["status"] in ACCEPTED}
    return {
        "summary": {
            "label": label,
            "root": str(root),
            "total": len(cases),
            "statuses": dict(Counter(case["status"] for case in cases)),
            "accepted": sorted(accepted),
            "controls_kept": sorted(accepted & CONTROLS),
            "difficult_recovered": sorted(accepted - CONTROLS),
            "turns": sum(case["turns"] for case in cases),
            "attempts": sum(case["attempts"] for case in cases),
            "commits": sum(case["commits"] for case in cases),
            "search_turns": sum(case["search_turns"] for case in cases),
            "search_queries": sum(case["search_queries"] for case in cases),
            "search_results": sum(case["search_results"] for case in cases),
            "search_cache_hits": sum(case["search_cache_hits"] for case in cases),
            "search_commits": sum(case["search_commits"] for case in cases),
            "length_finishes": sum(case["length_finishes"] for case in cases),
            "tokens": sum(case["tokens"] for case in cases),
            "llm_calls": sum(case["llm_calls"] for case in cases),
            "duration_seconds": elapsed_seconds(root),
        },
        "cases": cases,
    }


def main() -> None:
    runs = {label: summarize_run(label, root) for label, root in RUNS.items()}
    incomplete = {label: run["summary"]["total"] for label, run in runs.items() if run["summary"]["total"] != 10}
    if incomplete:
        raise RuntimeError(f"incomplete v11 runs: {incomplete}")
    payload = {
        "runs": runs,
        "decision": {
            "preferred_arm": "directEditTurn16",
            "quality_reason": "It is the only arm to recover a difficult case (cmimc_2025/23), and accepted 3/10.",
            "production_ready": False,
            "production_reason": "Only 1/7 difficult cases recovered; search was used heavily but the recovered difficult case used no search.",
            "search_policy": "Keep the optional tool, but do not make it the default: directEdit selected it in nearly every turn with a very low search-to-commit ratio.",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    order = list(RUNS)
    by_run = {label: {case["source_id"]: case for case in run["cases"]} for label, run in runs.items()}
    ids = sorted(next(iter(by_run.values())))
    lines = [
        "# Phase 1B v11 Mathlib Search 四臂报告", "",
        "## 总览", "",
        "| arm | accepted | controls | difficult | semantic/structural/infra | turns | attempts | commits | search turns/queries/results | cache | search commits | tokens | calls | duration |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in order:
        s, statuses = runs[label]["summary"], runs[label]["summary"]["statuses"]
        duration = s["duration_seconds"] or 0
        lines.append(
            f"| {label} | {len(s['accepted'])}/10 | {len(s['controls_kept'])}/3 | {len(s['difficult_recovered'])}/7 | "
            f"{statuses.get('semanticRejected', 0)}/{statuses.get('structuralRejected', 0)}/{statuses.get('infraError', 0)} | "
            f"{s['turns']} | {s['attempts']} | {s['commits']} | {s['search_turns']}/{s['search_queries']}/{s['search_results']} | "
            f"{s['search_cache_hits']} | {s['search_commits']} | {s['tokens']} | {s['llm_calls']} | {duration:.1f}s |"
        )
    lines += ["", "## 逐样例", "", "| source_id | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
    for source_id in ids:
        cells = []
        for label in order:
            case = by_run[label][source_id]
            cells.append(f"{case['status']} ({case['initial_strict_defects']}→{case['final_strict_defects']}; search {case['search_turns']})")
        lines.append(f"| {source_id} | " + " | ".join(cells) + " |")
    lines += [
        "", "## 结论", "",
        "- `directEditTurn16` 最好：3/10 accepted，并恢复唯一困难样例 `cmimc_2025/23`；其余三臂没有恢复困难样例。质量仍未达到扩展完整 76 条的门槛。",
        "- 额外轮次只对 directEdit 有一次实际收益。planDirect 从 8 增至 16 turn 后仍为 1/10，token 约增加 89%；directEdit 从 2/10 增至 3/10，token 约增加 35%。",
        "- Search 的使用高度依赖 strategy：planDirect 分别只搜索 4/80、7/158 个 turn；directEdit 搜索 66/69、113/116 个 turn。后者把搜索近似当成默认前置步骤。",
        "- Search 后提交率很低：四臂分别为 1/4、4/66、2/7、3/113。唯一恢复的困难样例 `cmimc_2025/23` 在 directEditTurn16 中没有搜索，而是直接同时修改 `compute_EF`、`compute_GH`、`shoelace_vertices` 和 root，把 S001–S003 接入 root closure。",
        "- 40/40 case-run terminal，0 infra error；Search start/end 完整配对。出现 6 次非终局 `finish_reason=length`（directEditTurn8 1 次、planDirectTurn16 5 次），都被后续重试/轮次吸收，没有形成 terminal format/truncation error。",
        "- 因此 Mathlib search 应保留为按需辅助，但当前 prompt 还需收紧触发条件；现阶段不建议直接运行四个完整 76 条实验，优先进一步限制 directEdit 的无目的搜索。",
        "", "完整逐 case 统计见 `report.json`。", "",
    ]
    (OUT / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()

from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from typing import Any


WORK = Path("/ssd/czx/czx_work/cot_blueprint_refine")
BASELINE = WORK / "qwen3_8b_397b_wrong10_step_v11_direct_edit_search_turn8_smoke"
CURRENT = WORK / "qwen3_8b_397b_wrong10_step_v12_direct_edit_search_lean_errors_turn8"
OUT = WORK / "qwen3_8b_397b_wrong10_step_v12_search_policy_comparison_report"
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
    start = row.get("ready_at") or row.get("started_at")
    end = row.get("stopped_at")
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() if start and end else None


def comparator_defects(comparator: dict[str, Any]) -> int:
    total = 0
    for step in comparator.get("steps") or ():
        total += sum(len(step.get(field) or ()) for field in (
            "missing_clauses", "weakened_clauses", "unbound_objects", "wrong_relations"
        ))
    root = comparator.get("root") or {}
    total += int(root.get("target_object_preserved") is False)
    total += int(root.get("answer_grounded") is False)
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
    tokens = calls = queries = results = cache_hits = length_finishes = 0
    eligible_checks = eligible_hits = 0
    eligibility_reasons: Counter[str] = Counter()
    for event in trace:
        kind, args = str(event.get("kind") or ""), event.get("args") or {}
        if kind == "llm_request_end" and event.get("ok"):
            tokens += int(args.get("total_tokens") or 0)
            calls += 1
            length_finishes += int(args.get("finish_reason") == "length")
        elif kind == "phase1BMathlibSearchEligibility":
            eligible_checks += 1
            eligible_hits += int(bool(args.get("eligible")))
            for reason in args.get("reasons") or ():
                eligibility_reasons[str(reason)] += 1
        elif kind == "phase1BMathlibSearchStart":
            queries += int(args.get("queryCount") or 0)
        elif kind == "phase1BMathlibSearchResult":
            results += int(args.get("resultCount") or 0)
            cache_hits += int(bool(args.get("cacheHit")))
    assessments = [
        attempt.get("commitAssessment") or {}
        for attempt in attempts if attempt.get("commitAssessment")
    ]
    first = assessments[0] if assessments else {}
    hard_errors = Counter(
        str(error).split(":", 1)[0]
        for attempt in attempts for error in attempt.get("hardErrors") or ()
    )
    stable_errors = Counter(
        str(error).split(":", 1)[0]
        for attempt in attempts for error in (attempt.get("stableGate") or {}).get("errors") or ()
    )
    assessment_errors = Counter(
        str(error).split(":", 1)[0]
        for assessment in assessments for error in assessment.get("errors") or ()
    )
    return {
        "source_id": row["source_id"],
        "status": row.get("status"),
        "turns": len(history),
        "attempts": len(attempts),
        "commits": sum(bool(turn.get("committed")) for turn in history),
        "hard_errors": dict(hard_errors),
        "stable_errors": dict(stable_errors),
        "semantic_gate_errors": dict(assessment_errors),
        "eligibility_checks": eligible_checks,
        "eligibility_hits": eligible_hits,
        "eligibility_reasons": dict(eligibility_reasons),
        "search_turns": kinds["phase1BMathlibSearchStart"],
        "search_queries": queries,
        "search_results": results,
        "search_cache_hits": cache_hits,
        "length_finishes": length_finishes,
        "initial_strict_defects": first.get("strictDefectBefore"),
        "final_strict_defects": final_defects(row),
        "tokens": tokens,
        "llm_calls": calls,
        "trace_path": row.get("trace_path"),
        "blueprint_dir": row.get("blueprint_dir"),
    }


def summarize_run(label: str, root: Path) -> dict[str, Any]:
    cases = [summarize_case(row) for row in read_jsonl(root / "robustpa" / "blueprint" / "results.jsonl")]
    accepted = {case["source_id"] for case in cases if case["status"] in ACCEPTED}
    aggregate = Counter()
    for case in cases:
        for field in ("hard_errors", "stable_errors", "semantic_gate_errors", "eligibility_reasons"):
            aggregate.update({f"{field}:{key}": value for key, value in case[field].items()})
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
            "eligibility_checks": sum(case["eligibility_checks"] for case in cases),
            "eligibility_hits": sum(case["eligibility_hits"] for case in cases),
            "search_turns": sum(case["search_turns"] for case in cases),
            "search_queries": sum(case["search_queries"] for case in cases),
            "search_results": sum(case["search_results"] for case in cases),
            "search_cache_hits": sum(case["search_cache_hits"] for case in cases),
            "length_finishes": sum(case["length_finishes"] for case in cases),
            "tokens": sum(case["tokens"] for case in cases),
            "llm_calls": sum(case["llm_calls"] for case in cases),
            "duration_seconds": elapsed_seconds(root),
            "diagnostics": dict(aggregate),
        },
        "cases": cases,
    }


def main() -> None:
    runs = {
        "v11SearchOpen": summarize_run("v11SearchOpen", BASELINE),
        "v12LeanErrorsOnly": summarize_run("v12LeanErrorsOnly", CURRENT),
    }
    if any(run["summary"]["total"] != 10 for run in runs.values()):
        raise RuntimeError("expected 10 terminal results in both runs")
    payload = {"runs": runs}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    old, new = runs["v11SearchOpen"]["summary"], runs["v12LeanErrorsOnly"]["summary"]
    old_cases = {x["source_id"]: x for x in runs["v11SearchOpen"]["cases"]}
    new_cases = {x["source_id"]: x for x in runs["v12LeanErrorsOnly"]["cases"]}
    reduction = 100 * (1 - new["search_turns"] / old["search_turns"]) if old["search_turns"] else 0
    lines = [
        "# Phase 1B v12 Search Policy 对比", "",
        "## 总览", "",
        "| run | accepted | controls | difficult | statuses | commits | eligibility hit/check | search turns/queries/results | tokens | calls | duration |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("v11SearchOpen", "v12LeanErrorsOnly"):
        s = runs[label]["summary"]
        lines.append(
            f"| {label} | {len(s['accepted'])}/10 | {len(s['controls_kept'])}/3 | {len(s['difficult_recovered'])}/7 | "
            f"{s['statuses']} | {s['commits']} | {s['eligibility_hits']}/{s['eligibility_checks']} | "
            f"{s['search_turns']}/{s['search_queries']}/{s['search_results']} | {s['tokens']} | {s['llm_calls']} | "
            f"{(s['duration_seconds'] or 0):.1f}s |"
        )
    lines += ["", "## 逐样例", "", "| source_id | v11 | v12 | v12 search | v12 commits |", "|---|---|---|---:|---:|"]
    for source_id in sorted(new_cases):
        before, after = old_cases[source_id], new_cases[source_id]
        lines.append(
            f"| {source_id} | {before['status']} ({before['final_strict_defects']}) | "
            f"{after['status']} ({after['final_strict_defects']}) | {after['search_turns']} | {after['commits']} |"
        )
    lines += [
        "", "## 结论", "",
        f"- `leanErrorsOnly` 将实际搜索从 {old['search_turns']} 次降至 {new['search_turns']} 次（减少 {reduction:.1f}%）；"
        f"共检查 {new['eligibility_checks']} 个 Editor attempt，仅 {new['eligibility_hits']} 次命中资格。",
        "- 资格只由候选 retry 中的 `synthesisFailure` 和 `invalidFieldNotation` 触发；语义 obligation、DAG、Pending、普通类型不匹配没有开放搜索。",
        f"- 本次 v12 为 {len(new['accepted'])}/10 accepted，低于 v11 的 {len(old['accepted'])}/10；三个控制样例全部回归。"
        "由于两次实验均从 Phase 1A 随机生成，不能把该质量差异单独归因于搜索门控，但它说明门控目前只解决了无目的搜索开销，并未改善 Editor 的语义/格式稳定性。",
        "- v12 的失败集中于 Editor 重复提交已拒绝 candidate/no-op，以及最终 Stable/Semantic gate 无法收敛；搜索触发集中在 counting/430，没有转化成 accepted。",
        "- 本轮没有运行 16-turn。", "",
    ]
    (OUT / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()

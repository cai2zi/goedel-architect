from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


WORK = Path("/ssd/czx/czx_work/cot_blueprint_refine")
V5 = WORK / "qwen3_8b_397b_wrong76_step_v5_phase1_ab_semantic_judge" / "robustpa" / "blueprint"
AUDIT = WORK / "qwen3_8b_397b_wrong76_step_v6_semantic_audit44"
REPAIR = WORK / "qwen3_8b_397b_wrong22_step_v6_phase1b_repair" / "robustpa" / "blueprint"
REPORT = WORK / "qwen3_8b_397b_wrong76_step_v6_report"

NEGATIVES = {
    "MATH-500/test/counting_and_probability/765.json",
    "MATH-500/test/counting_and_probability/731.json", "aime_2025/14",
    "cmimc_2025/23", "MATH-500/test/precalculus/1056.json",
    "MATH-500/test/geometry/434.json", "MATH-500/test/prealgebra/1865.json",
    "MATH-500/test/prealgebra/378.json", "cmimc_2025/38", "hmmt_feb_2025/16",
    "hmmt_feb_2025/18", "MATH-500/test/counting_and_probability/430.json",
    "aime_2024/81", "aime_2025/20",
}
RISKS = {"cmimc_2025/5", "cmimc_2025/30", "aime_2024/62"}
CONTROLS = {
    "cmimc_2025/9", "brumo_2025/6",
    "MATH-500/test/intermediate_algebra/662.json", "aime_2025/15", "brumo_2025/3",
}
ACCEPTED = {"strictAccepted", "acceptedWithJustifiedSideBranches"}


def rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def events(path: Path) -> list[dict[str, Any]]:
    return rows(path)


def comparator_defect_inventory(event: dict[str, Any]) -> list[str]:
    result = ((event.get("args") or {}).get("result") or {})
    defects: list[str] = []
    for step in result.get("steps") or []:
        step_id = step.get("step_id") or "?"
        for category in ("missing_clauses", "weakened_clauses", "unbound_objects", "wrong_relations"):
            for issue in step.get(category) or []:
                nodes = ",".join(issue.get("node_names") or []) or "<none>"
                defects.append(f"{category}:{step_id}:{nodes}")
    root = result.get("root") or {}
    if root and not root.get("target_object_preserved", False):
        defects.append("root:targetObjectNotPreserved")
    if root and not root.get("answer_grounded", False):
        defects.append("root:answerNotGrounded")
    for issue in result.get("unreachable_steps") or []:
        if not issue.get("justified_side_branch", False):
            defects.append(f"dagDisconnected:{issue.get('step_id')}")
    for issue in result.get("dependency_issues") or []:
        defects.append(f"dependency:{issue.get('step_id')}:{issue.get('node_name')}")
    return defects


def main() -> None:
    audit_rows = rows(AUDIT / "results.jsonl")
    repair_rows = rows(REPAIR / "results.jsonl")
    v5_rows = {row["source_id"]: row for row in rows(V5 / "results.jsonl")}
    audit_by = {row["source_id"]: row for row in audit_rows}
    cases = []
    for row in repair_rows:
        sid = row["source_id"]
        trace = events(Path(row["trace_path"]))
        compare_results = [event for event in trace if event.get("kind") == "phase1BStrictCompareResult"]
        obligation_updates = [event for event in trace if event.get("kind") == "phase1BSemanticObligationsUpdated"]
        decompiles = [event for event in trace if event.get("kind") == "phase1BFormalDecompileEnd"]
        searches = [event for event in trace if event.get("kind") == "phase1BMathlibSearchResult"]
        edit_history = row.get("phase1b_edit_history") or []
        request_ends = [event for event in trace if event.get("kind") == "llm_request_end"]
        judge_ends = [event for event in request_ends if (event.get("args") or {}).get("operation") in {
            "phase1b_formal_decompiler", "phase1b_strict_comparator",
        }]
        edit_calls = [event for event in trace if event.get("kind") == "tool_call" and event.get("tool_name") == "editBlueprintNode"]
        search_calls = [event for event in trace if event.get("kind") == "tool_call" and event.get("tool_name") == "mathlib_search"]
        pass_curve = [bool((event.get("args") or {}).get("passed")) for event in compare_results]
        open_curve = [int((event.get("args") or {}).get("openObligationCount") or 0) for event in obligation_updates]
        cases.append({
            "source_id": sid,
            "cohort": "negative" if sid in NEGATIVES else "risk" if sid in RISKS else "control",
            "v5_status": (v5_rows.get(sid) or {}).get("status"),
            "v6_status": row.get("status"),
            "strict_compare_pass_curve": pass_curve,
            "open_obligation_curve": open_curve,
            "first_defects": comparator_defect_inventory(compare_results[0]) if compare_results else [],
            "final_defects": comparator_defect_inventory(compare_results[-1]) if compare_results else [],
            "fail_to_pass": bool(pass_curve and not pass_curve[0] and pass_curve[-1]),
            "max_open_obligations": max(open_curve, default=0),
            "resolved_obligation_events": sum(
                len((event.get("args") or {}).get("resolvedObligationIds") or [])
                for event in obligation_updates
            ),
            "formal_decompiler_calls": sum(not bool((event.get("args") or {}).get("cacheHit")) for event in decompiles),
            "formal_decompiler_cache_hits": sum(bool((event.get("args") or {}).get("cacheHit")) for event in decompiles),
            "strict_comparator_calls": len(compare_results),
            "search_calls": len(search_calls),
            "search_to_edit": bool(search_calls and edit_calls),
            "first_search": ((search_calls[0].get("args") or {}).get("arguments") or {}).get("query", "") if search_calls else "",
            "judge_tokens": sum(int((event.get("args") or {}).get("total_tokens") or 0) for event in judge_ends),
            "all_llm_tokens": sum(int((event.get("args") or {}).get("total_tokens") or 0) for event in request_ends),
            "edit_rounds": len(edit_history),
            "accepted_edits": sum(len(item.get("accepted") or []) for item in edit_history),
            "rejected_edits": sum(len(item.get("rejected") or []) for item in edit_history),
            "final_validation": row.get("phase1b_validation") or {},
            "error": row.get("error") or "",
        })
    summary = {
        "audit44": {
            "total": len(audit_rows),
            "statusCounts": dict(Counter(row.get("status") for row in audit_rows)),
            "negativeRejected": sorted(sid for sid in NEGATIVES if audit_by.get(sid, {}).get("status") == "semanticRejected"),
            "negativeMissed": sorted(sid for sid in NEGATIVES if audit_by.get(sid, {}).get("status") != "semanticRejected"),
            "infraErrors": sorted(row["source_id"] for row in audit_rows if row.get("status") == "infraError"),
        },
        "repair22": {
            "total": len(repair_rows),
            "statusCounts": dict(Counter(row.get("status") for row in repair_rows)),
            "negativeRecovered": sorted(row["source_id"] for row in repair_rows if row["source_id"] in NEGATIVES and row.get("status") in ACCEPTED),
            "negativeStillRejected": sorted(row["source_id"] for row in repair_rows if row["source_id"] in NEGATIVES and row.get("status") not in ACCEPTED),
            "controlsAccepted": sorted(row["source_id"] for row in repair_rows if row["source_id"] in CONTROLS and row.get("status") in ACCEPTED),
            "controlsRegressed": sorted(row["source_id"] for row in repair_rows if row["source_id"] in CONTROLS and row.get("status") not in ACCEPTED),
            "failToPass": sorted(case["source_id"] for case in cases if case["fail_to_pass"]),
            "searchToEdit": sorted(case["source_id"] for case in cases if case["search_to_edit"]),
            "judgeTokens": sum(case["judge_tokens"] for case in cases),
            "allLlmTokens": sum(case["all_llm_tokens"] for case in cases),
        },
        "cases": sorted(cases, key=lambda item: item["source_id"]),
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Phase 1 v6 experiment report", "",
        f"- Offline audit: {summary['audit44']['total']} cases; {summary['audit44']['statusCounts']}",
        f"- High-confidence negatives rejected: {len(summary['audit44']['negativeRejected'])}/14",
        f"- Offline format/infra errors: {len(summary['audit44']['infraErrors'])}",
        f"- Seed repair: {summary['repair22']['total']} cases; {summary['repair22']['statusCounts']}",
        f"- Negative repairs accepted: {len(summary['repair22']['negativeRecovered'])}/14",
        f"- Positive controls accepted: {len(summary['repair22']['controlsAccepted'])}/5",
        f"- FAIL→PASS cases: {len(summary['repair22']['failToPass'])}; search→edit cases: {len(summary['repair22']['searchToEdit'])}",
        f"- Judge tokens: {summary['repair22']['judgeTokens']}; all Phase-1B LLM tokens: {summary['repair22']['allLlmTokens']}",
        "- Production gate: **FAIL** — only 2/14 high-confidence negatives were repaired (required >=5/14).", "",
        "## Concrete findings", "",
        "- `counting_and_probability/765`: FAIL→PASS in one edit round. `known_quantities` changed from a chain of `let` bindings ending in `True` to the six asserted equalities, and `individual_counts` similarly changed from `True` to the seven claimed region counts. All four obligations closed.",
        "- `prealgebra/378`: search→edit→PASS. Search targeted triangle area/`MeasureTheory.volume`; the root changed from `let area_shaded := 6; area_shaded = 6` to `MeasureTheory.volume shaded_region = 6`, while `total_shaded_area` now identifies the union with the containing triangle. This removed the answer alias and bound the answer to the modeled region.",
        "- `intermediate_algebra/662` (control): FAIL→PASS after adding the omitted transformation nodes to the root dependency list. The formal target was unchanged; the repair was a DAG-use correction.",
        "- `precalculus/1056`: correctly remained rejected. The offline audit isolates S011: COT describes the enclosed interior `x²+y²+z² ≤ 36`, but the Blueprint only formalizes the boundary `= 36`. Sixteen searches/eleven accepted edits did not eliminate the two persistent obligations.",
        "- `aime_2024/81`: correctly rejected rather than accepting matching answer literals. S002 formalizes points on a unit circle but not that diagonals are diameters; S003 constructs endpoint sets without rectangle/uniqueness constraints. Subsequent edits never produced a structurally valid eligible candidate.",
        "- `cmimc_2025/35` (offline wrong-COT control): rejection was translation-specific—S007 used a vacuous sum reindexing and S001–S010 were disconnected from root. No defect asserted that the COT arithmetic itself was false.",
        "- Reachability risks did not recover: `cmimc_2025/5`, `cmimc_2025/30`, and `aime_2024/62` all ended semanticRejected. Comparator therefore did not silently reinterpret required-path disconnections as side branches.",
        "- Positive-control regression remains: `aime_2025/15` and `brumo_2025/3` were rejected. The stricter audit exposed real semantic obligations, but current node-edit repair could not discharge them within 16 rounds.", "",
        "## Per-case repair trajectory", "",
        "| source_id | cohort | v6 status | comparator pass curve | open obligations | edits | searches | judge tokens |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for case in summary["cases"]:
        lines.append(
            f"| {case['source_id']} | {case['cohort']} | {case['v6_status']} | "
            f"{case['strict_compare_pass_curve']} | {case['open_obligation_curve']} | "
            f"{case['accepted_edits']} | {case['search_calls']} | {case['judge_tokens']} |"
        )
    (REPORT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT / "report.md")


if __name__ == "__main__":
    main()

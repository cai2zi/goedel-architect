from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


WORK = Path("/ssd/czx/czx_work/cot_blueprint_refine")
V7 = WORK / "qwen3_8b_397b_wrong10_step_v7_phase1b_plan_subgraph" / "robustpa" / "blueprint"
V7_REPORT = WORK / "qwen3_8b_397b_wrong10_step_v7_phase1b_plan_subgraph_report" / "report.json"
V8 = WORK / "qwen3_8b_397b_wrong10_step_v8_phase1b_repair_spec" / "robustpa" / "blueprint"
OUT = WORK / "qwen3_8b_397b_wrong10_step_v8_phase1b_repair_spec_report"

CONTROLS = {
    "MATH-500/test/counting_and_probability/765.json",
    "MATH-500/test/prealgebra/378.json",
    "MATH-500/test/intermediate_algebra/662.json",
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def final_metrics(row: dict) -> dict:
    audit = ((row.get("phase1b_validation") or {}).get("semanticAudit") or {})
    comparator = audit.get("strictComparator") or {}
    steps = comparator.get("steps") or []
    root = comparator.get("root") or {}
    counts = Counter()
    for step in steps:
        counts["missingClauses"] += len(step.get("missing_clauses") or ())
        counts["weakenedClauses"] += len(step.get("weakened_clauses") or ())
        counts["unboundObjects"] += len(step.get("unbound_objects") or ())
        counts["wrongRelations"] += len(step.get("wrong_relations") or ())
    counts["dependencyIssues"] = len(comparator.get("dependency_issues") or ())
    counts["requiredPathDisconnections"] = sum(
        not bool(item.get("justified_side_branch"))
        for item in (comparator.get("unreachable_steps") or ())
    )
    return {
        "openObligations": len(audit.get("openObligations") or ()),
        **dict(counts),
        "rootTargetObject": root.get("target_object_preserved"),
        "rootAnswerGrounding": root.get("answer_grounded"),
    }


def longest_equal_hash_run(history: list[dict]) -> int:
    longest = current = 0
    previous = None
    for item in history:
        value = item.get("repairSpecHash")
        current = current + 1 if value and value == previous else 1
        previous = value
        longest = max(longest, current)
    return longest


def analyze_case(row: dict, v7_case: dict) -> dict:
    trace = read_jsonl(Path(row["trace_path"]))
    history = list(row.get("phase1b_edit_history") or ())
    usage = Counter()
    length_finishes = 0
    length_finish_phases = Counter()
    for event in trace:
        if event.get("kind") != "llm_request_end" or not event.get("ok"):
            continue
        args = event.get("args") or {}
        usage[str(args.get("phase") or "unknown")] += int(args.get("total_tokens") or 0)
        if str(args.get("finish_reason") or "").lower() == "length":
            length_finishes += 1
            length_finish_phases[str(args.get("phase") or "unknown")] += 1

    commits = [event for event in trace if event.get("kind") == "phase1BSubgraphCommit"]
    rollbacks = [event for event in trace if event.get("kind") == "phase1BSubgraphRollback"]
    no_ops = [event for event in trace if event.get("kind") == "phase1BNoOpFiltered"]
    deltas = [event for event in trace if event.get("kind") == "phase1BSemanticDelta"]
    stagnation = [event for event in trace if event.get("kind") == "phase1BSemanticStagnation"]
    rollback_reasons = [
        reason
        for item in history
        for reason in (item.get("rollbackReasons") or ())
    ]
    format_rollbacks = sum(
        any(token in str(reason) for token in (
            "phase1BPlan", "unknown", "requires", "actionMismatch",
            "repairSpec", "duplicateRepairSpec",
        ))
        for item in history
        for reason in [" | ".join(str(x) for x in item.get("rollbackReasons") or ())]
        if reason
    )
    effective_sizes = [len(item.get("effectiveNodes") or ()) for item in history]
    metrics = final_metrics(row)
    return {
        "source_id": row["source_id"],
        "v7_status": v7_case.get("v7_status"),
        "v8_status": row.get("status"),
        "error": row.get("error", ""),
        "turns": len(history),
        "commits": len(commits),
        "rollbacks": len(rollbacks),
        "rollbackHashPreserved": all(
            (event.get("args") or {}).get("committedHashBefore")
            == (event.get("args") or {}).get("committedHashAfter")
            for event in rollbacks
        ),
        "plannedEdits": sum(len(item.get("plannedNodes") or ()) for item in history),
        "actualEdits": sum(len(item.get("actualNodes") or ()) for item in history),
        "effectiveEdits": sum(effective_sizes),
        "noOpEdits": sum(len(item.get("noOpNodes") or ()) for item in history),
        "maxEffectiveSubgraph": max(effective_sizes or [0]),
        "multiNodeEffectiveCommit": any(
            bool(item.get("committed")) and len(item.get("effectiveNodes") or ()) >= 2
            for item in history
        ),
        "identicalWholeBatchRollbacks": sum(
            "identicalReplacement" in str(reason) for reason in rollback_reasons
        ),
        "repairSpecFormatRollbacks": format_rollbacks,
        "longestRepeatedRepairSpec": longest_equal_hash_run(history),
        "semanticStagnationEvents": len(stagnation),
        "semanticDeltaCount": len(deltas),
        "openObligationCurve": [
            len((((item.get("semanticDelta") or {}).get("stillOpenTargetObligations")) or ()))
            for item in history if item.get("semanticDelta") is not None
        ],
        "v7FinalOpenObligations": int(v7_case.get("final_open_obligations") or 0),
        "finalMetrics": metrics,
        "tokensByPhase": dict(usage),
        "totalTokens": sum(usage.values()),
        "lengthFinishes": length_finishes,
        "lengthFinishPhases": dict(length_finish_phases),
        "history": history,
    }


def main() -> None:
    rows = read_jsonl(V8 / "results.jsonl")
    v7_payload = json.loads(V7_REPORT.read_text(encoding="utf-8"))
    v7_cases = {item["source_id"]: item for item in v7_payload["cases"]}
    cases = [analyze_case(row, v7_cases.get(row["source_id"], {})) for row in rows]
    statuses = Counter(item["v8_status"] for item in cases)
    controls_kept = sorted(
        item["source_id"] for item in cases
        if item["source_id"] in CONTROLS and item["v8_status"] == "strictAccepted"
    )
    hard_recovered = sorted(
        item["source_id"] for item in cases
        if item["source_id"] not in CONTROLS and item["v8_status"] == "strictAccepted"
    )
    lower_obligation_hard = sorted(
        item["source_id"] for item in cases
        if item["source_id"] not in CONTROLS
        and item["finalMetrics"]["openObligations"] < item["v7FinalOpenObligations"]
    )
    total_tokens = sum(item["totalTokens"] for item in cases)
    v7_tokens = int(v7_payload["summary"]["total_tokens"])
    summary = {
        "total": len(cases),
        "statusCounts": dict(statuses),
        "terminalCount": len(cases),
        "infraErrors": [item["source_id"] for item in cases if "infra" in item["error"].lower()],
        "lengthFinishCount": sum(item["lengthFinishes"] for item in cases),
        "lengthFinishPhases": dict(sum(
            (Counter(item["lengthFinishPhases"]) for item in cases), Counter()
        )),
        "controlsKept": controls_kept,
        "controlsRegressed": sorted(CONTROLS - set(controls_kept)),
        "hardCasesRecovered": hard_recovered,
        "hardCasesWithLowerOpenObligations": lower_obligation_hard,
        "multiNodeEffectiveCommitCases": sum(item["multiNodeEffectiveCommit"] for item in cases),
        "identicalWholeBatchRollbacks": sum(item["identicalWholeBatchRollbacks"] for item in cases),
        "noOpEditsFiltered": sum(item["noOpEdits"] for item in cases),
        "rollbackHashesPreserved": all(item["rollbackHashPreserved"] for item in cases),
        "fiveRepeatedRepairSpecCases": [
            item["source_id"] for item in cases if item["longestRepeatedRepairSpec"] >= 5
        ],
        "hmmt18EffectiveCommit": any(
            item["source_id"] == "hmmt_feb_2025/18" and item["commits"] > 0
            for item in cases
        ),
        "totalTokens": total_tokens,
        "v7TotalTokens": v7_tokens,
        "tokenRatioVsV7": total_tokens / v7_tokens if v7_tokens else None,
        "under115PercentTokenCap": total_tokens <= v7_tokens * 1.15,
        "plannerTokens": sum(item["tokensByPhase"].get("phase1BPlanner", 0) for item in cases),
        "editorTokens": sum(item["tokensByPhase"].get("phase1B", 0) for item in cases),
        "judgeTokens": sum(
            item["tokensByPhase"].get("phase1BFormalDecompiler", 0)
            + item["tokensByPhase"].get("phase1BStrictComparator", 0)
            for item in cases
        ),
    }
    gates = {
        "allTerminal": len(cases) == 10,
        "noTerminalInfraOrFormatError": not summary["infraErrors"] and all(
            item["v8_status"] in {"strictAccepted", "semanticRejected"} for item in cases
        ),
        "allControlsKept": len(controls_kept) == 3,
        "twoHardRecoveries": len(hard_recovered) >= 2,
        "hmmt18Committed": summary["hmmt18EffectiveCommit"],
        "noIdenticalWholeBatchRollback": summary["identicalWholeBatchRollbacks"] == 0,
        "fiveMultiNodeCases": summary["multiNodeEffectiveCommitCases"] >= 5,
        "fiveHardObligationImprovements": len(lower_obligation_hard) >= 5,
        "noFiveRepeatedRepairSpecs": not summary["fiveRepeatedRepairSpecCases"],
        "tokenCap": summary["under115PercentTokenCap"],
    }
    summary["acceptanceGates"] = gates
    summary["promotionPassed"] = all(gates.values())
    payload = {"summary": summary, "cases": cases}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Phase 1B v8 RepairSpec pilot report", "",
        "## Summary", "",
        f"- Status: `{dict(statuses)}`; promotion passed: **{summary['promotionPassed']}**.",
        f"- Controls kept: {len(controls_kept)}/3; regressions: `{summary['controlsRegressed']}`.",
        f"- Hard-case recoveries: {len(hard_recovered)}/7; lower open-obligation count: {len(lower_obligation_hard)}/7.",
        f"- HMMT/18 effective commit: {summary['hmmt18EffectiveCommit']}.",
        f"- Multi-node effective commit cases: {summary['multiNodeEffectiveCommitCases']}/10.",
        f"- Filtered no-op edits: {summary['noOpEditsFiltered']}; whole-batch `identicalReplacement` rollbacks: {summary['identicalWholeBatchRollbacks']}.",
        f"- Tokens: v8={total_tokens}, v7={v7_tokens}, ratio={summary['tokenRatioVsV7']:.1%}; under 115% cap: {summary['under115PercentTokenCap']}.",
        f"- Infra errors: {len(summary['infraErrors'])}; recovered request-level length finishes: {summary['lengthFinishCount']} `{summary['lengthFinishPhases']}`.",
        "", "## Acceptance gates", "",
    ]
    lines.extend(f"- `{name}`: {passed}" for name, passed in gates.items())
    lines.extend([
        "", "## Per-case comparison", "",
        "| source_id | v7 | v8 | turns | commit/rollback | effective/no-op | max batch | obligations v7→v8 | root object/answer | tokens |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---:|",
    ])
    for item in cases:
        metrics = item["finalMetrics"]
        lines.append(
            f"| {item['source_id']} | {item['v7_status']} | {item['v8_status']} | "
            f"{item['turns']} | {item['commits']}/{item['rollbacks']} | "
            f"{item['effectiveEdits']}/{item['noOpEdits']} | {item['maxEffectiveSubgraph']} | "
            f"{item['v7FinalOpenObligations']}→{metrics['openObligations']} | "
            f"{metrics['rootTargetObject']}/{metrics['rootAnswerGrounding']} | {item['totalTokens']} |"
        )
    lines.extend([
        "", "## Findings", "",
        "- `intermediate_algebra/662` is the only strict pass. One root edit adds the missing dependency and closes the sole DAG obligation.",
        "- `hmmt/18` no longer deadlocks on an identical replacement: round 6 commits `window_of_influence`, satisfying the explicit mechanical criterion. The edit resolves no target obligation, while earlier four-node attempts fail Lean or contract checks; five semantic obligations remain.",
        "- `counting/430` commits two root edits but resolves no target obligation and introduces a dependency-fidelity defect. Later attempts either omit the planned tool call, violate dependency contracts, or become no-ops; explicit random-selection semantics remain absent.",
        "- `cmimc/23` commits `shoelace_sum1_value` and then a seven-node circle/shoelace rebuild. Three vacuous-node obligations resolve, but root target/grounding defects are introduced and the construction remains unbound; open obligations move only `9→8`.",
        "- `precalculus/1056` resolves one DAG disconnection, then introduces root target/grounding and relation defects. Its later object rebuild still describes a sphere surface rather than preserving the bounded volume-region target; obligations regress `6→7`.",
        "- `geometry/434` commits two five-node object repairs and one local edit, but resolves only one vacuous node while creating new missing-clause/DAG obligations. The formal declarations remain weak substitutes for the actual diagram relations; obligations regress `3→10`.",
        "- `counting/731` obtains one five-node effective commit after no-op filtering, yet resolves no obligation. Later retries either fail Lean or are rejected as repeated RepairSpecs; the rectangle/rhombus/probability target remains ungrounded.",
        "- `aime/81` eventually commits a three-node object rebuild after filtering two no-ops. It lowers the ledger count `6→5`, but the rectangle–diameter construction and root target/answer binding are still missing, so it is not a semantic recovery.",
        "- Control `prealgebra/378` never obtains a valid RepairSpec because each plan names a would-be shared object without adding it to `EDIT_NODES`; `counting/765` repeatedly rewrites two nodes without resolving any target. Both regress from v7 strict acceptance.",
        "", "## Conclusion", "",
        "v8 validates two mechanical changes: identical replacements are filtered without discarding useful co-edits, and HMMT/18 finally commits. It does not validate promotion to the 76-case run. The remaining bottleneck is the executable contract itself: semantic concepts are not reliably materialized as typed Blueprint declarations, and deterministic commit without semantic checkpointing can destroy already-good seeds. The next iteration should not add more Judge detail; it should introduce a typed object/declaration patch format (or a constrained object-rebuild template) and preserve the last strict/best semantic checkpoint for controls.",
    ])
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

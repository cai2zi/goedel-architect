from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


RUNS = {
    "direct": {
        "baseline": Path(
            "/ssd/czx/czx_work/cot_blueprint_refine/"
            "qwen3_8b_397b_wrong76_global_defs_direct_named_t00/"
            "robustpa/blueprint/results.jsonl"
        ),
        "repair": Path(
            "/ssd/czx/czx_work/cot_blueprint_refine/"
            "qwen3_8b_397b_wrong76_global_defs_direct_repair_v1_semrej12_t00/"
            "robustpa/blueprint/results.jsonl"
        ),
    },
    "compact_separate": {
        "baseline": Path(
            "/ssd/czx/czx_work/cot_blueprint_refine/"
            "qwen3_8b_397b_wrong76_global_defs_compact_separate_named_t00_rerun1/"
            "robustpa/blueprint/results.jsonl"
        ),
        "repair": Path(
            "/ssd/czx/czx_work/cot_blueprint_refine/"
            "qwen3_8b_397b_wrong76_global_defs_compact_separate_repair_v1_semrej11_t00/"
            "robustpa/blueprint/results.jsonl"
        ),
    },
}
REPAIR_CODES = {"answerPreassigned", "targetCoverageIncomplete"}
MANUAL_ASSESSMENTS = {
    ("direct", "MATH-500/test/geometry/465.json"): (
        "triggeredAndAccepted", "theta became a binder constrained by the area model; targetCoverageIncomplete was overbroad."
    ),
    ("direct", "MATH-500/test/prealgebra/874.json"): (
        "specificDefectFixedButOtherReject", "preassignment disappeared, but root assumes rather than derives the two angle relations."
    ),
    ("direct", "MATH-500/test/precalculus/768.json"): (
        "triggeredAndAccepted", "root now gives the complete two-solution iff for a bound x."
    ),
    ("direct", "aime_2025/11"): (
        "triggeredAndAccepted", "root and supporting lemmas now assert solution-set completeness before extracting the answer."
    ),
    ("direct", "brumo_2025/3"): (
        "triggeredAndAccepted", "target length is defined by the geometric intersection measure and derived via interval characterization; coverage label was overbroad."
    ),
    ("direct", "cmimc_2025/27"): (
        "triggeredUnresolved", "final geometry is still rigged by constant coordinates and remains answerPreassigned."
    ),
    ("direct", "cmimc_2025/32"): (
        "triggeredUnresolved", "four retained repair rounds ended in canonical Lean failures, so no later semantic audit verified a repair."
    ),
    ("direct", "cmimc_2025/34"): (
        "specificDefectFixedButOtherReject", "a later audit removed both repair labels, but subsequent full regenerations ended in canonical Lean failures."
    ),
    ("direct", "cmimc_2025/39"): (
        "notApplicable", "baseline dependency defect was repaired without either new code."
    ),
    ("direct", "cmimc_2025/40"): (
        "notApplicable", "baseline relation defects were repaired without either new code."
    ),
    ("direct", "cmimc_2025/5"): (
        "notApplicable", "baseline use-chain defect was repaired without either new code."
    ),
    ("direct", "hmmt_feb_2025/19"): (
        "triggeredUnresolved", "probability remains defined as 1/2 over vacuous geometry and event predicates."
    ),
    ("compact_separate", "MATH-500/test/counting_and_probability/765.json"): (
        "triggeredAndAccepted", "physics count became a binder, but accepted intermediates overclaim equations for every P; comparator still has a quantifier false positive."
    ),
    ("compact_separate", "MATH-500/test/geometry/434.json"): (
        "triggeredUnresolved", "final x_geometric still embeds 62."
    ),
    ("compact_separate", "MATH-500/test/geometry/711.json"): (
        "triggeredAndAccepted", "areas are now computed from coordinates and cross products instead of answer-component constants."
    ),
    ("compact_separate", "MATH-500/test/prealgebra/874.json"): (
        "triggeredAndAccepted", "x and the target angle became binders linked by explicit hypotheses, though the geometry remains abstract."
    ),
    ("compact_separate", "aime_2024/85"): (
        "notApplicable", "accepted without either repair code."
    ),
    ("compact_separate", "aime_2025/20"): (
        "triggeredUnresolved", "hardcoded coordinate instance and vacuous angle relation remain; final audit reports incomplete target coverage."
    ),
    ("compact_separate", "aime_2025/27"): (
        "triggeredAndAccepted", "root now binds the geometric sequence and carries area, cosine, and perimeter constraints to the exact answer form."
    ),
    ("compact_separate", "aime_2025/8"): (
        "triggeredAndAccepted", "root now states existence and completeness of the tangency solution set rather than a constant m+n object."
    ),
    ("compact_separate", "cmimc_2025/11"): (
        "triggeredUnresolved", "mechanically accepted false positive: favorable_arrangements remains defined as 1 and its new support lemma is tautological."
    ),
    ("compact_separate", "cmimc_2025/13"): (
        "triggeredUnresolved", "battle predicate still omits Michael and James and guarantees the claimed probability structurally."
    ),
    ("compact_separate", "hmmt_feb_2025/16"): (
        "specificDefectFixedButOtherReject", "preassignment label disappeared, while a reversed disjointness inequality remains correctly rejected."
    ),
}


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["source_id"])] = row
    return rows


def _error_codes(values: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("code") or "") for item in values]


def _rounds(row: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for item in row.get("generation_history") or []:
        validation = item.get("validation") or {}
        semantic = item.get("semanticErrors") or []
        result.append({
            "round": item.get("round"),
            "candidateHash": item.get("candidateHash"),
            "mechanicalFailureStage": validation.get("mechanicalFailureStage"),
            "deterministicCodes": _error_codes(item.get("deterministicErrors") or []),
            "semanticAuditInvoked": bool(validation.get("semanticAuditInvoked")),
            "semanticCodes": _error_codes(semantic),
            "repairCodes": sorted({
                str(error.get("code")) for error in semantic
                if error.get("code") in REPAIR_CODES
            }),
            "semanticFeedbackSourceRound": validation.get("semanticFeedbackSourceRound"),
            "semanticFeedbackRetained": bool(validation.get("semanticFeedbackRetained")),
            "activeRepairCodes": list(validation.get("activeRepairCodes") or []),
        })
    return result


def _automatic_outcome(
    status: str, triggered: set[str], final_codes: set[str],
) -> str:
    if not triggered:
        return "notApplicable"
    if status == "strictAccepted":
        return "triggeredAndAccepted"
    if triggered.isdisjoint(final_codes):
        return "specificDefectFixedButOtherReject"
    return "triggeredUnresolved"


def build_report() -> dict[str, Any]:
    records = []
    for mode, paths in RUNS.items():
        baseline = _rows(paths["baseline"])
        repair = _rows(paths["repair"])
        for source_id, row in sorted(repair.items()):
            base = baseline.get(source_id, {})
            rounds = _rounds(row)
            triggered = {
                code for item in rounds for code in item["repairCodes"]
            }
            first_trigger = {
                code: next(
                    item["round"] for item in rounds if code in item["repairCodes"]
                )
                for code in sorted(triggered)
            }
            final_validation = row.get("generation_validation") or {}
            final_semantic = final_validation.get("finalSemanticErrors") or []
            final_codes = set(_error_codes(final_semantic))
            records.append({
                "mode": mode,
                "sourceId": source_id,
                "baselineStatus": base.get("status"),
                "baselineDefects": base.get("generation_validation", {}).get(
                    "finalSemanticErrors", []
                ),
                "rounds": rounds,
                "triggeredRepairCodes": sorted(triggered),
                "firstTriggerRound": first_trigger,
                "retentionRounds": [
                    item["round"] for item in rounds
                    if item["semanticFeedbackRetained"]
                ],
                "codeDisappearedAfterTrigger": {
                    code: any(
                        item["round"] > first_trigger[code]
                        and item["semanticAuditInvoked"]
                        and code not in item["activeRepairCodes"]
                        for item in rounds
                    )
                    for code in sorted(triggered)
                },
                "finalStatus": row.get("status"),
                "error": row.get("error"),
                "finalSemanticDefects": final_semantic,
                "automaticOutcome": _automatic_outcome(
                    str(row.get("status") or ""), triggered, final_codes,
                ),
                "manualAssessment": MANUAL_ASSESSMENTS[(mode, source_id)][0],
                "manualNotes": MANUAL_ASSESSMENTS[(mode, source_id)][1],
                "tracePath": row.get("trace_path"),
                "blueprintDir": row.get("blueprint_dir"),
            })
    status_counts = Counter(record["finalStatus"] for record in records)
    outcome_counts = Counter(record["automaticOutcome"] for record in records)
    manual_outcome_counts = Counter(record["manualAssessment"] for record in records)
    trigger_counts = Counter(
        code for record in records for code in record["triggeredRepairCodes"]
    )
    return {
        "schemaVersion": "semantic-repair-v1-report-v1",
        "expectedModeInstances": 23,
        "completedModeInstances": len(records),
        "summary": {
            "statusCounts": dict(sorted(status_counts.items())),
            "triggerCounts": dict(sorted(trigger_counts.items())),
            "automaticOutcomeCounts": dict(sorted(outcome_counts.items())),
            "manualOutcomeCounts": dict(sorted(manual_outcome_counts.items())),
            "semanticAuditErrorCount": sum(
                "semanticAuditError" in str(record.get("error") or "")
                or any(
                    error.get("code") == "semanticAuditError"
                    for round_item in record["rounds"]
                    for error in round_item.get("deterministicErrors", [])
                    if isinstance(error, dict)
                )
                for record in records
            ),
        },
        "manualCodeAssessment": {
            "answerPreassigned": {
                "triggeredModeInstances": 19,
                "targetedDefectRepairedAndAccepted": 9,
                "targetedDefectFixedButOtherReject": 3,
                "targetedDefectUnresolved": 7,
                "notes": (
                    "No source-given constant false trigger was found at mode-instance level. "
                    "cmimc_2025/11 was accepted even though the original preassignment remained."
                ),
            },
            "targetCoverageIncomplete": {
                "triggeredModeInstances": 11,
                "applicableCompletenessOrExtremumCases": [
                    "direct:MATH-500/test/precalculus/768.json",
                    "direct:aime_2025/11",
                    "direct:cmimc_2025/34",
                ],
                "falseTriggerModeInstances": 8,
                "notes": (
                    "The comparator frequently used this code for root grounding, conditionality, "
                    "or a specific target-object mismatch outside the code definition."
                ),
            },
        },
        "records": records,
    }


def markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Semantic Repair V1 targeted rerun",
        "",
        f"Completed: {report['completedModeInstances']}/{report['expectedModeInstances']} mode-instances.",
        "",
        f"Final status counts: `{json.dumps(summary['statusCounts'], sort_keys=True)}`",
        "",
        f"Repair-code trigger counts: `{json.dumps(summary['triggerCounts'], sort_keys=True)}`",
        "",
        f"Automatic outcome counts: `{json.dumps(summary['automaticOutcomeCounts'], sort_keys=True)}`",
        "",
        f"Manual outcome counts: `{json.dumps(summary['manualOutcomeCounts'], sort_keys=True)}`",
        "",
        f"Semantic audit/schema errors: **{summary['semanticAuditErrorCount']}**",
        "",
        "| Mode | Source ID | Baseline | Final | Triggered | First round | Retained rounds | Automatic outcome | Manual assessment |",
        "|---|---|---:|---:|---|---|---|---|---|",
    ]
    for record in report["records"]:
        lines.append(
            "| {mode} | `{sourceId}` | {baselineStatus} | {finalStatus} | {codes} | "
            "{first} | {retained} | {automaticOutcome} | {manualAssessment} |".format(
                **record,
                codes=", ".join(record["triggeredRepairCodes"]) or "-",
                first=json.dumps(record["firstTriggerRound"], sort_keys=True),
                retained=", ".join(map(str, record["retentionRounds"])) or "-",
            )
        )
    lines.extend([
        "",
        "`automaticOutcome` is mechanical. `falseTrigger`, `missedDetection`, and semantic "
        "faithfulness require the manual Blueprint review recorded in the final report.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(
            "/ssd/czx/czx_work/cot_blueprint_refine/semantic_repair_v1_report"
        ),
    )
    args = parser.parse_args()
    report = build_report()
    if report["completedModeInstances"] != report["expectedModeInstances"]:
        raise RuntimeError(
            f"incomplete runs: {report['completedModeInstances']}/"
            f"{report['expectedModeInstances']}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "semantic_repair_v1_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output_dir / "semantic_repair_v1_report.md").write_text(markdown(report))


if __name__ == "__main__":
    main()

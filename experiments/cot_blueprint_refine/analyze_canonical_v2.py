from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


OUTPUT = Path("/ssd/czx/czx_work/cot_blueprint_refine/canonical_v2_report")
EXPERIMENTS = {
    "direct_v2": Path(
        "/ssd/czx/czx_work/cot_blueprint_refine/"
        "qwen3_8b_397b_wrong76_global_defs_direct_canonical_v2_t00"
    ),
    "compact_v2": Path(
        "/ssd/czx/czx_work/cot_blueprint_refine/"
        "qwen3_8b_397b_wrong76_global_defs_compact_separate_canonical_v2_t00"
    ),
}
BASELINES = {
    "direct_v2": Path(
        "/ssd/czx/czx_work/cot_blueprint_refine/"
        "qwen3_8b_397b_wrong76_global_defs_direct_named_t00"
    ),
    "compact_v2": Path(
        "/ssd/czx/czx_work/cot_blueprint_refine/"
        "qwen3_8b_397b_wrong76_global_defs_compact_separate_named_t00_rerun1"
    ),
}
FOCUS_IDS = (
    "MATH-500/test/counting_and_probability/765.json",
    "cmimc_2025/11",
    "MATH-500/test/geometry/434.json",
    "aime_2025/20",
    "MATH-500/test/precalculus/768.json",
    "aime_2025/11",
    "hmmt_feb_2025/16",
    "cmimc_2025/39",
    "MATH-500/test/prealgebra/874.json",
    "cmimc_2025/5",
    "hmmt_feb_2025/19",
)
REPETITION_RE = re.compile(
    r"\b(wait|however|actually|looking closer|potential issue|issue [0-9]+)\b",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def results(root: Path) -> list[dict[str, Any]]:
    return read_jsonl(root / "robustpa/blueprint/results.jsonl")


def comparator_for_round(round_item: dict[str, Any]) -> dict[str, Any] | None:
    audit = (round_item.get("validation") or {}).get("semanticAudit") or {}
    comparator = audit.get("wholeCotComparator")
    return comparator if isinstance(comparator, dict) else None


def trace_llm_requests(root: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    trace_root = root / "robustpa/blueprint/traces"
    for path in trace_root.rglob("*.jsonl"):
        for event in read_jsonl(path):
            if event.get("kind") == "llm_request_end":
                output.append(event)
    return output


def operation_metrics(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        groups[str((event.get("args") or {}).get("operation") or "unknown")].append(event)
    report = {}
    for operation, values in sorted(groups.items()):
        prompt = [int((item.get("args") or {}).get("prompt_tokens") or 0) for item in values]
        completion = [int((item.get("args") or {}).get("completion_tokens") or 0) for item in values]
        total = [int((item.get("args") or {}).get("total_tokens") or 0) for item in values]
        latency = [float(item.get("duration_ms") or 0.0) for item in values]
        finishes = Counter(str((item.get("args") or {}).get("finish_reason") or "") for item in values)
        report[operation] = {
            "requests": len(values),
            "promptTokens": sum(prompt),
            "completionTokens": sum(completion),
            "totalTokens": sum(total),
            "meanPromptTokens": round(mean(prompt), 2),
            "meanCompletionTokens": round(mean(completion), 2),
            "meanLatencyMs": round(mean(latency), 2),
            "finishReasons": dict(finishes),
        }
    return report


def summarize(name: str, root: Path, baseline_root: Path) -> dict[str, Any]:
    rows = results(root)
    baseline_rows = results(baseline_root)
    by_id = {str(row["source_id"]): row for row in rows}
    baseline_by_id = {str(row["source_id"]): row for row in baseline_rows}
    if len(rows) != 76 or len(by_id) != 76:
        raise RuntimeError(f"{name} is not a complete unique Wrong76 run")

    statuses = Counter(str(row.get("status")) for row in rows)
    baseline_statuses = Counter(str(row.get("status")) for row in baseline_rows)
    transitions: dict[str, list[str]] = defaultdict(list)
    final_codes: dict[str, list[str]] = defaultdict(list)
    all_families: dict[str, list[str]] = defaultdict(list)
    comparator_rows: list[dict[str, Any]] = []
    semantic_rounds = 0
    attempts = 0
    for row in rows:
        source_id = str(row["source_id"])
        transitions[
            f"{baseline_by_id[source_id].get('status')} -> {row.get('status')}"
        ].append(source_id)
        for error in (row.get("generation_validation") or {}).get("finalSemanticErrors") or []:
            final_codes[str(error.get("code") or "unknown")].append(source_id)
        for round_item in row.get("generation_history") or []:
            comparator = comparator_for_round(round_item)
            if comparator is None:
                continue
            semantic_rounds += 1
            comparator_rows.append(comparator)
            attempts += len(comparator.get("attempts") or [])
            for issue in comparator.get("issues") or []:
                all_families[str(issue.get("family") or "unknown")].append(source_id)

    runtime = json.loads((root / "robustpa/blueprint/runtime_history.json").read_text())
    reasoning_lengths = [len(str(item.get("reasoning_content") or "")) for item in comparator_rows]
    raw_lengths = [len(str(item.get("raw_content") or "")) for item in comparator_rows]
    repetition_counts = [
        len(REPETITION_RE.findall(str(item.get("reasoning_content") or "")))
        for item in comparator_rows
    ]
    comparator_completion = [int(item.get("completion_tokens") or 0) for item in comparator_rows]

    focus = {}
    for source_id in FOCUS_IDS:
        row = by_id[source_id]
        rounds = []
        for round_item in row.get("generation_history") or []:
            comparator = comparator_for_round(round_item)
            rounds.append({
                "round": round_item.get("round"),
                "candidateHash": round_item.get("candidateHash"),
                "mechanicalFailureStage": (
                    (round_item.get("validation") or {}).get("mechanicalFailureStage")
                ),
                "semanticCodes": [
                    item.get("code") for item in round_item.get("semanticErrors") or []
                ],
                "canonicalIssues": (comparator or {}).get("issues") or [],
                "completionTokens": (comparator or {}).get("completion_tokens"),
                "reasoningChars": len(str((comparator or {}).get("reasoning_content") or "")),
            })
        blueprint_dir = Path(str(row.get("blueprint_dir") or ""))
        artifact = (
            blueprint_dir / "round_00_phase1.lean"
            if row.get("status") == "strictAccepted"
            else blueprint_dir / "phase1_failed_last.lean"
        )
        focus[source_id] = {
            "baselineStatus": baseline_by_id[source_id].get("status"),
            "status": row.get("status"),
            "artifact": str(artifact) if artifact.exists() else "",
            "rounds": rounds,
        }

    return {
        "name": name,
        "root": str(root),
        "baselineRoot": str(baseline_root),
        "rows": len(rows),
        "uniqueIds": len(by_id),
        "statusCounts": dict(statuses),
        "baselineStatusCounts": dict(baseline_statuses),
        "strictAcceptedIds": sorted(
            source_id for source_id, row in by_id.items()
            if row.get("status") == "strictAccepted"
        ),
        "transitions": {key: sorted(value) for key, value in sorted(transitions.items())},
        "finalSemanticCodes": {
            key: {"occurrences": len(value), "ids": sorted(set(value))}
            for key, value in sorted(final_codes.items())
        },
        "allComparatorFamilies": {
            key: {"occurrences": len(value), "uniqueIds": len(set(value)), "ids": sorted(set(value))}
            for key, value in sorted(all_families.items())
        },
        "rounds": {
            "totalGenerationRounds": sum(len(row.get("generation_history") or []) for row in rows),
            "meanGenerationRounds": round(mean(len(row.get("generation_history") or []) for row in rows), 3),
            "semanticRounds": semantic_rounds,
            "semanticAttempts": attempts,
        },
        "wallTimeSeconds": runtime["total_elapsed_s"],
        "llmOperations": operation_metrics(trace_llm_requests(root)),
        "comparatorThinking": {
            "calls": len(comparator_rows),
            "meanReasoningChars": round(mean(reasoning_lengths), 2) if reasoning_lengths else 0,
            "maxReasoningChars": max(reasoning_lengths, default=0),
            "meanRawJsonChars": round(mean(raw_lengths), 2) if raw_lengths else 0,
            "meanCompletionTokens": round(mean(comparator_completion), 2) if comparator_completion else 0,
            "maxCompletionTokens": max(comparator_completion, default=0),
            "meanReconsiderationMarkers": round(mean(repetition_counts), 2) if repetition_counts else 0,
            "maxReconsiderationMarkers": max(repetition_counts, default=0),
        },
        "semanticAuditErrors": [{
            "id": row["source_id"],
            "message": ((row.get("generation_validation") or {}).get("semanticAuditError") or {}).get("message"),
            "requests": (row.get("generation_validation") or {}).get("semanticActualRequestCount"),
            "trace": row.get("trace_path"),
        } for row in rows if row.get("status") == "semanticAuditError"],
        "focus": focus,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = ["# Canonical V2 experiment report", "", "## Summary", ""]
    for name, item in report["experiments"].items():
        status = item["statusCounts"]
        lines.append(
            f"- {name}: strictAccepted={status.get('strictAccepted', 0)}, "
            f"semanticRejected={status.get('semanticRejected', 0)}, "
            f"structuralRejected={status.get('structuralRejected', 0)}, "
            f"semanticAuditError={status.get('semanticAuditError', 0)}, "
            f"wall={item['wallTimeSeconds']:.1f}s."
        )
    lines += ["", "## Status transitions", ""]
    for name, item in report["experiments"].items():
        lines.append(f"### {name}")
        lines.append("")
        for transition, ids in item["transitions"].items():
            lines.append(f"- {transition}: {len(ids)} — {', '.join(ids)}")
        lines.append("")
    lines += ["## Semantic families", ""]
    for name, item in report["experiments"].items():
        lines.append(f"### {name}")
        lines.append("")
        for family, values in item["allComparatorFamilies"].items():
            lines.append(
                f"- {family}: {values['occurrences']} occurrences, "
                f"{values['uniqueIds']} unique IDs."
            )
        lines.append("")
    lines += ["## Runtime and thinking", ""]
    for name, item in report["experiments"].items():
        think = item["comparatorThinking"]
        lines.append(
            f"- {name}: generation rounds={item['rounds']['totalGenerationRounds']}, "
            f"semantic calls={think['calls']}, mean comparator completion="
            f"{think['meanCompletionTokens']}, mean reasoning chars="
            f"{think['meanReasoningChars']}, mean reconsideration markers="
            f"{think['meanReconsiderationMarkers']}."
        )
    lines += ["", "## Focus IDs", ""]
    for source_id in FOCUS_IDS:
        parts = []
        for name, item in report["experiments"].items():
            focus = item["focus"][source_id]
            parts.append(
                f"{name}: {focus['baselineStatus']} -> {focus['status']} "
                f"({focus['artifact']})"
            )
        lines.append(f"- {source_id}: " + "; ".join(parts))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    report = {
        "protocol": "canonical_v2-r4",
        "experiments": {
            name: summarize(name, root, BASELINES[name])
            for name, root in EXPERIMENTS.items()
        },
    }
    direct = set(report["experiments"]["direct_v2"]["strictAcceptedIds"])
    compact = set(report["experiments"]["compact_v2"]["strictAcceptedIds"])
    report["crossMode"] = {
        "acceptedBoth": sorted(direct & compact),
        "directOnly": sorted(direct - compact),
        "compactOnly": sorted(compact - direct),
        "acceptedEither": sorted(direct | compact),
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "canonical_v2_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    (OUTPUT / "canonical_v2_report.md").write_text(markdown(report), encoding="utf-8")
    print(OUTPUT / "canonical_v2_report.json")


if __name__ == "__main__":
    main()

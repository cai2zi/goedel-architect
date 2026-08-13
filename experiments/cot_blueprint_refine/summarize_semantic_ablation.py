from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR.parent))

from cot_blueprint_refine.common import latest_by, read_jsonl, write_json  # noqa: E402


OUTPUT_BASE = Path("/ssd/czx/czx_work/cot_blueprint_refine")
PROFILES = (
    "qwen3_8b_397b_wrong76_subtractive_separate_t06",
    "qwen3_8b_397b_wrong76_subtractive_separate_t00",
    "qwen3_8b_397b_wrong76_subtractive_joint_t06",
    "qwen3_8b_397b_wrong76_subtractive_joint_t00",
    "qwen3_8b_397b_all646_subtractive_separate_t06",
)
PAIRS = (
    (PROFILES[0], PROFILES[1]),
    (PROFILES[0], PROFILES[2]),
    (PROFILES[1], PROFILES[3]),
    (PROFILES[0], PROFILES[4]),
)
ACCEPTED = {"strictAccepted", "acceptedWithWarnings"}


def _results(
    profile: str, output_base: Path = OUTPUT_BASE,
) -> dict[str, dict[str, Any]]:
    path = output_base / profile / "robustpa/blueprint/results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing completed experiment results: {path}")
    return latest_by(read_jsonl(path), "source_id")


def _metrics(profile: str, output_base: Path = OUTPUT_BASE) -> dict[str, Any]:
    path = output_base / profile / "robustpa/blueprint/metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"missing completed experiment metrics: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _root_audit(row: dict[str, Any]) -> dict[str, Any]:
    validation = row.get("generation_validation") or {}
    audit = validation.get("semanticAudit") or {}
    comparator = audit.get("wholeCotComparator") or {}
    return comparator.get("root") or {}


def build_summary(output_base: Path = OUTPUT_BASE) -> dict[str, Any]:
    result_maps = {
        profile: _results(profile, output_base) for profile in PROFILES
    }
    experiments = {}
    for profile in PROFILES:
        rows = result_maps[profile]
        metrics = _metrics(profile, output_base)
        case_1056 = [
            {"sourceId": source_id, "status": row.get("status"),
             "rootAudit": _root_audit(row),
             "manualFalsePositiveReviewRequired": row.get("status") in ACCEPTED}
            for source_id, row in rows.items() if "1056.json" in source_id
        ]
        experiments[profile] = {
            "resultCount": len(rows),
            "statusDistribution": dict(sorted(Counter(
                str(row.get("status") or "unknown") for row in rows.values()
            ).items())),
            "blueprintGeneration": metrics.get("blueprint_generation") or {},
            "case1056": case_1056,
        }

    comparisons = []
    for left, right in PAIRS:
        left_rows = result_maps[left]
        right_rows = result_maps[right]
        shared = sorted(set(left_rows) & set(right_rows))
        transitions = Counter()
        changed = []
        for source_id in shared:
            left_status = str(left_rows[source_id].get("status") or "unknown")
            right_status = str(right_rows[source_id].get("status") or "unknown")
            transitions[f"{left_status}->{right_status}"] += 1
            if left_status != right_status:
                changed.append({
                    "sourceId": source_id, "leftStatus": left_status,
                    "rightStatus": right_status,
                })
        comparisons.append({
            "left": left, "right": right, "sharedCount": len(shared),
            "leftOnlyCount": len(set(left_rows) - set(right_rows)),
            "rightOnlyCount": len(set(right_rows) - set(left_rows)),
            "statusTransitions": dict(sorted(transitions.items())),
            "changedRows": changed,
        })
    return {"experiments": experiments, "comparisons": comparisons}


def main() -> None:
    output = (
        OUTPUT_BASE / "qwen3_8b_397b_semantic_audit_ablation_suite_runtime"
        / "semantic_ablation_summary.json"
    )
    write_json(output, build_summary())
    print(f"[semantic-ablation-summary] {output}", flush=True)


if __name__ == "__main__":
    main()

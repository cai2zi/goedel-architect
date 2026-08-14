from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
GOLD_ROOT = WORKSPACE_ROOT / "czx_work/wrong76_nonthinking_gold"
PIPELINE_RESULTS = (
    WORKSPACE_ROOT
    / "czx_work/cot_blueprint_refine/"
    "qwen3_8b_397b_wrong76_whole_cot_blueprint_generation_thinking_judge/"
    "robustpa/blueprint/results.jsonl"
)
OUTPUT_JSON = GOLD_ROOT / "pipeline_comparison.json"
OUTPUT_MD = HERE / "PIPELINE_COMPARISON.md"


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _group_summary(rows: list[dict[str, Any]], gold: dict[str, dict[str, Any]]) -> dict[str, Any]:
    old_nodes = [int(row.get("total_nodes") or 0) for row in rows]
    old_proved = [int(row.get("proved_node_count") or 0) for row in rows]
    gold_nodes = [len(gold[str(row["source_id"])]["nodes"]) for row in rows]
    gold_target = Counter()
    records_with_disproof = 0
    for row in rows:
        record = gold[str(row["source_id"])]
        gold_target[str(record["labels"][record["target_theorem"]]["label"])] += 1
        records_with_disproof += any(label["label"] == "disproved" for label in record["labels"].values())
    return {
        "records": len(rows),
        "pipeline_total_nodes": sum(old_nodes),
        "pipeline_mean_nodes": sum(old_nodes) / len(rows),
        "pipeline_median_nodes": statistics.median(old_nodes),
        "pipeline_proved_nodes": sum(old_proved),
        "pipeline_aggregate_proved_fraction": (sum(old_proved) / sum(old_nodes)) if sum(old_nodes) else None,
        "pipeline_root_proved": sum(bool(row.get("root_proved")) for row in rows),
        "gold_total_nodes": sum(gold_nodes),
        "gold_mean_nodes": sum(gold_nodes) / len(rows),
        "gold_median_nodes": statistics.median(gold_nodes),
        "gold_records_with_disproof": records_with_disproof,
        "gold_target_labels": dict(sorted(gold_target.items())),
    }


def main() -> None:
    pipeline = [json.loads(line) for line in PIPELINE_RESULTS.read_text(encoding="utf-8").splitlines() if line]
    gold: dict[str, dict[str, Any]] = {}
    for path in GOLD_ROOT.glob("records/*/gold/record.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        gold[str(record["source_id"])] = record
    if len(pipeline) != 76 or len(gold) != 76:
        raise SystemExit(f"expected matched 76/76, got pipeline={len(pipeline)} gold={len(gold)}")
    if {str(row["source_id"]) for row in pipeline} != set(gold):
        raise SystemExit("pipeline/Gold source IDs differ")
    accepted = [
        row for row in pipeline
        if row.get("status") == "strictAccepted" and row.get("semantic_status") == "strictAccepted"
    ]
    rejected = [row for row in pipeline if row not in accepted]
    matched_rows: list[dict[str, Any]] = []
    for row in sorted(pipeline, key=lambda value: str(value["source_id"])):
        record = gold[str(row["source_id"])]
        labels = Counter(label["label"] for label in record["labels"].values())
        matched_rows.append({
            "source_id": row["source_id"],
            "record_id": row["record_id"],
            "pipeline_status": row.get("status"),
            "pipeline_semantic_status": row.get("semantic_status"),
            "pipeline_phase": row.get("phase"),
            "pipeline_nodes": int(row.get("total_nodes") or 0),
            "pipeline_proved_nodes": int(row.get("proved_node_count") or 0),
            "pipeline_root_proved": bool(row.get("root_proved")),
            "gold_nodes": len(record["nodes"]),
            "gold_proved": labels["proved"],
            "gold_disproved": labels["disproved"],
            "gold_blocked": labels["blocked_by_dependency"],
            "gold_target_label": record["labels"][record["target_theorem"]]["label"],
        })
    comparison = {
        "schema_version": "wrong76_gold_pipeline_comparison_v1",
        "comparison_scope": "matched source IDs; existing result is Phase 1 and Gold is fully labeled, so proof rates are not treated as a same-stage model comparison",
        "pipeline_results_path": str(PIPELINE_RESULTS),
        "pipeline_results_sha256": hashlib.sha256(PIPELINE_RESULTS.read_bytes()).hexdigest(),
        "all": _group_summary(pipeline, gold),
        "strict_and_semantic_accepted": _group_summary(accepted, gold),
        "not_strict_and_semantic_accepted": _group_summary(rejected, gold),
        "rows": matched_rows,
    }
    _atomic_text(OUTPUT_JSON, json.dumps(comparison, ensure_ascii=False, indent=2) + "\n")

    a = comparison["strict_and_semantic_accepted"]
    r = comparison["not_strict_and_semantic_accepted"]
    factor = a["pipeline_mean_nodes"] / a["gold_mean_nodes"]
    lines = [
        "# Comparison with the Existing 397B Whole-COT Pipeline",
        "",
        "This is an exact source-ID match against the frozen 76-record Gold set. The existing artifact reports `phase=phase1`; "
        "therefore its proof counters are descriptive provenance, not a same-stage prover accuracy comparison.",
        "",
        "The final Gold revision includes a post-freeze reference-answer audit and is a comparison set, not a blind held-out evaluation set; see `REPORT.md` for provenance.",
        "",
        "## Main observations",
        "",
        f"- Existing Phase 1: **45/76** are both structural `strictAccepted` and semantic `strictAccepted`; **31/76** are rejected before that point.",
        f"- On the accepted 45, the generated Blueprint has **{a['pipeline_total_nodes']} nodes** "
        f"(mean {a['pipeline_mean_nodes']:.2f}, median {a['pipeline_median_nodes']:.0f}); Gold has **{a['gold_total_nodes']} nodes** "
        f"(mean {a['gold_mean_nodes']:.2f}, median {a['gold_median_nodes']:.0f}). The generated graph is **{factor:.2f}x** larger by mean node count.",
        f"- Among those 45 strict+semantic accepted inputs, **{a['gold_records_with_disproof']}/45** contain at least one Lean-verified Gold disproof. "
        f"Gold target labels are `{a['gold_target_labels']}`.",
        f"- The existing artifact has **{a['pipeline_root_proved']}/45** roots proved. It records "
        f"{a['pipeline_proved_nodes']}/{a['pipeline_total_nodes']} individual nodes proved, but this is not comparable to Gold label accuracy because node inventories differ.",
        f"- The 31 rejected inputs still admit compact Gold graphs (mean {r['gold_mean_nodes']:.2f} nodes); "
        f"{r['gold_records_with_disproof']}/31 contain a verified disproof. Thus rejection is often a generation/translation failure, not an absence of a tractable formal decomposition.",
        "",
        "## What the comparison says about the current pipeline",
        "",
        "`strictAccepted` checks that a generated formal artifact is structurally usable and passes the configured semantic audit; it is not a mathematical-correctness label for the COT. "
        "The Gold set demonstrates that local, faithful counterexamples can survive those gates and should be first-class proof targets.",
        "",
        "The largest practical gap is node design. The existing accepted graphs are substantially more fragmented, while Gold isolates one material claim per node and uses short arithmetic, finite enumeration, orientation tests, or explicit witnesses. "
        "This reduces both proof search depth and the amount of context a small prover must carry. Geometry and probability reductions remain explicit `formal_bridge` nodes instead of being hidden in definitions or expanded into monolithic theorems.",
        "",
        "A second gap is polarity selection. A pipeline that always attempts the positive theorem first can spend most of its budget on a false node. Gold exposes a local negative witness as soon as a COT step is false, then marks downstream conclusions `blocked_by_dependency`. "
        "That is the behavior needed for Blueprint proofs to judge the COT rather than merely restate it.",
        "",
        "## Matched inventory",
        "",
        "| source_id | pipeline status | old nodes | old proved | Gold nodes | Gold disproved | Gold target |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in matched_rows:
        status = f"{row['pipeline_status']}/{row['pipeline_semantic_status'] or '-'}"
        lines.append(
            f"| `{row['source_id']}` | `{status}` | {row['pipeline_nodes']} | {row['pipeline_proved_nodes']} | "
            f"{row['gold_nodes']} | {row['gold_disproved']} | `{row['gold_target_label']}` |"
        )
    lines.append("")
    _atomic_text(OUTPUT_MD, "\n".join(lines))
    print(json.dumps({
        "accepted": len(accepted),
        "rejected": len(rejected),
        "accepted_pipeline_mean_nodes": a["pipeline_mean_nodes"],
        "accepted_gold_mean_nodes": a["gold_mean_nodes"],
        "accepted_gold_records_with_disproof": a["gold_records_with_disproof"],
        "accepted_gold_target_labels": a["gold_target_labels"],
        "json": str(OUTPUT_JSON),
        "report": str(OUTPUT_MD),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
OUTPUT_ROOT = WORKSPACE_ROOT / "czx_work/wrong76_nonthinking_gold"
RECORDS_ROOT = OUTPUT_ROOT / "records"
BLIND_INPUTS = OUTPUT_ROOT / "blind_inputs.jsonl"
SUMMARY_PATH = OUTPUT_ROOT / "final_summary.json"
MANIFEST_PATH = OUTPUT_ROOT / "freeze_manifest.json"
REPORT_PATH = HERE / "REPORT.md"


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    blind_rows = [json.loads(line) for line in BLIND_INPUTS.read_text(encoding="utf-8").splitlines() if line]
    expected = {str(row["source_id"]): row for row in blind_rows}
    paths = sorted(RECORDS_ROOT.glob("*/gold/record.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    actual = {str(record["source_id"]): record for record in records}
    if len(blind_rows) != 76 or len(expected) != 76:
        raise SystemExit(f"expected 76 distinct blind inputs, got rows={len(blind_rows)} distinct={len(expected)}")
    if len(records) != 76 or set(actual) != set(expected):
        raise SystemExit(
            f"record coverage mismatch: records={len(records)} missing={sorted(set(expected)-set(actual))} "
            f"extra={sorted(set(actual)-set(expected))}"
        )

    label_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    validation_counts: Counter[str] = Counter()
    records_with_disproof = 0
    total_steps = 0
    rows: list[dict[str, Any]] = []
    manifest_records: list[dict[str, Any]] = []
    failures: list[str] = []

    for record_path, record in zip(paths, records, strict=True):
        source_id = str(record["source_id"])
        if record["record_id"] != record_path.parent.parent.name:
            failures.append(f"record/path mismatch: {source_id}")
        if record["source"]["nonthinking_cot_sha256"] != expected[source_id]["nonthinking_cot_sha256"]:
            failures.append(f"blind COT hash mismatch: {source_id}")
        completion = record.get("completion") or {}
        validation = record.get("deterministic_validation") or {}
        status = str(completion.get("status"))
        validation_counts[status] += 1
        if status != "gold_complete":
            failures.append(f"not gold_complete: {source_id}:{status}")
        if validation.get("passed") is not True:
            failures.append(f"deterministic validation failed: {source_id}")
        if completion.get("all_proof_artifacts_verified") is not True:
            failures.append(f"proof replay failed: {source_id}")
        if record.get("gold_fidelity_review", {}).get("passed") is not True:
            failures.append(f"fidelity review failed: {source_id}")

        labels = record["labels"]
        local = Counter(str(label["label"]) for label in labels.values())
        label_counts.update(local)
        if local["disproved"]:
            records_with_disproof += 1
        for name, metadata in record["nodes"].items():
            role_counts[str(metadata["node_role"])] += 1
            kind_counts[str(metadata["kind"])] += 1
            if name not in labels:
                failures.append(f"missing label: {source_id}:{name}")
        target_label = str(labels[record["target_theorem"]]["label"])
        target_counts[target_label] += 1
        total_steps += len(record["steps"])
        rows.append({
            "source_id": source_id,
            "record_id": record["record_id"],
            "steps": len(record["steps"]),
            "nodes": len(record["nodes"]),
            "definitions": local["definition_valid"],
            "proved": local["proved"],
            "disproved": local["disproved"],
            "blocked": local["blocked_by_dependency"],
            "target_label": target_label,
            "record_path": str(record_path),
        })

        gold_dir = record_path.parent
        artifacts = sorted(
            path for path in gold_dir.rglob("*")
            if path.is_file() and path.name not in {"freeze_manifest.json"}
        )
        manifest_records.append({
            "source_id": source_id,
            "record_id": record["record_id"],
            "artifacts": [
                {
                    "path": str(path.relative_to(OUTPUT_ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha_file(path),
                }
                for path in artifacts
            ],
        })

    if failures:
        raise SystemExit("freeze refused:\n" + "\n".join(failures))

    summary = {
        "schema_version": "wrong76_nonthinking_gold_summary_v1",
        "record_count": len(records),
        "distinct_source_count": len(actual),
        "blind_input_sha256": _sha_file(BLIND_INPUTS),
        "total_cot_steps": total_steps,
        "total_active_nodes": sum(label_counts.values()),
        "labels": dict(sorted(label_counts.items())),
        "node_roles": dict(sorted(role_counts.items())),
        "node_kinds": dict(sorted(kind_counts.items())),
        "target_labels": dict(sorted(target_counts.items())),
        "records_with_at_least_one_disproof": records_with_disproof,
        "record_completion": dict(sorted(validation_counts.items())),
        "all_76_covered": len(records) == len(actual) == 76,
        "all_gold_complete": validation_counts == Counter({"gold_complete": 76}),
        "all_proof_artifacts_verified": True,
        "rows": rows,
    }
    frozen_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": "wrong76_nonthinking_gold_freeze_v1",
        "frozen_at_utc": frozen_at,
        "authoring_provenance": {
            "initial_blind_build_included": ["problem", "nonthinking_cot_after_last_think_close", "claimed_answer"],
            "initial_blind_build_excluded": [
                "dataset_gold_answer", "is_correct", "hidden_thinking", "previous_blueprint_labels", "model_judge_results"
            ],
            "postfreeze_audit": (
                "After the first 76/76 blind freeze, extracted reference answers were inspected for the 16 records whose "
                "Gold target was still proved. The final revision is therefore not a strictly blind evaluation artifact. "
                "Corrections were implemented from explicit problem/COT contradictions and local witnesses; unreliable "
                "reference extractions were not forced onto the Gold labels."
            ),
        },
        "blind_input_path": str(BLIND_INPUTS),
        "blind_input_sha256": summary["blind_input_sha256"],
        "record_count": 76,
        "records": manifest_records,
    }
    _atomic_json(SUMMARY_PATH, summary)
    _atomic_json(MANIFEST_PATH, manifest)

    lines = [
        "# Wrong76 Non-Thinking Gold: Final Build Report",
        "",
        f"Frozen at `{frozen_at}` after a complete 76/76 per-record Lean replay.",
        "",
        "## Aggregate result",
        "",
        f"- Records: **{len(records)}/76 gold_complete**",
        f"- Material COT steps: **{total_steps}**",
        f"- Active nodes: **{sum(label_counts.values())}**",
        f"- Labels: **{label_counts['definition_valid']} definition_valid**, **{label_counts['proved']} proved**, "
        f"**{label_counts['disproved']} disproved**, **{label_counts['blocked_by_dependency']} blocked_by_dependency**",
        f"- Target labels: **{target_counts['proved']} proved**, **{target_counts['disproved']} disproved**, "
        f"**{target_counts['blocked_by_dependency']} blocked_by_dependency**",
        f"- Records containing a Lean-verified disproof: **{records_with_disproof}/76**",
        "- Mechanical checks: parse, whole-file Lean, canonical rebuild and Lean, Phase-2 contract, "
        "Phase-2 standalone, definition bundle, metadata/source-span contract, and independent proof/disproof replay.",
        "",
        "## Interpretation boundary",
        "",
        "A `proved` node means the closed Lean node represented in this Gold graph has a replayable proof. "
        "A `disproved` node has a replayable proof of the exact closed-theorem negation. "
        "A `blocked_by_dependency` target preserves the COT conclusion but does not award it a proof after an upstream claim is refuted. "
        "The deterministic suite is intentionally mechanical and does not claim to mechanize natural-language semantic equivalence.",
        "",
        "The Gold graphs favor short arithmetic, finite witnesses, and local counterexamples. Geometry/probability reductions that are "
        "mathematically meaningful but expensive in Lean are isolated as explicit `formal_bridge` nodes and documented in each record. "
        "This is the intended comparison point for diagnosing whether the generation pipeline loses semantics or creates unnecessarily hard nodes.",
        "",
        "## Authoring provenance",
        "",
        "The initial 76-record build and first complete freeze used only the problem, the final non-thinking COT, and its claimed answer. "
        "A post-freeze audit then inspected extracted reference answers for the 16 records whose target was still marked `proved`; this exposed several cases where the first Gold graph had verified only terminal arithmetic. "
        "Those cases were repaired with explicit problem/COT contradictions and local witnesses. Consequently, the final revision is a Gold comparison set, not a strictly blind evaluation set. "
        "Reference extraction was used as an audit signal rather than accepted as truth: four targets remain `proved` because their written COT is verifiable under the agreed semantic source despite external answer/extraction disagreement.",
        "",
        "## Per-record inventory",
        "",
        "| source_id | steps | nodes | def | proved | disproved | blocked | target |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['source_id']}` | {row['steps']} | {row['nodes']} | {row['definitions']} | "
            f"{row['proved']} | {row['disproved']} | {row['blocked']} | `{row['target_label']}` |"
        )
    lines.extend([
        "",
        "## Reproduction handles",
        "",
        f"- Machine-readable summary: `{SUMMARY_PATH}`",
        f"- Frozen artifact hashes: `{MANIFEST_PATH}`",
        f"- Records and Lean files: `{RECORDS_ROOT}`",
        "",
    ])
    _atomic_text(REPORT_PATH, "\n".join(lines))
    print(json.dumps({
        "records": len(records),
        "gold_complete": validation_counts["gold_complete"],
        "active_nodes": sum(label_counts.values()),
        "labels": dict(label_counts),
        "target_labels": dict(target_counts),
        "summary": str(SUMMARY_PATH),
        "manifest": str(MANIFEST_PATH),
        "report": str(REPORT_PATH),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

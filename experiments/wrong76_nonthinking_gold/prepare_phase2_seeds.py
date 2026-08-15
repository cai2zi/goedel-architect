from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from blueprint import _parse_blueprint  # noqa: E402
from checkpoint import CheckpointState  # noqa: E402
from orchestrator import active_node_names  # noqa: E402
from robustpa_refine.io_utils import safe_stem  # noqa: E402


SCHEMA_VERSION = "wrong76_gold_phase2_seed_v2"
DATASET_SUBSET = "qwen3_8b_math_verify"
DEFAULT_GOLD_ROOT = WORKSPACE_ROOT / "czx_work/wrong76_nonthinking_gold"
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "czx_work/wrong76_nonthinking_gold_phase2_seed"
EXPECTED_RECORDS = 76
EXPECTED_NODES = 381
FORBIDDEN_SEED_TEXT = (
    '"labels"',
    '"proof_body"',
    '"negated_declaration"',
    "/gold/lean/",
    ".positive.lean",
    ".negative.lean",
)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _safe_filename(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_]", "_", value).strip("_")
    return result or "unknown"


def _informal_statement(row: dict[str, Any]) -> str:
    return (
        "Original problem:\n"
        f"{str(row['problem']).strip()}\n\n"
        "Claimed final answer from the original response:\n"
        f"\\boxed{{{row['claimed_answer']}}}\n\n"
        "Formalize and verify the claim that this answer solves the original problem."
    )


def _runtime_identity(row: dict[str, Any]) -> tuple[str, str, str]:
    source_id = str(row["source_id"])
    split = str(row["split"])
    record_id = safe_stem(source_id, prefix="robustpa_")
    theorem_name = safe_stem(
        f"robustpa_{safe_stem(DATASET_SUBSET)}_{safe_stem(split)}_{record_id}"
    )
    unique_id = f"{DATASET_SUBSET}__{split}__{record_id}"
    return record_id, theorem_name, unique_id


def _gold_blueprint_path(gold_root: Path, row: dict[str, Any]) -> Path:
    return gold_root / "records" / str(row["record_id"]) / "gold" / "blueprint.lean"


def _manifest_artifact_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for record in manifest.get("records", []):
        source_id = str(record.get("source_id") or "")
        if not source_id or source_id in index:
            raise RuntimeError(f"invalid duplicate freeze-manifest source_id: {source_id!r}")
        index[source_id] = record
    return index


def _verify_gold_freeze(gold_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    blind_path = gold_root / "blind_inputs.jsonl"
    manifest_path = gold_root / "freeze_manifest.json"
    summary_path = gold_root / "final_summary.json"
    for required in (blind_path, manifest_path, summary_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    rows = _read_jsonl(blind_path)
    manifest = _read_json(manifest_path)
    summary = _read_json(summary_path)
    source_ids = [str(row.get("source_id") or "") for row in rows]
    if len(rows) != EXPECTED_RECORDS or len(set(source_ids)) != EXPECTED_RECORDS:
        raise RuntimeError(
            f"expected {EXPECTED_RECORDS} distinct blind inputs, got rows={len(rows)} "
            f"distinct={len(set(source_ids))}"
        )
    if manifest.get("record_count") != EXPECTED_RECORDS:
        raise RuntimeError(f"freeze manifest record_count={manifest.get('record_count')}")
    if manifest.get("blind_input_sha256") != _sha_file(blind_path):
        raise RuntimeError("blind_inputs.jsonl does not match freeze manifest")
    if summary.get("total_active_nodes") != EXPECTED_NODES:
        raise RuntimeError(f"Gold summary node count={summary.get('total_active_nodes')}")
    if summary.get("all_76_covered") is not True or summary.get("all_gold_complete") is not True:
        raise RuntimeError("Gold summary is not complete")

    manifest_index = _manifest_artifact_index(manifest)
    if set(manifest_index) != set(source_ids):
        raise RuntimeError("freeze manifest IDs do not equal blind input IDs")
    for source_id, record in manifest_index.items():
        for artifact in record.get("artifacts", []):
            path = gold_root / str(artifact["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != int(artifact["bytes"]):
                raise RuntimeError(f"frozen artifact size drift: {path}")
            if _sha_file(path) != str(artifact["sha256"]):
                raise RuntimeError(f"frozen artifact hash drift: {path}")
    return rows, manifest


def _extract_gold_target(lean_code: str) -> str:
    names = re.findall(r"(?m)^\s*theorem\s+([A-Za-z_][A-Za-z0-9_']*)\b", lean_code)
    if not names:
        raise RuntimeError("Gold Blueprint has no theorem declaration")
    return names[-1]


def _rename_root(lean_code: str, gold_target: str, runtime_target: str) -> str:
    pattern = re.compile(
        rf"(?m)^(\s*theorem\s+){re.escape(gold_target)}\b"
    )
    renamed, replacements = pattern.subn(
        lambda match: f"{match.group(1)}{runtime_target}", lean_code, count=1
    )
    if replacements != 1:
        raise RuntimeError(
            f"expected one root declaration for {gold_target}, got {replacements}"
        )
    return renamed


def _validate_blueprint_adaptation(
    original_code: str,
    runtime_code: str,
    gold_target: str,
    runtime_target: str,
) -> tuple[Any, int]:
    original = _parse_blueprint(original_code, gold_target)
    runtime = _parse_blueprint(runtime_code, runtime_target)
    if not original.nodes or original.nodes[-1].name != gold_target:
        raise RuntimeError(f"Gold target is not the final Blueprint node: {gold_target}")
    if not runtime.nodes or runtime.nodes[-1].name != runtime_target:
        raise RuntimeError(f"runtime target is not the final Blueprint node: {runtime_target}")
    if len(original.nodes) != len(runtime.nodes):
        raise RuntimeError("root adaptation changed the node count")
    for old, new in zip(original.nodes, runtime.nodes, strict=True):
        expected_name = runtime_target if old.name == gold_target else old.name
        if new.name != expected_name:
            raise RuntimeError(f"node name drift: {old.name} -> {new.name}")
        if old.kind != new.kind or old.dependencies != new.dependencies:
            raise RuntimeError(f"node contract drift during root rename: {old.name}")
        if old.statement != new.statement or old.proof_sketch != new.proof_sketch:
            raise RuntimeError(f"node metadata drift during root rename: {old.name}")
    active = active_node_names(runtime)
    if active != {node.name for node in runtime.nodes}:
        missing = sorted({node.name for node in runtime.nodes} - active)
        raise RuntimeError(f"runtime Blueprint contains nodes outside root closure: {missing}")
    return runtime, len(runtime.nodes)


def _generated_files(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": _sha_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "seed_manifest.json"
    ]


def _validate_existing(output_root: Path, gold_manifest_sha256: str) -> dict[str, Any]:
    seed_manifest_path = output_root / "seed_manifest.json"
    if not seed_manifest_path.is_file():
        raise RuntimeError(
            f"existing seed root has no seed_manifest.json and will not be overwritten: {output_root}"
        )
    seed_manifest = _read_json(seed_manifest_path)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "source_freeze_manifest_sha256": gold_manifest_sha256,
        "record_count": EXPECTED_RECORDS,
        "node_count": EXPECTED_NODES,
    }
    for key, value in expected.items():
        if seed_manifest.get(key) != value:
            raise RuntimeError(
                f"existing seed manifest drift for {key}: "
                f"expected={value!r} actual={seed_manifest.get(key)!r}"
            )
    for artifact in seed_manifest.get("generated_artifacts", []):
        path = output_root / str(artifact["path"])
        if not path.is_file() or path.stat().st_size != int(artifact["bytes"]):
            raise RuntimeError(f"generated seed artifact missing/size drift: {path}")
        if _sha_file(path) != str(artifact["sha256"]):
            raise RuntimeError(f"generated seed artifact hash drift: {path}")
    _verify_no_label_leakage(output_root)
    return seed_manifest


def _verify_no_label_leakage(output_root: Path) -> None:
    inspected = [
        path
        for path in output_root.rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl", ".lean"}
    ]
    for path in inspected:
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_SEED_TEXT:
            if forbidden in text:
                raise RuntimeError(f"Gold label/proof leakage `{forbidden}` in {path}")


def _build_seed_tree(
    build_root: Path,
    output_root: Path,
    gold_root: Path,
    rows: list[dict[str, Any]],
    freeze_manifest_sha256: str,
) -> dict[str, Any]:
    prepared_root = build_root / "prepared"
    data_root = prepared_root / "data" / DATASET_SUBSET
    seed_root = build_root / "robustpa" / "blueprint"
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_results: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    total_nodes = 0

    ordered_rows = sorted(
        rows,
        key=lambda row: (
            str(row["split"]), int(row.get("source_row_index", -1)), str(row["source_id"])
        ),
    )
    for row in ordered_rows:
        record_id, runtime_target, unique_id = _runtime_identity(row)
        if record_id != str(row["record_id"]):
            raise RuntimeError(
                f"record ID drift for {row['source_id']}: {record_id} != {row['record_id']}"
            )
        original_path = _gold_blueprint_path(gold_root, row)
        original_code = original_path.read_text(encoding="utf-8")
        gold_target = _extract_gold_target(original_code)
        runtime_code = _rename_root(original_code, gold_target, runtime_target)
        blueprint, node_count = _validate_blueprint_adaptation(
            original_code, runtime_code, gold_target, runtime_target
        )
        total_nodes += node_count

        informal_statement = _informal_statement(row)
        informal_proof = str(row["nonthinking_cot"])
        claimed_answer = str(row["claimed_answer"])
        split = str(row["split"])
        row_index = int(row.get("source_row_index", -1))
        grouped[split].append({
            "name": str(row["source_id"]),
            "source": split,
            "row_index": row_index,
            "problem": str(row["problem"]),
            "claimed_answer": claimed_answer,
            "post_think_cot": informal_proof,
            "informal_statement": informal_statement,
            "informal_proof": informal_proof,
        })

        checkpoint = CheckpointState(
            informal_statement=informal_statement,
            informal_proof=informal_proof,
            claimed_answer=claimed_answer,
            model="wrong76-nonthinking-gold",
            semantic_fidelity_enabled=True,
            semantic_static_gate=False,
            semantic_minimal_ir=False,
            semantic_freeze_refinement=False,
            semantic_status="strictAccepted",
        )
        checkpoint.set_blueprint(blueprint)
        relative = Path(DATASET_SUBSET) / split / f"{record_id}.json"
        build_checkpoint_path = seed_root / "phase1_seeds" / "checkpoints" / relative
        final_checkpoint_path = (
            output_root / "robustpa" / "blueprint" / "phase1_seeds" / "checkpoints" / relative
        )
        checkpoint.save(build_checkpoint_path)

        source_results.append({
            "id": unique_id,
            "record_id": record_id,
            "source_id": str(row["source_id"]),
            "subset": DATASET_SUBSET,
            "split": split,
            "row_index": row_index,
            "theorem_name": runtime_target,
            "status": "strictAccepted",
            "semantic_status": "strictAccepted",
            "phase": "phase1",
            "success": True,
            "root_proved": False,
            "checkpoint_path": str(final_checkpoint_path),
        })
        mappings.append({
            "source_id": str(row["source_id"]),
            "record_id": record_id,
            "subset": DATASET_SUBSET,
            "split": split,
            "row_index": row_index,
            "gold_target": gold_target,
            "runtime_target": runtime_target,
            "node_count": node_count,
            "gold_blueprint_path": str(original_path),
            "gold_blueprint_sha256": _sha_file(original_path),
            "runtime_blueprint_sha256": _sha_bytes(runtime_code.encode("utf-8")),
            "checkpoint_path": str(final_checkpoint_path),
        })

    if total_nodes != EXPECTED_NODES:
        raise RuntimeError(f"expected {EXPECTED_NODES} nodes, adapted {total_nodes}")
    for split, split_rows in sorted(grouped.items()):
        path = data_root / f"{_safe_filename(split)}-00000-of-00001.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(split_rows), path, compression="zstd")

    generation_rows = [
        row for split in sorted(grouped) for row in grouped[split]
    ]
    _write_jsonl(prepared_root / "generation_inputs.jsonl", generation_rows)
    _write_jsonl(seed_root / "results.jsonl", source_results)
    _write_json(build_root / "target_mapping.json", {
        "schema_version": SCHEMA_VERSION,
        "records": mappings,
    })
    _verify_no_label_leakage(build_root)

    seed_manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_gold_root": str(gold_root),
        "source_freeze_manifest_sha256": freeze_manifest_sha256,
        "source_blind_inputs_sha256": _sha_file(gold_root / "blind_inputs.jsonl"),
        "record_count": len(mappings),
        "node_count": total_nodes,
        "dataset_subset": DATASET_SUBSET,
        "label_oracle_exposed": False,
        "root_rename_only": True,
        "generated_artifacts": _generated_files(build_root),
    }
    _write_json(build_root / "seed_manifest.json", seed_manifest)
    return seed_manifest


def prepare(gold_root: Path, output_root: Path) -> dict[str, Any]:
    rows, _freeze_manifest = _verify_gold_freeze(gold_root)
    freeze_manifest_sha256 = _sha_file(gold_root / "freeze_manifest.json")
    if output_root.exists():
        manifest = _validate_existing(output_root, freeze_manifest_sha256)
        print(
            f"[gold-seed] reused records={manifest['record_count']} "
            f"nodes={manifest['node_count']} root={output_root}",
            flush=True,
        )
        return manifest

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output_root.name}.tmp_", dir=output_root.parent
    ))
    try:
        manifest = _build_seed_tree(
            temporary, output_root, gold_root, rows, freeze_manifest_sha256
        )
        os.replace(temporary, output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        f"[gold-seed] built records={manifest['record_count']} "
        f"nodes={manifest['node_count']} root={output_root}",
        flush=True,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build immutable Phase 1-compatible seeds from Wrong76 Gold Blueprints."
    )
    parser.add_argument("--gold-root", type=Path, default=DEFAULT_GOLD_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare(args.gold_root.expanduser().resolve(), args.output_root.expanduser().resolve())


if __name__ == "__main__":
    main()

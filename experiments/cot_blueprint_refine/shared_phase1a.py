#!/usr/bin/env python3
"""Validate and publish the immutable shared Phase-1A seed set."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def _latest(path: Path, key: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row[key])] = row
    return rows


def _atomic_write_json(path: Path, payload: object) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def validate(
    root: Path,
    expected: int,
    *,
    drop_source_ids: set[str] | None = None,
) -> Path:
    dropped_ids = set(drop_source_ids or ())
    result_root = root / "robustpa" / "blueprint"
    rows = _latest(result_root / "results.jsonl", "source_id")
    prepared = _latest(root / "prepared" / "generation_inputs.jsonl", "name")
    if len(prepared) != expected:
        raise RuntimeError(
            f"shared prepared set is incomplete: expected={expected} actual={len(prepared)}"
        )
    if len(rows) != expected:
        raise RuntimeError(f"shared Phase 1A is incomplete: expected={expected} actual={len(rows)}")
    if set(rows) != set(prepared):
        raise RuntimeError("shared Phase 1A source IDs do not match the prepared set")
    unknown_drops = dropped_ids - set(rows)
    if unknown_drops:
        raise RuntimeError(f"drop source IDs are not present: {sorted(unknown_drops)}")
    records = []
    dropped = []
    for source_id, row in sorted(rows.items()):
        if source_id in dropped_ids:
            dropped.append({
                "source_id": source_id,
                "status": row.get("status"),
                "error": row.get("error"),
                "failure_stage": row.get("failed_blueprint_failure_stage"),
                "diagnostics_path": row.get("failed_blueprint_diagnostics_path"),
            })
            continue
        if row.get("status") != "phase1aReady":
            raise RuntimeError(f"{source_id}: status={row.get('status')} is not phase1aReady")
        path = (result_root / "blueprints" / str(row["subset"]) / str(row["split"])
                / str(row["record_id"]) / "phase1a_canonical.lean")
        if not path.is_file():
            raise RuntimeError(f"{source_id}: missing {path}")
        lean_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        cot = str(row.get("cot_manifest_json") or "")
        records.append({
            "source_id": source_id,
            "theorem_name": row.get("theorem_name"),
            "claimed_answer": row.get("claimed_answer"),
            "cot_manifest_hash": hashlib.sha256(cot.encode()).hexdigest(),
            "lean_hash": lean_hash,
            "seed_path": str(path),
        })
    payload = {
        "schema_version": 2,
        "ready": True,
        "totalCount": len(rows),
        "readyCount": len(records),
        "droppedCount": len(dropped),
        "records": records,
        "dropped": dropped,
    }
    target = result_root / "PHASE1A_READY.json"
    _atomic_write_json(target, payload)
    include_target = result_root / "phase1a_ready_ids.json"
    _atomic_write_json(
        include_target,
        {"include_ids": [item["source_id"] for item in records]},
    )
    print(
        f"[shared-phase1a] prepared={len(prepared)} ready={len(records)} "
        f"dropped={len(dropped)} armInput={len(records)}",
        flush=True,
    )
    if dropped:
        print(
            "[shared-phase1a-dropped] "
            + ", ".join(item["source_id"] for item in dropped),
            flush=True,
        )
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--drop-source-id", action="append", default=[])
    args = parser.parse_args()
    print(validate(
        args.root,
        args.expected,
        drop_source_ids=set(args.drop_source_id),
    ))


if __name__ == "__main__":
    main()

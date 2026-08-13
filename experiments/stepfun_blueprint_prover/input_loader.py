from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from blueprint import Blueprint, phase2_contract_errors
from checkpoint import CheckpointState
from orchestrator import active_node_names


@dataclass(frozen=True)
class AcceptedBlueprint:
    record_id: str
    source_id: str
    subset: str
    split: str
    checkpoint_path: Path
    checkpoint_sha256: str
    state: CheckpointState
    blueprint: Blueprint
    source_result: dict[str, Any]

    @property
    def key(self) -> str:
        return f"{self.subset}/{self.split}/{self.record_id}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_accepted_blueprints(
    source_root: Path,
    *,
    limit: int | None = None,
    include_ids: set[str] | None = None,
) -> list[AcceptedBlueprint]:
    results_path = source_root / "results.jsonl"
    if not results_path.is_file():
        raise FileNotFoundError(f"source results not found: {results_path}")
    selected: list[AcceptedBlueprint] = []
    seen: set[str] = set()
    with results_path.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            row = json.loads(raw)
            if row.get("status") != "strictAccepted" or row.get("semantic_status") != "strictAccepted":
                continue
            record_id = str(row["record_id"])
            source_id = str(row.get("source_id") or record_id)
            if include_ids and record_id not in include_ids and source_id not in include_ids:
                continue
            if record_id in seen:
                raise ValueError(f"duplicate accepted record_id: {record_id}")
            checkpoint_path = Path(str(row["checkpoint_path"])).resolve()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"accepted checkpoint missing: {checkpoint_path}")
            state = CheckpointState.load(checkpoint_path)
            blueprint = state.get_blueprint()
            if blueprint is None:
                raise ValueError(f"accepted checkpoint has no Blueprint: {checkpoint_path}")
            if state.semantic_status != "strictAccepted" or state.status.value != "running":
                raise ValueError(
                    f"accepted checkpoint is not a pristine strict seed: {checkpoint_path} "
                    f"semantic={state.semantic_status} status={state.status.value}"
                )
            errors = phase2_contract_errors(blueprint)
            if errors:
                raise ValueError(f"Phase-2 contract rejected {record_id}: {errors}")
            active_node_names(blueprint)
            selected.append(AcceptedBlueprint(
                record_id=record_id,
                source_id=source_id,
                subset=str(row.get("subset") or "unknown"),
                split=str(row.get("split") or "unknown"),
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=_sha256(checkpoint_path),
                state=state,
                blueprint=blueprint,
                source_result=row,
            ))
            seen.add(record_id)
            if limit is not None and len(selected) >= limit:
                break
    return selected

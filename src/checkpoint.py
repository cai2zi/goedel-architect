"""Checkpoint schema for the Kimina-only RobustPA pipeline."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

from blueprint import Blueprint, _parse_blueprint
from prover import ProofSignal, ProverResult


class RunStatus(str, Enum):
    RUNNING = "running"
    SOLVED = "solved"
    EXHAUSTED = "exhausted"
    ERROR = "error"


@dataclass
class CheckpointState:
    informal_statement: str
    model: str
    status: RunStatus = RunStatus.RUNNING
    iteration: int = 0
    blueprint_lean_file: str = ""
    blueprint_target: str = ""
    blueprint_phase2_header: str = ""
    proved_cache: dict[str, str] = field(default_factory=dict)
    proof_cache_keys: dict[str, str] = field(default_factory=dict)
    node_results: dict[str, dict] = field(default_factory=dict)
    refinement_history: list[str] = field(default_factory=list)
    final_lean_file: str = ""
    final_lean_errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.status = RunStatus(self.status)

    @property
    def root_proved(self) -> bool:
        return self.status == RunStatus.SOLVED

    def set_blueprint(self, blueprint: Blueprint) -> None:
        self.blueprint_lean_file = blueprint.lean_file
        self.blueprint_target = blueprint.target_theorem
        self.blueprint_phase2_header = blueprint.phase2_header

    def get_blueprint(self) -> Blueprint | None:
        if not self.blueprint_lean_file:
            return None
        blueprint = _parse_blueprint(self.blueprint_lean_file, self.blueprint_target)
        if self.blueprint_phase2_header:
            blueprint.phase2_header = self.blueprint_phase2_header
        return blueprint

    def set_node_results(self, node_results: dict) -> None:
        self.node_results = {
            name: {
                "signal": node_result.result.signal.value,
                "proof_body": node_result.result.proof_body,
                "lean_errors": list(node_result.result.lean_errors),
            }
            for name, node_result in node_results.items()
        }

    def get_prover_results(self) -> dict[str, ProverResult]:
        return {
            name: ProverResult(
                ProofSignal(data["signal"]),
                data.get("proof_body", ""),
                list(data.get("lean_errors", [])),
            )
            for name, data in self.node_results.items()
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            dir=path.parent, prefix=".tmp_ckpt_",
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(asdict(self), handle, ensure_ascii=False, indent=2)
            os.replace(temporary_path, path)
        except Exception:
            Path(temporary_path).unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: Path) -> CheckpointState:
        with path.open("r", encoding="utf-8") as handle:
            return cls(**json.load(handle))

    @classmethod
    def load_or_none(cls, path: Path | None) -> CheckpointState | None:
        if path is None or not path.exists():
            return None
        return cls.load(path)


def path_for_theorem(checkpoint_dir: Path, theorem_name: str) -> Path:
    safe_name = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in theorem_name
    )
    return checkpoint_dir / f"{safe_name}.json"

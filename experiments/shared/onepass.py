from __future__ import annotations

import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from blueprint import Blueprint
from checkpoint import CheckpointState
from lean_compiler import AbstractLeanCompiler, LeanCompiler
from mathlib_retrieval import MathlibRetrieval
from pipeline import run_phase1, run_phase2
from tracer import JsonlTracer, TraceEvent


def _blueprint_score(blueprint: Blueprint | None, orch_result: Any | None) -> dict[str, Any]:
    if blueprint is None:
        return {
            "root_theorem": "",
            "root_proved": False,
            "total_nodes": 0,
            "proved_node_count": 0,
            "proved_ratio": 0.0,
            "failed_nodes": [],
            "proved_nodes": [],
            "all_proved": False,
        }

    total_nodes = len(blueprint.nodes)
    proved = set(getattr(orch_result, "proved", set()) or set())
    failed = set(getattr(orch_result, "failed", set()) or set())
    root_theorem = blueprint.target_theorem
    proved_node_count = len(proved)
    proved_ratio = round(proved_node_count / total_nodes, 4) if total_nodes else 0.0
    return {
        "root_theorem": root_theorem,
        "root_proved": root_theorem in proved,
        "total_nodes": total_nodes,
        "proved_node_count": proved_node_count,
        "proved_ratio": proved_ratio,
        "failed_nodes": sorted(failed),
        "proved_nodes": sorted(proved),
        "all_proved": total_nodes > 0 and proved_node_count == total_nodes,
    }


def _checkpoint_score(checkpoint_path: Path) -> dict[str, Any] | None:
    if not checkpoint_path.exists():
        return None
    state = CheckpointState.load(checkpoint_path)
    blueprint = state.get_blueprint()
    if blueprint is None:
        return None
    node_names = set(blueprint.nodes_by_name())
    proved = set(state.proved_cache) & node_names
    failed = node_names - proved

    class Result:
        pass

    result = Result()
    result.proved = proved
    result.failed = failed
    return _blueprint_score(blueprint, result)


def _load_resumable_blueprint(
    checkpoint_path: Path,
    theorem_stmt: str,
) -> tuple[CheckpointState | None, Blueprint | None]:
    """Load a checkpoint blueprint only when it is safe to skip Phase 1."""
    state = CheckpointState.load_or_none(checkpoint_path)
    if state is None:
        return None, None
    if state.theorem_stmt.strip() != theorem_stmt.strip():
        raise RuntimeError(
            f"Cannot resume {checkpoint_path}: checkpoint theorem statement does not match the input."
        )

    blueprint = state.get_blueprint()
    if (
        blueprint is None
        or not blueprint.nodes
        or not state.blueprint_fully_validated
    ):
        return state, None
    return state, blueprint


def run_onepass_record(
    *,
    record_id: str,
    theorem_stmt: str,
    nl_proof: str,
    model: str,
    output_root: Path,
    node_timeout_s: int = 300,
    resume: bool = False,
    compiler: AbstractLeanCompiler | None = None,
    compiler_factory: Callable[[], AbstractLeanCompiler] | None = LeanCompiler,
) -> dict[str, Any]:
    checkpoint_path = output_root / "checkpoints" / f"{record_id}.json"
    trace_path = output_root / "traces" / f"{record_id}.jsonl"
    blueprint_path = output_root / "blueprints" / f"{record_id}.lean"

    if not resume:
        for path in (checkpoint_path, trace_path, blueprint_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    resume_state: CheckpointState | None = None
    blueprint: Blueprint | None = None
    blueprint_reused = False
    if resume:
        resume_state, blueprint = _load_resumable_blueprint(checkpoint_path, theorem_stmt)
        blueprint_reused = blueprint is not None
        cached_score = _checkpoint_score(checkpoint_path) if blueprint_reused else None
        if (
            resume_state is not None
            and resume_state.done
            and resume_state.success
            and cached_score is not None
            and cached_score["root_proved"]
        ):
            blueprint_path.parent.mkdir(parents=True, exist_ok=True)
            blueprint_path.write_text(blueprint.lean_file, encoding="utf-8")
            return {
                "id": record_id,
                "blueprint_success": cached_score["total_nodes"] > 0,
                "blueprint_reused": True,
                "phase1_skipped": True,
                "error": "",
                "checkpoint_path": str(checkpoint_path),
                "trace_path": str(trace_path),
                "blueprint_path": str(blueprint_path),
                **cached_score,
            }

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    blueprint_path.parent.mkdir(parents=True, exist_ok=True)

    tracer = JsonlTracer(trace_path)
    active_compiler = compiler or LeanCompiler()
    try:
        if blueprint_reused:
            assert blueprint is not None
            tracer.emit(TraceEvent(
                kind="resume",
                thm_name=record_id,
                args={
                    "blueprint_reused": True,
                    "proved_cache_count": len(resume_state.proved_cache) if resume_state else 0,
                },
                ok=True,
            ))
        else:
            blueprint = run_phase1(
                theorem_stmt=theorem_stmt,
                nl_proof=nl_proof or "",
                model=model,
                compiler=active_compiler,
                checkpoint_path=checkpoint_path,
                tracer=tracer,
                thm_name=record_id,
            )
        blueprint_path.write_text(blueprint.lean_file, encoding="utf-8")

        if not blueprint.nodes:
            score = _blueprint_score(blueprint, None)
            return {
                "id": record_id,
                "blueprint_success": False,
                "blueprint_reused": blueprint_reused,
                "phase1_skipped": blueprint_reused,
                "error": "phase1 generated zero blueprint nodes",
                "checkpoint_path": str(checkpoint_path),
                "trace_path": str(trace_path),
                "blueprint_path": str(blueprint_path),
                **score,
            }

        orch_result = run_phase2(
            checkpoint_path=checkpoint_path,
            compiler=active_compiler,
            compiler_factory=compiler_factory,
            retrieval=MathlibRetrieval(),
            tracer=tracer,
            node_timeout_s=node_timeout_s,
            model=model,
        )
        score = _blueprint_score(blueprint, orch_result)
        return {
            "id": record_id,
            "blueprint_success": True,
            "blueprint_reused": blueprint_reused,
            "phase1_skipped": blueprint_reused,
            "error": "",
            "checkpoint_path": str(checkpoint_path),
            "trace_path": str(trace_path),
            "blueprint_path": str(blueprint_path),
            **score,
        }
    except Exception as exc:
        cached_score = _checkpoint_score(checkpoint_path) if blueprint_reused else None
        score = cached_score or _blueprint_score(blueprint, None)
        return {
            "id": record_id,
            "blueprint_success": bool(
                blueprint is not None
                and blueprint.nodes
                and blueprint.fully_validated
            ),
            "blueprint_reused": blueprint_reused,
            "phase1_skipped": blueprint_reused,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "checkpoint_path": str(checkpoint_path),
            "trace_path": str(trace_path),
            "blueprint_path": str(blueprint_path),
            **score,
        }
    finally:
        tracer.close()

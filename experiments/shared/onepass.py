from __future__ import annotations

import asyncio
import traceback
from collections.abc import Callable
from concurrent.futures import Executor
from pathlib import Path
from typing import Any

from blueprint import Blueprint, phase2_contract_error_counts, phase2_contract_errors
from checkpoint import CheckpointState
from lean_compiler import AbstractLeanCompiler, LeanCompiler
from mathlib_retrieval import MathlibRetrieval
from pipeline import run_phase1, run_phase2, run_phase2_async
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
) -> tuple[CheckpointState | None, Blueprint | None, list[str]]:
    """Load a checkpoint blueprint only when it is safe to skip Phase 1."""
    state = CheckpointState.load_or_none(checkpoint_path)
    if state is None:
        return None, None, []
    if state.theorem_stmt.strip() != theorem_stmt.strip():
        raise RuntimeError(
            f"Cannot resume {checkpoint_path}: checkpoint theorem statement does not match the input."
        )

    blueprint = state.get_blueprint()
    if blueprint is None:
        return state, None, ["missing_blueprint: checkpoint contains no blueprint"]
    if not blueprint.nodes:
        return state, None, ["empty_blueprint: checkpoint blueprint contains no nodes"]
    if not state.blueprint_fully_validated:
        return state, None, ["unvalidated_blueprint: checkpoint blueprint was not fully validated"]
    contract_errors = phase2_contract_errors(blueprint)
    if contract_errors:
        return state, None, contract_errors
    return state, blueprint, []


def _paths(output_root: Path, record_id: str) -> tuple[Path, Path, Path]:
    return (
        output_root / "checkpoints" / f"{record_id}.json",
        output_root / "traces" / f"{record_id}.jsonl",
        output_root / "blueprints" / f"{record_id}.lean",
    )


def _base_row(
    *,
    record_id: str,
    checkpoint_path: Path,
    trace_path: Path,
    blueprint_path: Path,
    blueprint_success: bool,
    blueprint_reused: bool,
    phase1_skipped: bool,
    error: str = "",
) -> dict[str, Any]:
    return {
        "id": record_id,
        "blueprint_success": blueprint_success,
        "blueprint_reused": blueprint_reused,
        "phase1_skipped": phase1_skipped,
        "error": error,
        "checkpoint_path": str(checkpoint_path),
        "trace_path": str(trace_path),
        "blueprint_path": str(blueprint_path),
    }


def _resume_meta(invalid_errors: list[str]) -> dict[str, Any]:
    error_counts = phase2_contract_error_counts(invalid_errors)
    return {
        "resume_blueprint_invalid_errors": list(invalid_errors),
        "resume_blueprint_invalid_categories": {category: 1 for category in error_counts},
        "resume_blueprint_invalid_error_counts": error_counts,
    }


def run_onepass_phase1(
    *,
    record_id: str,
    theorem_stmt: str,
    nl_proof: str,
    model: str,
    output_root: Path,
    resume: bool = False,
    compiler: AbstractLeanCompiler | None = None,
) -> dict[str, Any]:
    checkpoint_path, trace_path, blueprint_path = _paths(output_root, record_id)

    if not resume:
        for path in (checkpoint_path, trace_path, blueprint_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    resume_state: CheckpointState | None = None
    blueprint: Blueprint | None = None
    blueprint_reused = False
    resume_invalid_errors: list[str] = []
    if resume:
        resume_state, blueprint, resume_invalid_errors = _load_resumable_blueprint(checkpoint_path, theorem_stmt)
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
                "status": "complete",
                "row": {
                    **_base_row(
                        record_id=record_id,
                        checkpoint_path=checkpoint_path,
                        trace_path=trace_path,
                        blueprint_path=blueprint_path,
                        blueprint_success=cached_score["total_nodes"] > 0,
                        blueprint_reused=True,
                        phase1_skipped=True,
                    ),
                    **cached_score,
                },
                "blueprint_reused": True,
                "phase1_skipped": True,
                "checkpoint_path": str(checkpoint_path),
                "trace_path": str(trace_path),
                "blueprint_path": str(blueprint_path),
                "blueprint_success": cached_score["total_nodes"] > 0,
                **_resume_meta(resume_invalid_errors),
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
                "status": "failed",
                "row": {
                    **_base_row(
                        record_id=record_id,
                        checkpoint_path=checkpoint_path,
                        trace_path=trace_path,
                        blueprint_path=blueprint_path,
                        blueprint_success=False,
                        blueprint_reused=blueprint_reused,
                        phase1_skipped=blueprint_reused,
                        error="phase1 generated zero blueprint nodes",
                    ),
                    **score,
                },
                "blueprint_reused": blueprint_reused,
                "phase1_skipped": blueprint_reused,
                "checkpoint_path": str(checkpoint_path),
                "trace_path": str(trace_path),
                "blueprint_path": str(blueprint_path),
                "blueprint_success": False,
                **_resume_meta(resume_invalid_errors),
                **score,
            }

        return {
            "status": "ready",
            "blueprint": blueprint,
            "blueprint_reused": blueprint_reused,
            "phase1_skipped": blueprint_reused,
            "checkpoint_path": str(checkpoint_path),
            "trace_path": str(trace_path),
            "blueprint_path": str(blueprint_path),
            "blueprint_success": True,
            **_resume_meta(resume_invalid_errors),
        }
    except Exception as exc:
        cached_score = _checkpoint_score(checkpoint_path) if blueprint_reused else None
        score = cached_score or _blueprint_score(blueprint, None)
        return {
            "status": "failed",
            "row": {
                **_base_row(
                    record_id=record_id,
                    checkpoint_path=checkpoint_path,
                    trace_path=trace_path,
                    blueprint_path=blueprint_path,
                    blueprint_success=bool(
                        blueprint is not None
                        and blueprint.nodes
                        and getattr(blueprint, "fully_validated", True)
                    ),
                    blueprint_reused=blueprint_reused,
                    phase1_skipped=blueprint_reused,
                    error=str(exc),
                ),
                "traceback": traceback.format_exc(),
                **score,
            },
            "blueprint_reused": blueprint_reused,
            "phase1_skipped": blueprint_reused,
            "checkpoint_path": str(checkpoint_path),
            "trace_path": str(trace_path),
            "blueprint_path": str(blueprint_path),
            "blueprint_success": bool(
                blueprint is not None
                and blueprint.nodes
                and getattr(blueprint, "fully_validated", True)
            ),
            **_resume_meta(resume_invalid_errors),
            **score,
        }
    finally:
        tracer.close()


def _load_phase2_blueprint(checkpoint_path: Path, theorem_stmt: str) -> Blueprint:
    state = CheckpointState.load(checkpoint_path)
    if state.theorem_stmt.strip() != theorem_stmt.strip():
        raise RuntimeError(
            f"Cannot run Phase 2 for {checkpoint_path}: checkpoint theorem statement does not match the input."
        )
    blueprint = state.get_blueprint()
    if blueprint is None:
        raise RuntimeError(f"No blueprint in checkpoint {checkpoint_path} - run Phase 1 first.")
    return blueprint


def _phase2_row(
    *,
    record_id: str,
    output_root: Path,
    blueprint: Blueprint | None,
    orch_result: Any | None,
    blueprint_reused: bool,
    phase1_skipped: bool,
    error: str = "",
    traceback_text: str | None = None,
) -> dict[str, Any]:
    checkpoint_path, trace_path, blueprint_path = _paths(output_root, record_id)
    row = {
        **_base_row(
            record_id=record_id,
            checkpoint_path=checkpoint_path,
            trace_path=trace_path,
            blueprint_path=blueprint_path,
            blueprint_success=bool(blueprint and blueprint.nodes),
            blueprint_reused=blueprint_reused,
            phase1_skipped=phase1_skipped,
            error=error,
        ),
        **_blueprint_score(blueprint, orch_result),
    }
    if traceback_text is not None:
        row["traceback"] = traceback_text
    return row


async def run_onepass_phase2_async(
    *,
    record_id: str,
    theorem_stmt: str,
    model: str,
    output_root: Path,
    node_timeout_s: int = 300,
    compiler: AbstractLeanCompiler | None = None,
    compiler_factory: Callable[[], AbstractLeanCompiler] | None = LeanCompiler,
    blueprint_reused: bool = False,
    phase1_skipped: bool = False,
    node_executor: Executor | None = None,
    node_semaphore: asyncio.Semaphore | None = None,
    blueprint_hint: Blueprint | None = None,
) -> dict[str, Any]:
    checkpoint_path, trace_path, _blueprint_path = _paths(output_root, record_id)
    tracer = JsonlTracer(trace_path)
    active_compiler = compiler or LeanCompiler()
    blueprint: Blueprint | None = blueprint_hint
    try:
        if blueprint is None:
            blueprint = _load_phase2_blueprint(checkpoint_path, theorem_stmt)
        orch_result = await run_phase2_async(
            checkpoint_path=checkpoint_path,
            compiler=active_compiler,
            compiler_factory=compiler_factory,
            retrieval=MathlibRetrieval(),
            tracer=tracer,
            node_timeout_s=node_timeout_s,
            model=model,
            node_executor=node_executor,
            node_semaphore=node_semaphore,
        )
        return _phase2_row(
            record_id=record_id,
            output_root=output_root,
            blueprint=blueprint,
            orch_result=orch_result,
            blueprint_reused=blueprint_reused,
            phase1_skipped=phase1_skipped,
        )
    except Exception as exc:
        if blueprint is None:
            try:
                blueprint = _load_phase2_blueprint(checkpoint_path, theorem_stmt)
            except Exception:
                blueprint = None
        return _phase2_row(
            record_id=record_id,
            output_root=output_root,
            blueprint=blueprint,
            orch_result=None,
            blueprint_reused=blueprint_reused,
            phase1_skipped=phase1_skipped,
            error=str(exc),
            traceback_text=traceback.format_exc(),
        )
    finally:
        tracer.close()


def run_onepass_phase2(
    *,
    record_id: str,
    theorem_stmt: str,
    model: str,
    output_root: Path,
    node_timeout_s: int = 300,
    compiler: AbstractLeanCompiler | None = None,
    compiler_factory: Callable[[], AbstractLeanCompiler] | None = LeanCompiler,
    blueprint_reused: bool = False,
    phase1_skipped: bool = False,
    blueprint_hint: Blueprint | None = None,
) -> dict[str, Any]:
    checkpoint_path, trace_path, _blueprint_path = _paths(output_root, record_id)
    tracer = JsonlTracer(trace_path)
    active_compiler = compiler or LeanCompiler()
    blueprint: Blueprint | None = blueprint_hint
    try:
        if blueprint is None:
            blueprint = _load_phase2_blueprint(checkpoint_path, theorem_stmt)
        orch_result = run_phase2(
            checkpoint_path=checkpoint_path,
            compiler=active_compiler,
            compiler_factory=compiler_factory,
            retrieval=MathlibRetrieval(),
            tracer=tracer,
            node_timeout_s=node_timeout_s,
            model=model,
        )
        return _phase2_row(
            record_id=record_id,
            output_root=output_root,
            blueprint=blueprint,
            orch_result=orch_result,
            blueprint_reused=blueprint_reused,
            phase1_skipped=phase1_skipped,
        )
    except Exception as exc:
        if blueprint is None:
            try:
                blueprint = _load_phase2_blueprint(checkpoint_path, theorem_stmt)
            except Exception:
                blueprint = None
        return _phase2_row(
            record_id=record_id,
            output_root=output_root,
            blueprint=blueprint,
            orch_result=None,
            blueprint_reused=blueprint_reused,
            phase1_skipped=phase1_skipped,
            error=str(exc),
            traceback_text=traceback.format_exc(),
        )
    finally:
        tracer.close()


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
    phase1 = run_onepass_phase1(
        record_id=record_id,
        theorem_stmt=theorem_stmt,
        nl_proof=nl_proof,
        model=model,
        output_root=output_root,
        resume=resume,
        compiler=compiler,
    )
    if phase1["status"] in {"complete", "failed"}:
        return phase1["row"]
    return run_onepass_phase2(
        record_id=record_id,
        theorem_stmt=theorem_stmt,
        model=model,
        output_root=output_root,
        node_timeout_s=node_timeout_s,
        compiler=compiler,
        compiler_factory=compiler_factory,
        blueprint_reused=bool(phase1.get("blueprint_reused")),
        phase1_skipped=bool(phase1.get("phase1_skipped")),
        blueprint_hint=phase1.get("blueprint"),
    )

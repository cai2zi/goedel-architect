from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from blueprint import Blueprint
from checkpoint import CheckpointState
from lean_compiler import LeanCompiler
from mathlib_retrieval import MathlibRetrieval
from pipeline import run_phase1, run_phase2
from tracer import JsonlTracer


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
    proved = {
        name
        for name, result in state.node_results.items()
        if getattr(result, "success", False)
    }
    failed = set(state.node_results) - proved

    class Result:
        pass

    result = Result()
    result.proved = proved
    result.failed = failed
    return _blueprint_score(blueprint, result)


def run_onepass_record(
    *,
    record_id: str,
    theorem_stmt: str,
    nl_proof: str,
    model: str,
    output_root: Path,
    node_timeout_s: int = 300,
    resume: bool = False,
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

    if resume:
        cached_score = _checkpoint_score(checkpoint_path)
        if cached_score is not None:
            return {
                "id": record_id,
                "blueprint_success": cached_score["total_nodes"] > 0,
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
    compiler = LeanCompiler()
    blueprint: Blueprint | None = None
    try:
        blueprint = run_phase1(
            theorem_stmt=theorem_stmt,
            nl_proof=nl_proof or "",
            model=model,
            compiler=compiler,
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
                "error": "phase1 generated zero blueprint nodes",
                "checkpoint_path": str(checkpoint_path),
                "trace_path": str(trace_path),
                "blueprint_path": str(blueprint_path),
                **score,
            }

        orch_result = run_phase2(
            checkpoint_path=checkpoint_path,
            compiler=compiler,
            compiler_factory=LeanCompiler,
            retrieval=MathlibRetrieval(),
            tracer=tracer,
            node_timeout_s=node_timeout_s,
            model=model,
        )
        score = _blueprint_score(blueprint, orch_result)
        return {
            "id": record_id,
            "blueprint_success": True,
            "error": "",
            "checkpoint_path": str(checkpoint_path),
            "trace_path": str(trace_path),
            "blueprint_path": str(blueprint_path),
            **score,
        }
    except Exception as exc:
        score = _blueprint_score(blueprint, None)
        return {
            "id": record_id,
            "blueprint_success": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "checkpoint_path": str(checkpoint_path),
            "trace_path": str(trace_path),
            "blueprint_path": str(blueprint_path),
            **score,
        }
    finally:
        tracer.close()

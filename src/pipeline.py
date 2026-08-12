"""Kimina-only Blueprint -> Proving -> Refinement pipeline."""
from __future__ import annotations
import asyncio
from concurrent.futures import Executor
from pathlib import Path

from blueprint import (
    Blueprint,
    _emit_semantic_check,
    _enabled_semantic_issues,
    format_phase2_contract_errors,
    phase2_contract_errors,
    render_solved_declaration,
)
from checkpoint import CheckpointState, RunStatus
from kimina_lean_compiler import KiminaInfrastructureError, KiminaLeanCompiler
from mathlib_retrieval import MathlibRetrieval
from orchestrator import (
    NodeResult,
    OrchestratorResult,
    active_node_names,
    prove_dag,
)
from prover import ProofSignal, ProverResult
from refinement import SemanticRefinementError, refine_blueprint
from semantic_fidelity import (
    format_semantic_issues,
    parse_cot_manifest,
    semantic_snapshot_from_dict,
    snapshot_blueprint_semantics,
    validate_blueprint_fidelity,
)
from tracer import NullTracer, TraceEvent


MAX_REFINEMENT_ITERATIONS = 8


def _invalidate_stale_proofs(
    blueprint: Blueprint,
    proved_cache: dict[str, str],
    proof_cache_keys: dict[str, str],
) -> dict[str, str]:
    active = active_node_names(blueprint)
    return {
        name: proof
        for name, proof in proved_cache.items()
        if name in active
        and (node := blueprint.node_by_name(name)) is not None
        and node.kind in {"lemma", "theorem"}
        and proof_cache_keys.get(name) == node.cache_key()
    }


def _proof_bodies(
    blueprint: Blueprint,
    orch_result: OrchestratorResult | None,
    proved_cache: dict[str, str] | None = None,
) -> dict[str, str]:
    bodies = dict(proved_cache or {})
    if orch_result is not None:
        for name, node_result in orch_result.node_results.items():
            if node_result.result.signal == ProofSignal.SOLVED and node_result.result.proof_body:
                bodies[name] = node_result.result.proof_body
    return bodies


def _assemble_final_file(
    blueprint: Blueprint,
    orch_result: OrchestratorResult | None = None,
    proved_cache: dict[str, str] | None = None,
) -> str:
    active = active_node_names(blueprint)
    bodies = _proof_bodies(blueprint, orch_result, proved_cache)
    parts = [blueprint.phase2_header.rstrip()]
    for node in blueprint.dependency_order():
        if node.name not in active:
            continue
        if node.kind == "definition":
            parts.append(node.full_declaration().strip())
            continue
        body = bodies.get(node.name)
        if not body:
            raise ValueError(f"Active proof node `{node.name}` has no solved proof")
        parts.append(render_solved_declaration(node, body).strip())
    return "\n\n".join(part for part in parts if part) + "\n"


def _assemble_partial_file(
    blueprint: Blueprint,
    orch_result: OrchestratorResult | None,
    proved_cache: dict[str, str],
) -> str:
    active = active_node_names(blueprint)
    bodies = _proof_bodies(blueprint, orch_result, proved_cache)
    parts = [blueprint.phase2_header.rstrip()]
    for node in blueprint.dependency_order():
        if node.name not in active:
            continue
        body = bodies.get(node.name)
        if node.kind == "definition":
            parts.append(node.full_declaration().strip())
        elif body:
            parts.append(render_solved_declaration(node, body).strip())
        else:
            parts.append(node.full_declaration().strip())
    return "\n\n".join(part for part in parts if part) + "\n"



async def run_phase2_async(
    *,
    checkpoint_path: Path,
    compiler: KiminaLeanCompiler,
    retrieval: MathlibRetrieval | None = None,
    tracer=None,
    node_timeout_s: float | None = 300.0,
    llm_api_timeout_s: float | None = 120.0,
    model: str | None = None,
    node_max_prove_turns: int | None = None,
    node_max_negation_probe_turns: int = 1,
    max_tool_calls_per_turn: int = 3,
    proof_policy: str = "full",
    critical_negation_max_turns: int = 0,
    node_executor: Executor | None = None,
    node_semaphore: asyncio.Semaphore | None = None,
) -> OrchestratorResult:
    state = CheckpointState.load(checkpoint_path)
    blueprint = state.get_blueprint()
    if blueprint is None:
        raise RuntimeError(f"No blueprint in checkpoint {checkpoint_path}")
    if state.semantic_fidelity_enabled:
        semantic_issues = _enabled_semantic_issues(
            validate_blueprint_fidelity(
                blueprint,
                parse_cot_manifest(state.cot_manifest_json),
                claimed_answer=state.claimed_answer,
                require_step_bindings=state.semantic_require_step_ids,
            ),
            require_step_ids=state.semantic_require_step_ids,
            static_gate=state.semantic_static_gate,
        )
        _emit_semantic_check(
            tracer,
            thm_name=blueprint.target_theorem,
            phase="phase2_checkpoint",
            attempt=state.iteration + 1,
            issues=semantic_issues,
        )
        semantic_errors = [
            issue for issue in semantic_issues if issue.severity == "error"
        ]
        state.semantic_gate_results.append({
            "stage": "phase2_checkpoint",
            "passed": not semantic_errors,
            "issues": [issue.to_dict() for issue in semantic_issues],
            "warning_count": sum(
                issue.severity == "warning" for issue in semantic_issues
            ),
        })
        if semantic_errors:
            state.status = RunStatus.ERROR
            state.semantic_status = "phase2_checkpoint_rejected"
            state.final_lean_errors = [format_semantic_issues(semantic_errors)]
            state.save(checkpoint_path)
            return OrchestratorResult(
                active_nodes=active_node_names(blueprint),
                root_name=blueprint.target_theorem,
            )
    contract_errors = phase2_contract_errors(blueprint)
    if contract_errors:
        raise RuntimeError(format_phase2_contract_errors(contract_errors))
    retrieval = retrieval or MathlibRetrieval()
    state.proved_cache = _invalidate_stale_proofs(
        blueprint, state.proved_cache, state.proof_cache_keys,
    )
    active = active_node_names(blueprint)
    nodes_to_try = active - set(state.proved_cache)
    orch_result = await prove_dag(
        blueprint=blueprint,
        compiler=compiler,
        retrieval=retrieval,
        model=model or state.model,
        proved_cache=state.proved_cache,
        nodes_to_retry=nodes_to_try,
        tracer=tracer,
        node_timeout_s=node_timeout_s,
        llm_api_timeout_s=llm_api_timeout_s,
        node_max_prove_turns=node_max_prove_turns,
        node_max_negation_probe_turns=node_max_negation_probe_turns,
        max_tool_calls_per_turn=max_tool_calls_per_turn,
        proof_policy=proof_policy,
        critical_negation_max_turns=critical_negation_max_turns,
        node_executor=node_executor,
        node_semaphore=node_semaphore,
    )
    for name, node_result in orch_result.node_results.items():
        node = blueprint.node_by_name(name)
        if (
            node is not None
            and node.kind in {"lemma", "theorem"}
            and node_result.result.signal == ProofSignal.SOLVED
        ):
            state.proved_cache[name] = node_result.result.proof_body
            state.proof_cache_keys[name] = node.cache_key()
    state.set_node_results(orch_result.node_results)

    infra_nodes = [
        name for name, node_result in orch_result.node_results.items()
        if node_result.result.signal == ProofSignal.INFRA_ERROR
    ]
    if infra_nodes:
        state.status = RunStatus.ERROR
        state.save(checkpoint_path)
        return orch_result

    root_result = orch_result.node_results.get(blueprint.target_theorem)
    if root_result is not None and root_result.result.signal == ProofSignal.SOLVED:
        final_lean = _assemble_final_file(blueprint, orch_result, state.proved_cache)
        final_result = compiler.check(final_lean, allow_sorry=False)
        final_diagnostics = final_result.diagnostics
        (tracer or NullTracer()).emit(TraceEvent(
            kind="final_verify",
            thm_name=blueprint.target_theorem,
            ok=final_result.success,
            args={
                "lean_errors": final_diagnostics,
                "failure_kind": final_result.failure_kind,
                "lean_code": final_lean,
            },
        ))
        state.final_lean_file = final_lean
        state.final_lean_errors = final_diagnostics
        if final_result.success:
            state.status = RunStatus.SOLVED
        elif final_result.failure_kind == "infra":
            state.status = RunStatus.ERROR
        else:
            root_name = blueprint.target_theorem
            root_node = blueprint.node_by_name(root_name)
            root_proof = state.proved_cache.pop(root_name, "")
            state.proof_cache_keys.pop(root_name, None)
            if root_node is not None:
                root_failure = NodeResult(
                    root_node,
                    ProverResult(
                        ProofSignal.PROOF_TOO_HARD,
                        root_proof,
                        final_diagnostics,
                    ),
                )
                orch_result.node_results[root_name] = root_failure
                state.set_node_results(orch_result.node_results)
            state.status = RunStatus.RUNNING
    else:
        state.status = RunStatus.RUNNING
    state.save(checkpoint_path)
    return orch_result


def run_phase2(**kwargs) -> OrchestratorResult:
    return asyncio.run(run_phase2_async(**kwargs))


def run_phase3(
    *,
    checkpoint_path: Path,
    compiler: KiminaLeanCompiler,
    model: str | None = None,
    max_iterations: int = MAX_REFINEMENT_ITERATIONS,
    tracer=None,
    thm_name: str = "",
    blueprint_max_retries: int | None = None,
    phase2_contract_check_concurrency: int = 1,
    semantic_fidelity_enabled: bool | None = None,
    semantic_require_step_ids: bool | None = None,
    semantic_static_gate: bool | None = None,
    semantic_freeze_refinement: bool | None = None,
) -> Blueprint:
    state = CheckpointState.load(checkpoint_path)
    if state.status != RunStatus.RUNNING:
        raise RuntimeError(f"Checkpoint status is {state.status.value}; refinement is not allowed")
    blueprint = state.get_blueprint()
    if blueprint is None or not state.node_results:
        raise RuntimeError("Phase 3 requires a blueprint and Phase 2 node results")
    if state.iteration >= max_iterations:
        raise RuntimeError("Refinement iteration limit reached")
    orch_result = _orch_result_from_checkpoint(state, blueprint)
    history = list(state.refinement_history)
    try:
        revised = refine_blueprint(
            blueprint=blueprint,
            orch_result=orch_result,
            compiler=compiler,
            model=model or state.model,
            history=history,
            iteration=state.iteration,
            max_iterations=max_iterations,
            tracer=tracer,
            thm_name=thm_name,
            max_retries=blueprint_max_retries or 8,
            phase2_contract_check_concurrency=phase2_contract_check_concurrency,
            informal_statement=state.informal_statement,
            informal_proof=state.informal_proof,
            cot_manifest_json=state.cot_manifest_json,
            claimed_answer=state.claimed_answer,
            semantic_fidelity_enabled=(
                state.semantic_fidelity_enabled
                if semantic_fidelity_enabled is None
                else semantic_fidelity_enabled
            ),
            semantic_require_step_ids=(
                state.semantic_require_step_ids
                if semantic_require_step_ids is None
                else semantic_require_step_ids
            ),
            semantic_static_gate=(
                state.semantic_static_gate
                if semantic_static_gate is None
                else semantic_static_gate
            ),
            semantic_freeze_refinement=(
                state.semantic_freeze_refinement
                if semantic_freeze_refinement is None
                else semantic_freeze_refinement
            ),
            baseline_semantic_snapshot=(
                semantic_snapshot_from_dict(state.semantic_contract_snapshot)
                if state.semantic_contract_snapshot
                else None
            ),
        )
    except KiminaInfrastructureError as exc:
        state.status = RunStatus.ERROR
        state.final_lean_errors = [str(exc)]
        state.refinement_history = history
        state.save(checkpoint_path)
        return blueprint
    except SemanticRefinementError as exc:
        state.status = RunStatus.EXHAUSTED
        state.semantic_status = "refinement_rejected"
        state.semantic_gate_results.append({
            "stage": "phase3_local_gate",
            "passed": False,
            "issues": [
                issue.to_dict() if hasattr(issue, "to_dict") else {"message": str(issue)}
                for issue in exc.issues
            ],
            "message": str(exc),
        })
        state.final_lean_errors = [str(exc)]
        state.refinement_history = history
        state.save(checkpoint_path)
        return blueprint
    except RuntimeError as exc:
        state.status = RunStatus.EXHAUSTED
        state.final_lean_errors = [str(exc)]
        state.refinement_history = history
        state.save(checkpoint_path)
        return blueprint
    state.proved_cache = _invalidate_stale_proofs(
        revised, state.proved_cache, state.proof_cache_keys,
    )
    state.semantic_gate_results.extend(revised.semantic_gate_results)
    if state.semantic_fidelity_enabled:
        state.semantic_status = "refinement_accepted"
    state.proof_cache_keys = {
        name: key for name, key in state.proof_cache_keys.items()
        if name in state.proved_cache
    }
    state.set_blueprint(revised)
    state.refinement_history = history
    state.iteration += 1
    state.node_results = {}
    state.final_lean_file = ""
    state.final_lean_errors = []
    state.status = RunStatus.RUNNING
    state.save(checkpoint_path)
    return revised


def _orch_result_from_checkpoint(
    state: CheckpointState,
    blueprint: Blueprint,
) -> OrchestratorResult:
    results = state.get_prover_results()
    return OrchestratorResult(
        node_results={
            name: NodeResult(node, proof_result)
            for name, proof_result in results.items()
            if (node := blueprint.node_by_name(name)) is not None
        },
        active_nodes=active_node_names(blueprint),
        root_name=blueprint.target_theorem,
    )

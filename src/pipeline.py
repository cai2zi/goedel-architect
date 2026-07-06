"""Top-level pipeline: Blueprint → Parallel Proving → Refinement loop.

Wires all three phases together and runs up to 8 refinement iterations
(matching Appendix A of the paper).

The `compiler` parameter is injectable so the same pipeline works with:
  - LeanCompiler (standalone Lean projects via `lake env lean`)
  - VSBLeanCompiler (VeriSoftBench repos via LeanREPL)
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from blueprint import Blueprint, generate_blueprint
from checkpoint import CheckpointState
from lean_compiler import AbstractLeanCompiler, LeanCompiler
from mathlib_retrieval import MathlibRetrieval
from orchestrator import NodeResult, OrchestratorResult, prove_dag
from refinement import refine_blueprint
from tracer import NullTracer

# From Appendix A
MAX_REFINEMENT_ITERATIONS = 8


@dataclass
class ProofResult:
    success: bool
    theorem_name: str
    proof_body: str = ""        # proof body of the root node if solved
    final_lean_file: str = ""   # full assembled Lean file
    iterations: int = 0
    proved_nodes: list[str] = field(default_factory=list)
    failed_nodes: list[str] = field(default_factory=list)


def prove_theorem(
    theorem_stmt: str,
    nl_proof: str | None = None,
    model: str = "gpt-5.5",
    compiler: AbstractLeanCompiler | None = None,
    compiler_factory: Callable[[], AbstractLeanCompiler] | None = None,
    retrieval: MathlibRetrieval | None = None,
    repo_retrieval=None,
    max_iterations: int = MAX_REFINEMENT_ITERATIONS,
    tracer=None,
    project_root: Path | None = None,
    repo_context: str | None = None,
    node_timeout_s: float | None = 300.0,
    checkpoint_path: Path | None = None,
) -> ProofResult:
    """
    Full Goedel-Architect pipeline for a single theorem.

    1. Generate @[blueprint]-annotated dependency graph (Phase 1)
    2. Prove each node in parallel (Phase 2)
    3. Refine blueprint on failures and repeat (Phase 3)
    Up to `max_iterations` refinement loops (default 8, per Appendix A).

    Args:
        compiler: Shared compiler instance (used for all nodes).
        compiler_factory: Called once per node to get a fresh compiler.
            Use this for VSBLeanCompiler which tracks call state per-theorem.
            If both are provided, compiler_factory takes precedence.
        repo_retrieval: Optional RepoRetrieval for repo_search tool.
        tracer: Optional tracer for emitting events.
        node_timeout_s: Per-node wall-clock bound in Phase 2 (see
            orchestrator.prove_dag). None disables the bound.
        checkpoint_path: If given, state is saved after every phase and, if
            the file already exists, resumed from wherever it left off
            (skipping Phase 1 and any already-proved nodes). See checkpoint.py
            and run_phase1/run_phase2/run_phase3 below for running phases
            standalone instead of through this all-in-one loop.
    """
    tracer = tracer or NullTracer()

    if compiler is None and compiler_factory is None:
        root = project_root or Path(__file__).parent.parent / "goedel_lean"
        compiler = LeanCompiler(root)

    retrieval = retrieval or MathlibRetrieval()

    state = CheckpointState.load_or_none(checkpoint_path)

    if state and state.done:
        # A prior run already finished this theorem (success or exhausted
        # all iterations) — reconstruct the result from the checkpoint
        # instead of re-running Phase 2/3 (which would burn API calls
        # re-deriving an answer that's already on disk).
        print(f"[Resume] checkpoint at {checkpoint_path} already done "
              f"(success={state.success}) — returning cached result", flush=True)
        return _proof_result_from_checkpoint(state)

    resumed_blueprint = state.get_blueprint() if state else None

    if resumed_blueprint is not None:
        blueprint = resumed_blueprint
        proved_cache: dict[str, str] = dict(state.proved_cache)
        refinement_history: list[str] = list(state.refinement_history)
        start_iteration = state.iteration
        print(f"[Resume] loaded checkpoint at iteration {start_iteration + 1}, "
              f"{len(proved_cache)} node(s) already proved", flush=True)
    else:
        # Phase 1: Blueprint generation
        # Don't use compiler_factory for blueprint validation: factory compilers are
        # stateful (track call counts, write temp files) and Phase 1 would exhaust
        # retries on type-signature errors the LLM can't fix without repo context.
        # Pass only an explicitly-shared compiler (e.g. standalone LeanCompiler).
        blueprint_compiler = compiler  # None when only compiler_factory is provided
        blueprint = generate_blueprint(
            theorem_stmt=theorem_stmt,
            nl_proof=nl_proof,
            model=model,
            compiler=blueprint_compiler,
            repo_context=repo_context,
        )
        proved_cache = {}
        refinement_history = []
        start_iteration = 0
        state = CheckpointState(theorem_stmt=theorem_stmt, model=model, repo_context=repo_context or "")
        state.set_blueprint(blueprint)
        if checkpoint_path:
            state.save(checkpoint_path)

    orch_result: OrchestratorResult | None = None

    for iteration in range(start_iteration, max_iterations):
        # Phase 2: Parallel proving
        nodes_to_try = set(blueprint.nodes_by_name()) - set(proved_cache)
        print(f"\n[Phase 2 iteration {iteration+1}] Proving {len(nodes_to_try)} nodes: {sorted(nodes_to_try)}", flush=True)

        orch_result = asyncio.run(
            prove_dag(
                blueprint=blueprint,
                compiler=compiler,
                compiler_factory=compiler_factory,
                retrieval=retrieval,
                repo_retrieval=repo_retrieval,
                model=model,
                proved_cache=proved_cache,
                nodes_to_retry=nodes_to_try,
                tracer=tracer,
                node_timeout_s=node_timeout_s,
            )
        )

        for name, nr in orch_result.node_results.items():
            status = nr.result.signal.value
            proof_preview = repr(nr.result.proof_body[:60]) if nr.result.proof_body else ""
            print(f"  node '{name}': {status} {proof_preview}", flush=True)
            if status == "solved":
                proved_cache[name] = nr.result.proof_body

        print(f"  proved so far: {sorted(proved_cache.keys())}", flush=True)

        if checkpoint_path:
            state.iteration = iteration
            state.proved_cache = dict(proved_cache)
            state.set_node_results(orch_result.node_results)

        if orch_result.all_proved():
            root_name = blueprint.target_theorem
            root_proof = proved_cache.get(root_name, "")
            if checkpoint_path:
                state.done = True
                state.success = True
                state.save(checkpoint_path)
            return ProofResult(
                success=True,
                theorem_name=root_name,
                proof_body=root_proof,
                final_lean_file=_assemble_final_file(blueprint, orch_result),
                iterations=iteration + 1,
                proved_nodes=list(orch_result.proved),
                failed_nodes=[],
            )

        if checkpoint_path:
            state.save(checkpoint_path)

        if iteration == max_iterations - 1:
            break

        # Phase 3: Refinement
        print(f"\n[Phase 3 iteration {iteration+1}] Refining blueprint ...", flush=True)
        failed = [n for n in orch_result.node_results if orch_result.node_results[n].result.signal.value != "solved"]
        print(f"  failed nodes: {failed}", flush=True)
        refinement_compiler = compiler or (compiler_factory() if compiler_factory else None)
        if refinement_compiler is None:
            break
        try:
            blueprint = refine_blueprint(
                blueprint=blueprint,
                orch_result=orch_result,
                compiler=refinement_compiler,
                model=model,
                repo_context=repo_context,
                history=refinement_history,
                iteration=iteration,
                max_iterations=max_iterations,
            )
            print(f"  new blueprint has {len(blueprint.nodes)} nodes: {[n.name for n in blueprint.nodes]}", flush=True)
        except RuntimeError as e:
            print(f"  refinement failed: {e}", flush=True)
            break  # refinement failed, stop iterations

        if checkpoint_path:
            state.set_blueprint(blueprint)
            state.refinement_history = list(refinement_history)
            state.iteration = iteration + 1
            state.node_results = {}  # stale against the new blueprint
            state.save(checkpoint_path)

    proved = list(orch_result.proved) if orch_result else []
    failed = list(orch_result.failed.keys()) if orch_result else []
    root_proof = proved_cache.get(blueprint.target_theorem, "")
    if checkpoint_path:
        state.done = True
        state.success = False
        state.save(checkpoint_path)
    return ProofResult(
        success=False,
        theorem_name=blueprint.target_theorem,
        proof_body=root_proof,
        final_lean_file=_assemble_partial_file(blueprint, orch_result, proved_cache),
        iterations=max_iterations,
        proved_nodes=proved,
        failed_nodes=failed,
    )


# ---------------------------------------------------------------------------
# Standalone phase entry points
#
# Each function does exactly one phase against a checkpoint file on disk, so
# a caller can run e.g. Phase 2 without Phase 1 having just run in the same
# process (only having run at some point and left a checkpoint behind), and
# Phase 3 without re-running Phase 1 or Phase 2.
# ---------------------------------------------------------------------------

def run_phase1(
    theorem_stmt: str,
    nl_proof: str | None = None,
    model: str = "gpt-5.5",
    compiler: AbstractLeanCompiler | None = None,
    repo_context: str | None = None,
    checkpoint_path: Path | None = None,
) -> Blueprint:
    """Run Phase 1 (blueprint generation) alone and checkpoint the result."""
    blueprint = generate_blueprint(
        theorem_stmt=theorem_stmt,
        nl_proof=nl_proof,
        model=model,
        compiler=compiler,
        repo_context=repo_context,
    )
    if checkpoint_path:
        state = CheckpointState(theorem_stmt=theorem_stmt, model=model, repo_context=repo_context or "")
        state.set_blueprint(blueprint)
        state.save(checkpoint_path)
    return blueprint


def run_phase2(
    checkpoint_path: Path,
    compiler: AbstractLeanCompiler | None = None,
    compiler_factory: Callable[[], AbstractLeanCompiler] | None = None,
    retrieval: MathlibRetrieval | None = None,
    repo_retrieval=None,
    tracer=None,
    node_timeout_s: float | None = 300.0,
) -> OrchestratorResult:
    """Run one Phase 2 (parallel proving) pass against a checkpointed blueprint.

    Requires Phase 1 to have already produced a checkpoint at `checkpoint_path`
    (raises if it's missing or has no blueprint). Only nodes not already in
    `proved_cache` are attempted; the checkpoint is updated with the new
    `proved_cache` and `node_results` (the latter needed by Phase 3).
    """
    state = CheckpointState.load(checkpoint_path)
    blueprint = state.get_blueprint()
    if blueprint is None:
        raise RuntimeError(f"No blueprint in checkpoint {checkpoint_path} — run Phase 1 first.")

    retrieval = retrieval or MathlibRetrieval()
    proved_cache = dict(state.proved_cache)
    nodes_to_try = set(blueprint.nodes_by_name()) - set(proved_cache)

    orch_result = asyncio.run(
        prove_dag(
            blueprint=blueprint,
            compiler=compiler,
            compiler_factory=compiler_factory,
            retrieval=retrieval,
            repo_retrieval=repo_retrieval,
            model=state.model,
            proved_cache=proved_cache,
            nodes_to_retry=nodes_to_try,
            tracer=tracer,
            node_timeout_s=node_timeout_s,
        )
    )

    for name, nr in orch_result.node_results.items():
        if nr.result.signal.value == "solved":
            proved_cache[name] = nr.result.proof_body

    state.proved_cache = proved_cache
    state.set_node_results(orch_result.node_results)
    state.done = orch_result.all_proved()
    state.success = state.done
    state.save(checkpoint_path)
    return orch_result


def run_phase3(
    checkpoint_path: Path,
    compiler: AbstractLeanCompiler,
    model: str | None = None,
    repo_context: str | None = None,
    max_iterations: int = MAX_REFINEMENT_ITERATIONS,
) -> Blueprint:
    """Run one Phase 3 (refinement) pass against a checkpointed blueprint.

    Requires Phase 2 to have already run against this checkpoint (i.e.
    `node_results` present with at least one failure) — refinement needs
    those diagnostics to know what to fix. Raises if the checkpoint has no
    blueprint, no node results, or every node already solved.
    """
    state = CheckpointState.load(checkpoint_path)
    blueprint = state.get_blueprint()
    if blueprint is None:
        raise RuntimeError(f"No blueprint in checkpoint {checkpoint_path} — run Phase 1 first.")
    if not state.node_results:
        raise RuntimeError(f"No node results in checkpoint {checkpoint_path} — run Phase 2 first.")

    orch_result = _orch_result_from_checkpoint(state, blueprint)
    if orch_result.all_proved():
        raise RuntimeError(f"All nodes already proved in checkpoint {checkpoint_path} — nothing to refine.")

    refinement_history = list(state.refinement_history)
    new_blueprint = refine_blueprint(
        blueprint=blueprint,
        orch_result=orch_result,
        compiler=compiler,
        model=model or state.model,
        repo_context=repo_context if repo_context is not None else state.repo_context,
        history=refinement_history,
        iteration=state.iteration,
        max_iterations=max_iterations,
    )

    state.set_blueprint(new_blueprint)
    state.refinement_history = refinement_history
    state.iteration += 1
    state.node_results = {}  # stale against the new blueprint
    state.done = False
    state.success = False
    state.save(checkpoint_path)
    return new_blueprint


def _orch_result_from_checkpoint(state: CheckpointState, blueprint: Blueprint) -> OrchestratorResult:
    prover_results = state.get_prover_results()
    return OrchestratorResult(node_results={
        name: NodeResult(node=blueprint.node_by_name(name), result=pr)
        for name, pr in prover_results.items()
        if blueprint.node_by_name(name) is not None
    })


def _proof_result_from_checkpoint(state: CheckpointState) -> ProofResult:
    blueprint = state.get_blueprint()
    proved_cache = dict(state.proved_cache)
    orch_result = _orch_result_from_checkpoint(state, blueprint)
    root_name = blueprint.target_theorem
    root_proof = proved_cache.get(root_name, "")
    if state.success:
        return ProofResult(
            success=True,
            theorem_name=root_name,
            proof_body=root_proof,
            final_lean_file=_assemble_final_file(blueprint, orch_result),
            iterations=state.iteration + 1,
            proved_nodes=list(orch_result.proved),
            failed_nodes=[],
        )
    return ProofResult(
        success=False,
        theorem_name=root_name,
        proof_body=root_proof,
        final_lean_file=_assemble_partial_file(blueprint, orch_result, proved_cache),
        iterations=state.iteration + 1,
        proved_nodes=list(orch_result.proved),
        failed_nodes=list(orch_result.failed.keys()),
    )


def _substitute_proof(lean: str, name: str, proof_body: str) -> str:
    """Replace `name`'s `:= by sorry_using [...]` tail with a real proof body.

    Tolerant of whitespace/newlines between `by` and `sorry_using` (the model
    doesn't always keep them on one line), and uses a replacement function
    (not a template string) so backslashes in `proof_body` aren't
    misinterpreted as regex group references.
    """
    pattern = rf"(theorem|lemma)\s+{re.escape(name)}(.*?):=\s*by\s*sorry_using\s*\[.*?\]"
    return re.sub(
        pattern,
        lambda m: f"{m.group(1)} {name}{m.group(2)}:= {proof_body}",
        lean,
        flags=re.DOTALL,
    )


def _assemble_final_file(blueprint: Blueprint, orch_result: OrchestratorResult) -> str:
    lean = blueprint.lean_file
    for name, nr in orch_result.node_results.items():
        if nr.result.proof_body:
            lean = _substitute_proof(lean, name, nr.result.proof_body)
    return lean


def _assemble_partial_file(
    blueprint: Blueprint,
    orch_result: OrchestratorResult | None,
    proved_cache: dict[str, str],
) -> str:
    if orch_result is None:
        return blueprint.lean_file
    lean = blueprint.lean_file
    for name, body in proved_cache.items():
        lean = _substitute_proof(lean, name, body)
    return lean

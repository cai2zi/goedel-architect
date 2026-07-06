"""Phase 2: Parallel DAG traversal.

Proves blueprint nodes in topological order, running each wave in parallel.
Proved lemma bodies are threaded as context into dependent nodes.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable

import networkx as nx

from blueprint import Blueprint, BlueprintNode
from lean_compiler import AbstractLeanCompiler
from mathlib_retrieval import MathlibRetrieval
from prover import ProofSignal, ProverResult, prove_node
from tracer import NullTracer


@dataclass
class NodeResult:
    node: BlueprintNode
    result: ProverResult


@dataclass
class OrchestratorResult:
    node_results: dict[str, NodeResult] = field(default_factory=dict)

    @property
    def proved(self) -> set[str]:
        return {n for n, r in self.node_results.items() if r.result.signal == ProofSignal.SOLVED}

    @property
    def failed(self) -> dict[str, NodeResult]:
        return {n: r for n, r in self.node_results.items() if r.result.signal != ProofSignal.SOLVED}

    def all_proved(self) -> bool:
        # bool(self.node_results) guards against an empty result set being
        # vacuously "all proved" - a blueprint/refinement fallback that
        # produces zero nodes must never be reported as a successful proof.
        return bool(self.node_results) and not self.failed


def _transitive_deps(node: BlueprintNode, blueprint: Blueprint) -> set[str]:
    """All ancestor dependencies of `node`, direct and transitive.

    A node's own proof only ever references its *direct* `sorry_using [...]`
    deps by name, but a direct dependency's spliced-in proof body can itself
    reference ITS OWN dependencies (e.g. `add_comm_succ_case`'s proof calls
    `add_comm_succ_left_rewrite`, which is add_comm_succ_case's dependency,
    not add_comm's). Splicing only direct deps as aux lemmas leaves those
    transitive references unresolved - "unknown identifier" - even though
    the underlying lemma is fully proved.
    """
    seen: set[str] = set()
    stack = list(node.dependencies)
    while stack:
        dep = stack.pop()
        if dep in seen:
            continue
        seen.add(dep)
        dep_node = blueprint.node_by_name(dep)
        if dep_node:
            stack.extend(dep_node.dependencies)
    return seen


def _build_dag(blueprint: Blueprint) -> nx.DiGraph:
    dag = nx.DiGraph()
    node_names = {node.name for node in blueprint.nodes}
    for node in blueprint.nodes:
        dag.add_node(node.name)
    for node in blueprint.nodes:
        for dep in node.dependencies:
            if dep in node_names:
                dag.add_edge(dep, node.name)
    return dag


async def prove_dag(
    blueprint: Blueprint,
    compiler: AbstractLeanCompiler | None,
    retrieval: MathlibRetrieval,
    model: str = "gpt-4o",
    proved_cache: dict[str, str] | None = None,
    nodes_to_retry: set[str] | None = None,
    compiler_factory: Callable[[], AbstractLeanCompiler] | None = None,
    repo_retrieval=None,
    tracer=None,
    node_timeout_s: float | None = 300.0,
) -> OrchestratorResult:
    """
    Prove all nodes in the blueprint DAG in parallel waves.

    compiler_factory: if provided, called once per node to get a fresh compiler.
    node_timeout_s: wall-clock bound per node (covers the whole multi-turn tool
        loop). A node that exceeds this comes back as INFRA_ERROR instead of
        blocking the rest of the wave indefinitely. None disables the bound.
    """
    dag = _build_dag(blueprint)
    dag_node_names = set(dag.nodes)
    orch_result = OrchestratorResult()
    proof_bodies: dict[str, str] = dict(proved_cache or {})
    tracer = tracer or NullTracer()

    for name, body in proof_bodies.items():
        node = blueprint.node_by_name(name)
        if node:
            orch_result.node_results[name] = NodeResult(
                node=node,
                result=ProverResult(signal=ProofSignal.SOLVED, proof_body=body),
            )

    for generation in nx.topological_generations(dag):
        candidates = [
            name for name in generation
            if name not in proof_bodies
            and (nodes_to_retry is None or name in nodes_to_retry)
        ]
        if not candidates:
            continue

        # A node whose blueprint-graph dependency isn't solved yet will always
        # compile against an unresolved `sorry` stand-in for that dependency,
        # so it is guaranteed to fail with a generic "declaration uses 'sorry'"
        # error no matter what proof the model writes - and that error message
        # is indistinguishable from "your own tactic left a hole", so the model
        # can't tell it's structurally stuck and burns its whole tool-call
        # budget guessing tactics. Skip the doomed attempt entirely instead.
        wave: list[str] = []
        for name in candidates:
            node = blueprint.node_by_name(name)
            unresolved_deps = sorted(
                dep for dep in node.dependencies
                if dep in dag_node_names and dep not in proof_bodies
            )
            if unresolved_deps:
                print(f"    [node {name}] skipped - blocked on unresolved "
                      f"dependency {'/'.join(unresolved_deps)}", flush=True)
                orch_result.node_results[name] = NodeResult(
                    node=node,
                    result=ProverResult(
                        signal=ProofSignal.PROOF_TOO_HARD,
                        analysis=(
                            f"Skipped without attempting a proof: dependency "
                            f"{'/'.join(unresolved_deps)} is not yet proved, so "
                            "this node would compile against an unresolved "
                            "`sorry` stand-in regardless of its own tactic."
                        ),
                    ),
                )
            else:
                wave.append(name)

        if not wave:
            continue

        print(f"  [wave] starting {len(wave)} node(s): {sorted(wave)}", flush=True)

        tasks = [
            _prove_one(
                name=name,
                blueprint=blueprint,
                proof_bodies=proof_bodies,
                compiler=compiler,
                compiler_factory=compiler_factory,
                retrieval=retrieval,
                repo_retrieval=repo_retrieval,
                model=model,
                tracer=tracer,
                node_timeout_s=node_timeout_s,
            )
            for name in wave
        ]
        wave_results: list[NodeResult] = await asyncio.gather(*tasks)

        for nr in wave_results:
            orch_result.node_results[nr.node.name] = nr
            if nr.result.signal == ProofSignal.SOLVED:
                proof_bodies[nr.node.name] = nr.result.proof_body

    return orch_result


async def _prove_one(
    name: str,
    blueprint: Blueprint,
    proof_bodies: dict[str, str],
    compiler: AbstractLeanCompiler | None,
    retrieval: MathlibRetrieval,
    model: str,
    compiler_factory: Callable[[], AbstractLeanCompiler] | None = None,
    repo_retrieval=None,
    tracer=None,
    node_timeout_s: float | None = None,
) -> NodeResult:
    node = blueprint.node_by_name(name)
    assert node is not None

    ancestor_deps = _transitive_deps(node, blueprint)
    # Splice order must be topological - Lean rejects forward references, and
    # a set (from _transitive_deps) has no defined iteration order.
    ordered_deps = [
        n.name for n in blueprint.dependency_order()
        if n.name in ancestor_deps and n.name in proof_bodies
    ]
    parent_proofs = {dep: proof_bodies[dep] for dep in ordered_deps}
    active_compiler = compiler_factory() if compiler_factory else compiler
    assert active_compiler is not None

    # Proven dependencies are otherwise only shown to the model as prompt
    # text - re-declare them as real lemmas so `exact evalFuel_ret_sound h`
    # style references to already-solved siblings actually resolve instead of
    # hitting "unknown identifier".
    parent_lemma_decls = "\n\n".join(
        f"{dep_node.signature()} {body}"
        for dep, body in parent_proofs.items()
        if (dep_node := blueprint.node_by_name(dep)) is not None
    )

    t0 = time.monotonic()
    print(f"    [node {name}] started", flush=True)

    loop = asyncio.get_event_loop()
    future = loop.run_in_executor(
        None,
        prove_node,
        name,
        node.lean_declaration,
        parent_proofs,
        parent_lemma_decls,
        active_compiler,
        retrieval,
        model,
        node.statement,
        node.proof_sketch,
        repo_retrieval,
        tracer,
    )
    try:
        result = await asyncio.wait_for(future, timeout=node_timeout_s)
        dt = time.monotonic() - t0
        print(f"    [node {name}] finished after {dt:.1f}s -> {result.signal.value}", flush=True)
    except asyncio.TimeoutError:
        dt = time.monotonic() - t0
        print(f"    [node {name}] TIMED OUT after {dt:.1f}s (bound={node_timeout_s}s) "
              f"- the underlying call keeps running in its worker thread, "
              f"but this node is marked infra_error so the wave can proceed", flush=True)
        # Note: run_in_executor uses a real OS thread, which asyncio.wait_for
        # cannot forcibly kill - GoedelProver's own api_timeout_s (client-level
        # request timeout) is what actually bounds the underlying OpenAI call.
        # INFRA_ERROR (not PROOF_TOO_HARD): a timeout says nothing about
        # whether the sub-goal is actually hard, so it must not be fed to
        # Phase 3 refinement as if the model had genuinely tried and failed.
        result = ProverResult(
            signal=ProofSignal.INFRA_ERROR,
            analysis=f"Node timed out after {node_timeout_s}s (orchestrator bound).",
        )
    except Exception as exc:
        # An unhandled exception here (network blip, rate limit, etc.) would
        # otherwise propagate through asyncio.gather and take down the whole
        # wave - including sibling nodes that were succeeding - so it's
        # caught per-node the same way a timeout is. INFRA_ERROR, not
        # PROOF_TOO_HARD - see comment above.
        dt = time.monotonic() - t0
        print(f"    [node {name}] ERRORED after {dt:.1f}s -> {exc!r} "
              f"- marked infra_error so the wave can proceed", flush=True)
        result = ProverResult(
            signal=ProofSignal.INFRA_ERROR,
            analysis=f"Node raised {type(exc).__name__}: {exc}",
        )
    return NodeResult(node=node, result=result)

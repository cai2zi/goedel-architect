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
        return not self.failed


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
        loop). A node that exceeds this comes back as PROOF_TOO_HARD instead of
        blocking the rest of the wave indefinitely. None disables the bound.
    """
    dag = _build_dag(blueprint)
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
        wave = [
            name for name in generation
            if name not in proof_bodies
            and (nodes_to_retry is None or name in nodes_to_retry)
        ]
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

    parent_proofs = {dep: proof_bodies[dep] for dep in node.dependencies if dep in proof_bodies}
    active_compiler = compiler_factory() if compiler_factory else compiler
    assert active_compiler is not None

    t0 = time.monotonic()
    print(f"    [node {name}] started", flush=True)

    loop = asyncio.get_event_loop()
    future = loop.run_in_executor(
        None,
        prove_node,
        name,
        node.lean_declaration,
        parent_proofs,
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
              f"but this node is marked proof_too_hard so the wave can proceed", flush=True)
        # Note: run_in_executor uses a real OS thread, which asyncio.wait_for
        # cannot forcibly kill - GoedelProver's own api_timeout_s (client-level
        # request timeout) is what actually bounds the underlying OpenAI call.
        result = ProverResult(
            signal=ProofSignal.PROOF_TOO_HARD,
            analysis=f"Node timed out after {node_timeout_s}s (orchestrator bound).",
        )
    return NodeResult(node=node, result=result)

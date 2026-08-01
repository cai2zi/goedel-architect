"""Phase 2 traversal of the root theorem's active dependency closure."""
from __future__ import annotations

import asyncio
import functools
import time
from concurrent.futures import Executor
from dataclasses import dataclass, field

import networkx as nx

from blueprint import Blueprint, BlueprintNode, render_solved_declaration
from kimina_lean_compiler import KiminaLeanCompiler
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
    active_nodes: set[str] = field(default_factory=set)
    root_name: str = ""

    @property
    def proved(self) -> set[str]:
        return {
            name for name, node_result in self.node_results.items()
            if node_result.result.signal == ProofSignal.SOLVED
        }

    @property
    def failed(self) -> dict[str, NodeResult]:
        return {
            name: node_result for name, node_result in self.node_results.items()
            if node_result.result.signal != ProofSignal.SOLVED
        }

    @property
    def root_solved(self) -> bool:
        result = self.node_results.get(self.root_name)
        return result is not None and result.result.signal == ProofSignal.SOLVED


def active_node_names(blueprint: Blueprint) -> set[str]:
    """All definitions plus the root's transitive proof-node dependencies."""
    root = blueprint.node_by_name(blueprint.target_theorem)
    if root is None or root.kind not in {"lemma", "theorem"}:
        raise ValueError(
            f"Blueprint root `{blueprint.target_theorem}` is missing or is not a proof node"
        )
    node_map = blueprint.nodes_by_name()
    active = {node.name for node in blueprint.nodes if node.kind == "definition"}
    stack = [root.name]
    while stack:
        name = stack.pop()
        if name in active:
            continue
        node = node_map.get(name)
        if node is None:
            continue
        active.add(name)
        stack.extend(dep for dep in node.dependencies if dep in node_map)
    return active


def _transitive_deps(node: BlueprintNode, blueprint: Blueprint) -> set[str]:
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


def _build_dag(blueprint: Blueprint, active: set[str]) -> nx.DiGraph:
    dag = nx.DiGraph()
    for name in active:
        dag.add_node(name)
    for node in blueprint.nodes:
        if node.name not in active:
            continue
        for dep in node.dependencies:
            if dep in active:
                dag.add_edge(dep, node.name)
    return dag


async def prove_dag(
    blueprint: Blueprint,
    compiler: KiminaLeanCompiler,
    retrieval: MathlibRetrieval,
    model: str = "labs-leanstral-1-5",
    proved_cache: dict[str, str] | None = None,
    nodes_to_retry: set[str] | None = None,
    tracer=None,
    node_timeout_s: float | None = 300.0,
    llm_api_timeout_s: float | None = 120.0,
    node_max_prove_turns: int | None = None,
    max_tool_calls_per_turn: int = 3,
    node_executor: Executor | None = None,
    node_semaphore: asyncio.Semaphore | None = None,
) -> OrchestratorResult:
    active = active_node_names(blueprint)
    dag = _build_dag(blueprint, active)
    result = OrchestratorResult(active_nodes=active, root_name=blueprint.target_theorem)
    definitions = [
        node for node in blueprint.nodes
        if node.kind == "definition" and node.name in active
    ]
    definition_names = {node.name for node in definitions}
    proof_bodies = {
        name: body for name, body in (proved_cache or {}).items()
        if name in active
        and (node := blueprint.node_by_name(name)) is not None
        and node.kind != "definition"
    }
    available = definition_names | set(proof_bodies)
    tracer = tracer or NullTracer()

    for node in definitions:
        result.node_results[node.name] = NodeResult(
            node, ProverResult(ProofSignal.SOLVED),
        )
    for name, body in proof_bodies.items():
        node = blueprint.node_by_name(name)
        if node:
            result.node_results[name] = NodeResult(
                node, ProverResult(ProofSignal.SOLVED, body),
            )

    for generation in nx.topological_generations(dag):
        candidates = [
            name for name in generation
            if name not in available
            and name not in definition_names
            and (nodes_to_retry is None or name in nodes_to_retry)
        ]
        wave: list[str] = []
        for name in candidates:
            node = blueprint.node_by_name(name)
            assert node is not None
            unresolved = sorted(
                dep for dep in node.dependencies if dep in active and dep not in available
            )
            if unresolved:
                result.node_results[name] = NodeResult(
                    node,
                    ProverResult(
                        ProofSignal.BLOCKED_BY_DEPENDENCY,
                        lean_errors=[f"Unresolved dependencies: {', '.join(unresolved)}"],
                    ),
                )
            else:
                wave.append(name)
        if not wave:
            continue
        tasks = [
            _prove_one(
                name=name,
                blueprint=blueprint,
                proof_bodies=proof_bodies,
                compiler=compiler,
                retrieval=retrieval,
                model=model,
                tracer=tracer,
                node_timeout_s=node_timeout_s,
                llm_api_timeout_s=llm_api_timeout_s,
                node_max_prove_turns=node_max_prove_turns,
                max_tool_calls_per_turn=max_tool_calls_per_turn,
                node_executor=node_executor,
                node_semaphore=node_semaphore,
            )
            for name in wave
        ]
        wave_results = await asyncio.gather(*tasks)
        for node_result in wave_results:
            name = node_result.node.name
            result.node_results[name] = node_result
            if node_result.result.signal == ProofSignal.SOLVED:
                proof_bodies[name] = node_result.result.proof_body
                available.add(name)
    return result


async def _prove_one(
    *,
    name: str,
    blueprint: Blueprint,
    proof_bodies: dict[str, str],
    compiler: KiminaLeanCompiler,
    retrieval: MathlibRetrieval,
    model: str,
    tracer,
    node_timeout_s: float | None,
    llm_api_timeout_s: float | None,
    node_max_prove_turns: int | None,
    max_tool_calls_per_turn: int,
    node_executor: Executor | None,
    node_semaphore: asyncio.Semaphore | None,
) -> NodeResult:
    node = blueprint.node_by_name(name)
    assert node is not None
    ancestor_deps = _transitive_deps(node, blueprint)
    ordered_parents = [
        candidate for candidate in blueprint.dependency_order()
        if candidate.kind != "definition"
        and candidate.name in ancestor_deps
        and candidate.name in proof_bodies
    ]
    definitions = [
        definition.full_declaration()
        for definition in blueprint.nodes if definition.kind == "definition"
    ]
    parent_signatures = [parent.signature() for parent in ordered_parents]
    compiled_context = definitions + [
        render_solved_declaration(parent, proof_bodies[parent.name])
        for parent in ordered_parents
    ]
    parent_lemma_decls = "\n\n".join(compiled_context)

    async def attempt() -> ProverResult:
        loop = asyncio.get_running_loop()
        if node_semaphore is not None:
            await node_semaphore.acquire()
        future = loop.run_in_executor(
            node_executor,
            functools.partial(
                prove_node,
                node_name=name,
                canonical_stmt=node.lean_declaration,
                parent_signatures=parent_signatures,
                definition_decls=definitions,
                parent_lemma_decls=parent_lemma_decls,
                header=blueprint.phase2_header,
                compiler=compiler,
                retrieval=retrieval,
                model=model,
                node_statement_nl=node.statement,
                node_proof_sketch_nl=node.proof_sketch,
                tracer=tracer,
                api_timeout_s=llm_api_timeout_s,
                max_prove_turns=node_max_prove_turns,
                max_tool_calls_per_turn=max_tool_calls_per_turn,
            ),
        )
        if node_semaphore is not None:
            future.add_done_callback(lambda _: node_semaphore.release())
        if node_timeout_s is None:
            return await future
        return await asyncio.wait_for(asyncio.shield(future), timeout=node_timeout_s)

    started = time.monotonic()
    try:
        proof_result = await attempt()
    except asyncio.TimeoutError:
        proof_result = ProverResult(
            ProofSignal.INFRA_ERROR,
            lean_errors=[f"Node timed out after {node_timeout_s}s"],
        )
    except Exception as exc:
        proof_result = ProverResult(
            ProofSignal.INFRA_ERROR,
            lean_errors=[f"Node raised {type(exc).__name__}: {exc}"],
        )
    print(
        f"    [node {name}] finished after {time.monotonic() - started:.1f}s "
        f"-> {proof_result.signal.value}",
        flush=True,
    )
    return NodeResult(node, proof_result)

from __future__ import annotations

from dataclasses import dataclass

from blueprint import Blueprint, BlueprintNode, render_solved_declaration
from orchestrator import active_node_names
from prover import _build_negation_node_decl


@dataclass(frozen=True)
class NodeProblem:
    node_name: str
    stage: str
    node_decl: str
    parent_lemma_decls: str
    header: str
    complete_lean: str


def transitive_dependencies(blueprint: Blueprint, node: BlueprintNode) -> set[str]:
    found: set[str] = set()
    stack = list(node.dependencies)
    while stack:
        name = stack.pop()
        if name in found:
            continue
        found.add(name)
        dependency = blueprint.node_by_name(name)
        if dependency is not None:
            stack.extend(dependency.dependencies)
    return found


def build_node_problem(
    blueprint: Blueprint,
    node_name: str,
    proved_cache: dict[str, str],
    *,
    stage: str,
) -> NodeProblem:
    node = blueprint.node_by_name(node_name)
    if node is None or node.kind not in {"lemma", "theorem"}:
        raise ValueError(f"not a proof node: {node_name}")
    ancestors = transitive_dependencies(blueprint, node)
    active = active_node_names(blueprint)
    definitions = [
        candidate.full_declaration()
        for candidate in blueprint.nodes
        if candidate.kind == "definition" and candidate.name in active
    ]
    parents = [
        candidate
        for candidate in blueprint.dependency_order()
        if candidate.kind != "definition"
        and candidate.name in ancestors
        and candidate.name in proved_cache
    ]
    missing = sorted(
        name for name in ancestors
        if (candidate := blueprint.node_by_name(name)) is not None
        and candidate.kind != "definition"
        and name not in proved_cache
    )
    if missing:
        raise ValueError(f"unresolved ancestor proofs for {node_name}: {missing}")
    context = definitions + [
        render_solved_declaration(parent, proved_cache[parent.name])
        for parent in parents
    ]
    parent_lemma_decls = "\n\n".join(context)
    node_decl = (
        node.lean_declaration
        if stage == "positive"
        else _build_negation_node_decl(node.lean_declaration, node.name)
    )
    from blueprint_text import extract_current_node_decl
    from blueprint_text import BLUEPRINT_PROOF_RE
    prompt_decl, replacements = BLUEPRINT_PROOF_RE.subn(
        ":= by", extract_current_node_decl(node_decl), count=1,
    )
    if replacements != 1:
        raise ValueError(f"node declaration has no proof placeholder: {node_name}")
    complete = "\n\n".join(
        part for part in [
            blueprint.phase2_header.rstrip(),
            parent_lemma_decls.strip(),
            prompt_decl.strip(),
        ] if part
    ) + "\n"
    return NodeProblem(
        node_name=node_name,
        stage=stage,
        node_decl=node_decl,
        parent_lemma_decls=parent_lemma_decls,
        header=blueprint.phase2_header,
        complete_lean=complete,
    )

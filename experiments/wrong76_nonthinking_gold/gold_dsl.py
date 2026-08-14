from __future__ import annotations

from dataclasses import dataclass, field


NODE_ROLES = {"problem_grounding", "cot_claim", "formal_bridge"}
LABELS = {"definition_valid", "proved", "disproved", "blocked_by_dependency"}


@dataclass(frozen=True)
class Step:
    source_span: str


@dataclass(frozen=True)
class Node:
    name: str
    kind: str
    role: str
    declaration: str
    dependencies: tuple[str, ...] = ()
    source_steps: tuple[int, ...] = ()
    problem_source_span: str = ""
    label: str = ""
    proof: str = ""
    proof_method: str = ""
    statement: str = ""
    proof_sketch: str = ""

    def __post_init__(self) -> None:
        if self.kind not in {"definition", "lemma", "theorem"}:
            raise ValueError(f"invalid kind for {self.name}: {self.kind}")
        if self.role not in NODE_ROLES:
            raise ValueError(f"invalid role for {self.name}: {self.role}")
        if self.label not in LABELS:
            raise ValueError(f"invalid label for {self.name}: {self.label}")
        if self.kind == "definition" and self.label != "definition_valid":
            raise ValueError(f"definition must be definition_valid: {self.name}")
        if self.kind != "definition" and self.label == "definition_valid":
            raise ValueError(f"proof node cannot be definition_valid: {self.name}")
        if self.label in {"proved", "disproved"} and not self.proof.strip():
            raise ValueError(f"missing proof for {self.name}")


@dataclass(frozen=True)
class Case:
    source_id: str
    steps: tuple[Step, ...]
    nodes: tuple[Node, ...]
    fidelity_notes: tuple[str, ...] = ()
    target: str = ""
    header: str = "import Mathlib\nimport Architect\nset_option autoImplicit false\n"

    def __post_init__(self) -> None:
        names = [node.name for node in self.nodes]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate node in {self.source_id}")
        target = self.target or (names[-1] if names else "")
        if not target or target not in names:
            raise ValueError(f"invalid target for {self.source_id}: {target}")
        if next(node for node in self.nodes if node.name == target).kind != "theorem":
            raise ValueError(f"target is not a theorem: {target}")
        seen: set[str] = set()
        for node in self.nodes:
            missing = set(node.dependencies) - seen
            if missing:
                raise ValueError(f"forward/missing dependencies for {node.name}: {sorted(missing)}")
            for index in node.source_steps:
                if index < 1 or index > len(self.steps):
                    raise ValueError(f"invalid source step S{index:02d} for {node.name}")
            seen.add(node.name)

    @property
    def target_name(self) -> str:
        return self.target or self.nodes[-1].name


def definition(
    name: str,
    role: str,
    declaration: str,
    *,
    source_steps: tuple[int, ...] = (),
    problem_source_span: str = "",
    statement: str = "",
) -> Node:
    return Node(
        name=name,
        kind="definition",
        role=role,
        declaration=declaration,
        source_steps=source_steps,
        problem_source_span=problem_source_span,
        label="definition_valid",
        statement=statement,
    )


def claim(
    name: str,
    declaration: str,
    *,
    role: str = "cot_claim",
    dependencies: tuple[str, ...] = (),
    source_steps: tuple[int, ...] = (),
    label: str,
    proof: str = "",
    method: str = "",
    statement: str = "",
    proof_sketch: str = "",
) -> Node:
    return Node(
        name=name,
        kind="lemma",
        role=role,
        declaration=declaration,
        dependencies=dependencies,
        source_steps=source_steps,
        label=label,
        proof=proof,
        proof_method=method,
        statement=statement,
        proof_sketch=proof_sketch,
    )


def target(
    name: str,
    declaration: str,
    *,
    dependencies: tuple[str, ...] = (),
    source_steps: tuple[int, ...] = (),
    label: str,
    proof: str = "",
    method: str = "",
    statement: str = "",
) -> Node:
    return Node(
        name=name,
        kind="theorem",
        role="cot_claim",
        declaration=declaration,
        dependencies=dependencies,
        source_steps=source_steps,
        label=label,
        proof=proof,
        proof_method=method,
        statement=statement,
    )

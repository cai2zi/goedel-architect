"""Deterministic semantic-fidelity checks for COT-grounded blueprints.

The checks in this module are deliberately conservative and local.  They do
not try to decide whether the source COT is mathematically correct.  Instead,
they reject a small set of high-confidence ways in which a Lean blueprint can
cease to represent that COT: losing provenance, dropping steps from the root
dependency closure, replacing claims by vacuous propositions, hard-coding an
answer in a definition, or changing an accepted iter-0 statement during
refinement.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from blueprint import Blueprint, BlueprintNode


_STEP_ID_RE = re.compile(r"^S(?P<number>\d{3})(?:\.[A-Za-z0-9_-]+)?$")
_SIMPLE_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_DECL_HEAD_RE = re.compile(
    r"^\s*(?:noncomputable\s+)?(?:def|abbrev|lemma|theorem)\s+[^\s({:\[]+"
)
@dataclass(frozen=True)
class CotStep:
    step_id: str
    source_text: str = ""
    source_sha256: str = ""
    role: str = "derived_claim"
    depends_on: tuple[str, ...] = ()
    numbers: tuple[str, ...] = ()
    relations: tuple[str, ...] = ()
    requires_formalization: bool = True

    @property
    def base_id(self) -> str:
        return self.step_id.split(".", 1)[0]

    @property
    def ordinal(self) -> int:
        match = _STEP_ID_RE.fullmatch(self.step_id)
        return int(match.group("number")) if match else -1


@dataclass(frozen=True)
class CotManifest:
    steps: tuple[CotStep, ...] = ()

    @property
    def by_id(self) -> dict[str, CotStep]:
        return {step.step_id: step for step in self.steps}

    @property
    def final_step_id(self) -> str:
        conclusions = [
            step.step_id
            for step in self.steps
            if step.role in {"conclusion", "final", "final_claim"}
        ]
        return conclusions[-1] if conclusions else (self.steps[-1].step_id if self.steps else "")


@dataclass(frozen=True)
class SemanticIssue:
    code: str
    message: str
    node_name: str = ""
    step_id: str = ""
    category: str = "static"

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "node_name": self.node_name,
            "step_id": self.step_id,
            "category": self.category,
        }


@dataclass(frozen=True)
class NodeSemanticSnapshot:
    name: str
    kind: str
    source_step_id: str
    semantic_shape: str
    declaration_hash: str
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticSnapshot:
    root_name: str
    root_signature: str
    root_step_id: str
    nodes: tuple[NodeSemanticSnapshot, ...] = ()
    step_shapes: dict[str, tuple[str, ...]] = field(default_factory=dict)


def parse_cot_manifest(value: Any) -> CotManifest:
    """Parse the sole lossless formal-Step manifest."""
    if isinstance(value, CotManifest):
        return value
    from cot_blueprint_refine.formal_steps import decode_formal_step_manifest

    manifest = decode_formal_step_manifest(value)
    raw_steps = list(manifest["steps"])
    return CotManifest(tuple(
        CotStep(
            step_id=str(step["step_id"]),
            source_text=str(step["source_text"]),
            source_sha256=str(step["source_sha256"]),
            role="conclusion" if index == len(raw_steps) else "derived_claim",
            requires_formalization=True,
        )
        for index, step in enumerate(raw_steps, start=1)
    ))


def _base_step_id(value: str) -> str:
    return value.split(".", 1)[0]


def effective_blueprint_dependencies(
    node: BlueprintNode,
    node_names: Iterable[str],
) -> tuple[str, ...]:
    """Combine explicit DAG edges with references in the formal declaration.

    ``sorry_using`` remains the proof-dependency contract.  Identifier edges
    additionally prevent a definition or proposition used in a later node's
    type/body from being falsely reported as disconnected from the root.
    Attribute prose, comments, strings, and the declaration's own name are not
    considered references.
    """
    declaration = node.full_declaration() if node.kind == "definition" else node.signature()
    declaration = _strip_lean_comments(declaration)
    declaration = re.sub(r'"(?:\\.|[^"\\])*"', '""', declaration)
    identifiers = set(re.findall(r"[A-Za-z_][A-Za-z0-9_']*", declaration))
    available = set(node_names)
    inferred = sorted(
        name for name in available
        if name != node.name and name in identifiers
    )
    return tuple(dict.fromkeys([
        *(dependency for dependency in node.dependencies if dependency in available),
        *inferred,
    ]))


def _root_reachable_names(blueprint: Blueprint) -> set[str]:
    node_map = blueprint.nodes_by_name()
    reachable: set[str] = set()
    stack = [blueprint.target_theorem]
    while stack:
        name = stack.pop()
        if name in reachable:
            continue
        node = node_map.get(name)
        if node is None:
            continue
        reachable.add(name)
        stack.extend(effective_blueprint_dependencies(node, node_map))
    return reachable


def _scan_top_level(text: str, operator: str) -> list[int]:
    """Find top-level operators, ignoring comments, strings, and delimiters."""
    positions: list[int] = []
    depths = {"(": 0, "[": 0, "{": 0}
    closing = {")": "(", "]": "[", "}": "{"}
    i = 0
    in_string = False
    line_comment = False
    block_depth = 0
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if line_comment:
            if ch == "\n":
                line_comment = False
            i += 1
            continue
        if block_depth:
            if ch == "/" and nxt == "-":
                block_depth += 1
                i += 2
                continue
            if ch == "-" and nxt == "/":
                block_depth -= 1
                i += 2
                continue
            i += 1
            continue
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue
        if ch == "-" and nxt == "-":
            line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "-":
            block_depth = 1
            i += 2
            continue
        if ch == '"':
            in_string = True
            i += 1
            continue
        if ch in depths:
            depths[ch] += 1
            i += 1
            continue
        if ch in closing:
            key = closing[ch]
            depths[key] = max(0, depths[key] - 1)
            i += 1
            continue
        if not any(depths.values()) and text.startswith(operator, i):
            positions.append(i)
            i += len(operator)
            continue
        i += 1
    return positions


def _top_level_assignment(text: str) -> int | None:
    positions = _scan_top_level(text, ":=")
    return positions[0] if positions else None


def _strip_outer_parens(text: str) -> str:
    value = text.strip()
    while value.startswith("(") and value.endswith(")"):
        depth = 0
        closes_at_end = False
        for index, ch in enumerate(value):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    closes_at_end = index == len(value) - 1
                    break
        if not closes_at_end:
            break
        value = value[1:-1].strip()
    return value


def _normalize_expr(text: str) -> str:
    value = text.strip()
    # Normalize the common coercion-only numeric spelling used by generated
    # roots: ``(257 : ℚ)`` and ``257`` denote the same closed literal here.
    value = re.sub(
        r"\(\s*([-+]?\d+(?:\.\d+)?)\s*:\s*[A-Za-zℕℤℚℝ][A-Za-z0-9_.'ℕℤℚℝ]*\s*\)",
        r"\1",
        value,
    )
    value = _strip_outer_parens(value)
    value = re.sub(r"\s+", "", value)
    return _strip_outer_parens(value)


def _strip_lean_comments(text: str) -> str:
    """Blank Lean comments without changing strings, length, or line layout.

    Lean block comments nest.  Keeping one output character per input
    character lets callers continue to use positions found by the lightweight
    top-level scanner, while replacing comments with whitespace prevents
    comment prose from hiding a vacuous formal expression.  Comment markers
    inside string literals are data and must remain untouched.
    """
    output: list[str] = []
    index = 0
    in_string = False
    line_comment = False
    block_depth = 0
    while index < len(text):
        character = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""

        if line_comment:
            if character == "\n":
                output.append(character)
                line_comment = False
            else:
                output.append(" ")
            index += 1
            continue

        if block_depth:
            if character == "/" and following == "-":
                output.extend((" ", " "))
                block_depth += 1
                index += 2
                continue
            if character == "-" and following == "/":
                output.extend((" ", " "))
                block_depth -= 1
                index += 2
                continue
            output.append("\n" if character == "\n" else " ")
            index += 1
            continue

        if in_string:
            output.append(character)
            if character == "\\" and following:
                output.append(following)
                index += 2
                continue
            if character == '"':
                in_string = False
            index += 1
            continue

        if character == '"':
            output.append(character)
            in_string = True
            index += 1
            continue
        if character == "-" and following == "-":
            output.extend((" ", " "))
            line_comment = True
            index += 2
            continue
        if character == "/" and following == "-":
            output.extend((" ", " "))
            block_depth = 1
            index += 2
            continue

        output.append(character)
        index += 1

    return "".join(output)


def _node_conclusion(node: BlueprintNode) -> str:
    signature = _strip_lean_comments(node.signature())
    head = _DECL_HEAD_RE.match(signature)
    search = signature[head.end():] if head else signature
    positions = _scan_top_level(search, ":")
    return search[positions[0] + 1:].strip() if positions else ""


def _definition_parts(node: BlueprintNode) -> tuple[str, str]:
    declaration = _strip_lean_comments(node.full_declaration())
    assignment = _top_level_assignment(declaration)
    if assignment is None:
        return declaration, ""
    return declaration[:assignment].strip(), declaration[assignment + 2:].strip()


def _is_nullary_definition_prefix(prefix: str) -> bool:
    """Return whether a definition head declares no explicit parameters.

    This intentionally recognizes only the unambiguous generated forms
    ``def name := ...`` and ``def name : Type := ...``.  Parenthesized,
    implicit, instance, or bare binders make the result false; a conservative
    false negative is preferable to rejecting a computed function.
    """
    head = _DECL_HEAD_RE.match(prefix)
    if head is None:
        return False
    remainder = prefix[head.end():].strip()
    return not remainder or remainder.startswith(":")


def _top_level_equality(text: str) -> tuple[str, str] | None:
    positions = _scan_top_level(text, "=")
    valid = []
    for position in positions:
        previous = text[position - 1] if position else ""
        following = text[position + 1] if position + 1 < len(text) else ""
        if previous in {":", "<", ">", "!"} or following in {"=", ">"}:
            continue
        valid.append(position)
    if len(valid) != 1:
        return None
    position = valid[0]
    return text[:position].strip(), text[position + 1:].strip()


def _is_reflexive(text: str) -> bool:
    equality = _top_level_equality(_strip_outer_parens(text))
    if equality is None:
        return False
    left, right = equality
    return bool(left) and _normalize_expr(left) == _normalize_expr(right)


def _contains_true_shell(text: str) -> bool:
    """Return whether a compound proposition contains a literal ``True`` arm.

    Generated blueprints commonly replace a missing conjunct, implication
    conclusion, or quantified body with ``True``.  Looking only for a whole
    declaration equal to ``True`` misses those weakened shells.  This scanner
    follows only top-level logical structure (and a quantifier's body), so an
    identifier or comment containing the letters ``True`` is not enough.
    """
    value = _strip_outer_parens(text)
    if _normalize_expr(value) == "True":
        return True
    for operator in ("↔", "→", "∨", "∧"):
        positions = _scan_top_level(value, operator)
        if not positions:
            continue
        parts: list[str] = []
        start = 0
        for position in positions:
            parts.append(value[start:position])
            start = position + len(operator)
        parts.append(value[start:])
        return any(_contains_true_shell(part) for part in parts)
    if value.startswith(("∀", "∃")):
        payload = value[1:].lstrip()
        commas = _scan_top_level(payload, ",")
        if commas:
            return _contains_true_shell(payload[commas[0] + 1:])
    return False


def _is_unconstrained_exists(text: str) -> bool:
    value = _strip_outer_parens(text)
    payload = value.removeprefix("∃").lstrip() if value.startswith("∃") else ""
    # Catch the fully vacuous shape independently of binder syntax. Lean
    # permits several names in one binder, e.g. `∃ (Torus Sphere : Type),
    # True`, which the single-variable parser below need not interpret.
    comma_positions = _scan_top_level(payload, ",") if payload else []
    if comma_positions:
        existential_body = _strip_outer_parens(payload[comma_positions[0] + 1:])
        if _normalize_expr(existential_body) == "True":
            return True
    variable = ""
    body = ""
    if payload.startswith("("):
        depth = 0
        binder_end = -1
        for index, character in enumerate(payload):
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    binder_end = index
                    break
        if binder_end >= 0:
            binder = payload[1:binder_end].strip()
            remainder = payload[binder_end + 1:].lstrip()
            binder_match = re.fullmatch(
                r"([A-Za-z_][A-Za-z0-9_']*)\s*(?::\s*.+)?",
                binder,
                re.DOTALL,
            )
            if binder_match and remainder.startswith(","):
                variable = binder_match.group(1)
                body = _strip_outer_parens(remainder[1:])
    else:
        match = re.match(
            r"^([A-Za-z_][A-Za-z0-9_']*)\s*(?::\s*[^,]+)?\s*,\s*(.+)$",
            payload,
            re.DOTALL,
        )
        if match:
            variable, body = match.group(1), _strip_outer_parens(match.group(2))
    if not variable or not body:
        return False
    if any(token in body for token in ("∧", "∨", "→", "↔")):
        return False
    equality = _top_level_equality(body)
    if equality is None:
        return False
    left, right = equality
    left_is_var = _normalize_expr(left) == variable
    right_is_var = _normalize_expr(right) == variable
    if left_is_var == right_is_var:
        return False
    closed = right if left_is_var else left
    return re.search(rf"\b{re.escape(variable)}\b", closed) is None


def _simple_claimed_answer(value: str) -> str:
    compact = value.strip().replace(" ", "")
    if _SIMPLE_NUMBER_RE.fullmatch(compact):
        return compact
    fraction = re.fullmatch(r"\\(?:d?frac)\{([-+]?\d+)\}\{(\d+)\}", compact)
    if fraction:
        return f"{fraction.group(1)}/{fraction.group(2)}"
    if re.fullmatch(r"[-+]?\d+/\d+", compact):
        return compact
    return ""


def _contains_answer(expr: str, claimed_answer: str) -> bool:
    answer = _simple_claimed_answer(claimed_answer)
    if not answer:
        return True  # low-confidence formats are delegated to the LLM audit
    normalized = _normalize_expr(expr)
    if "/" in answer:
        numerator, denominator = answer.split("/", 1)
        return re.search(
            rf"(?<!\d){re.escape(numerator)}/{re.escape(denominator)}(?!\d)", normalized,
        ) is not None
    return re.search(rf"(?<!\d){re.escape(answer)}(?!\d)", normalized) is not None


def _semantic_shape(node: BlueprintNode) -> str:
    if node.kind == "definition":
        declaration = node.full_declaration()
        head = _DECL_HEAD_RE.match(declaration)
        value = declaration[head.end():] if head else declaration
    else:
        signature = node.signature()
        head = _DECL_HEAD_RE.match(signature)
        value = signature[head.end():] if head else signature
    return _normalize_expr(value)


def _issue(
    code: str,
    message: str,
    *,
    node: BlueprintNode | None = None,
    step_id: str = "",
    category: str = "static",
) -> SemanticIssue:
    return SemanticIssue(
        code=code,
        message=message,
        node_name=node.name if node is not None else "",
        step_id=step_id or (node.source_step_id if node is not None else ""),
        category=category,
    )


def validate_blueprint_fidelity(
    blueprint: Blueprint,
    manifest: CotManifest | Any = None,
    *,
    claimed_answer: str = "",
    require_step_bindings: bool = False,
) -> list[SemanticIssue]:
    """Return deterministic provenance and high-confidence degeneration issues."""
    contract = parse_cot_manifest(manifest)
    issues: list[SemanticIssue] = []
    valid_ids = contract.by_id
    root = blueprint.node_by_name(blueprint.target_theorem)

    if require_step_bindings and not contract.steps:
        issues.append(SemanticIssue(
            "EMPTY_COT_MANIFEST",
            "Claim bindings were required but the source COT manifest is empty.",
            category="binding",
        ))

    for node in blueprint.nodes:
        title_count = len(re.findall(r"\(title\s*:=", node.lean_declaration))
        if require_step_bindings and title_count != 1:
            issues.append(_issue(
                "MISSING_STEP_MAPPING" if title_count == 0 else "MULTIPLE_STEP_MAPPINGS",
                f"Node must have exactly one source-Step title; found {title_count}.",
                node=node,
                category="binding",
            ))
            continue
        if require_step_bindings and not node.source_step_id:
            issues.append(_issue(
                "MALFORMED_STEP_MAPPING",
                "Node title must be exactly COT_STEP:SNNN.",
                node=node,
                category="binding",
            ))
            continue
        if node.source_step_id:
            base_id = _base_step_id(node.source_step_id)
            if base_id not in valid_ids:
                issues.append(_issue(
                    "UNKNOWN_STEP_MAPPING",
                    f"Node refers to source Step {base_id}, which is not in the manifest.",
                    node=node,
                    category="binding",
                ))

    if root is None:
        issues.append(SemanticIssue(
            "MISSING_ROOT", "The target theorem is not a blueprint node.", category="binding",
        ))
        return issues

    if require_step_bindings and contract.final_step_id:
        if _base_step_id(root.source_step_id) != contract.final_step_id:
            issues.append(_issue(
                "ROOT_NOT_FINAL_STEP",
                f"Root must map to final source Step {contract.final_step_id}.",
                node=root,
                category="binding",
            ))

    reachable = _root_reachable_names(blueprint)
    node_map = blueprint.nodes_by_name()
    reachable_steps = {
        _base_step_id(node.source_step_id)
        for node in blueprint.nodes
        if node.name in reachable and node.source_step_id
    }
    mapped_steps = {
        _base_step_id(node.source_step_id)
        for node in blueprint.nodes
        if node.source_step_id
    }
    if require_step_bindings:
        for step in contract.steps:
            if not step.requires_formalization:
                continue
            if step.step_id not in mapped_steps:
                issues.append(SemanticIssue(
                    "STEP_MAPPING_ABSENT",
                    "No node in the blueprint maps to this source step.",
                    step_id=step.step_id,
                    category="binding",
                ))
            elif step.step_id not in reachable_steps:
                issues.append(SemanticIssue(
                    "STEP_NOT_ROOT_REACHABLE",
                    "A node maps to this source Step, but no mapped node is in the root "
                    "dependency closure.",
                    step_id=step.step_id,
                    category="binding",
                ))
        for node in blueprint.nodes:
            if node.name not in reachable:
                issues.append(_issue(
                    "NODE_NOT_ROOT_REACHABLE",
                    "Every Blueprint node must be in the root's transitive semantic "
                    "dependency closure.",
                    node=node,
                    category="binding",
                ))

    # High-confidence anti-degeneration checks operate on parsed declarations,
    # never on docstrings or raw prompt text.
    for node in blueprint.nodes:
        if node.kind in {"lemma", "theorem"}:
            conclusion = _node_conclusion(node)
            normalized_conclusion = _normalize_expr(conclusion)
            if normalized_conclusion == "True":
                issues.append(_issue(
                    "VACUOUS_TRUE_ROOT" if node.name == blueprint.target_theorem else "VACUOUS_TRUE_STEP",
                    "A source assertion was replaced by the proposition True.",
                    node=node,
                ))
            elif normalized_conclusion == "Prop":
                issues.append(_issue(
                    "VACUOUS_PROP_ROOT" if node.name == blueprint.target_theorem else "VACUOUS_PROP_STEP",
                    "A source assertion was replaced by an unspecified Prop shell.",
                    node=node,
                ))
            elif _contains_true_shell(conclusion):
                issues.append(_issue(
                    "VACUOUS_TRUE_SHELL_ROOT"
                    if node.name == blueprint.target_theorem
                    else "VACUOUS_TRUE_SHELL_STEP",
                    "A source assertion contains a literal True arm in place of a constraint.",
                    node=node,
                ))
            if node.name != blueprint.target_theorem and _is_reflexive(conclusion):
                issues.append(_issue(
                    "REFLEXIVE_STEP",
                    "A source assertion was replaced by a reflexive equality.",
                    node=node,
                ))
            if node.name != blueprint.target_theorem and _is_unconstrained_exists(conclusion):
                issues.append(_issue(
                    "UNCONSTRAINED_EXISTS_STEP",
                    "A source assertion only chooses an unconstrained closed witness.",
                    node=node,
                ))
        else:
            prefix, body = _definition_parts(node)
            normalized_body = _normalize_expr(body)
            if re.search(r":\s*Prop\s*$", prefix) and normalized_body == "True":
                issues.append(_issue(
                    "VACUOUS_PROP_DEFINITION",
                    "A Prop definition is hard-coded to True.",
                    node=node,
                ))
            elif re.search(r":\s*Prop\s*$", prefix) and _contains_true_shell(body):
                issues.append(_issue(
                    "VACUOUS_TRUE_SHELL_DEFINITION",
                    "A Prop definition contains a literal True arm in place of a constraint.",
                    node=node,
                ))
            if re.search(r":\s*Prop\s*$", prefix) and _is_unconstrained_exists(body):
                issues.append(_issue(
                    "UNCONSTRAINED_EXISTS_DEFINITION",
                    "A Prop definition only introduces unconstrained witnesses.",
                    node=node,
                ))
            if re.search(r":\s*Bool\s*$", prefix) and normalized_body in {"true", "false"}:
                issues.append(_issue(
                    "VACUOUS_BOOL_DEFINITION",
                    "A mathematical assertion is represented by a constant Bool definition.",
                    node=node,
                ))
            # A Step may legitimately introduce an object whose value happens
            # to equal the final answer. Answer-bearing definitions are routed
            # through the full semantic audit instead of rejected locally.

    root_conclusion = _node_conclusion(root)
    if _is_reflexive(root_conclusion):
        issues.append(_issue(
            "REFLEXIVE_ROOT",
            "The root is a reflexive equality and no longer states the source problem.",
            node=root,
        ))
    if _is_unconstrained_exists(root_conclusion):
        issues.append(_issue(
            "UNCONSTRAINED_EXISTS_ROOT",
            "The root only chooses a closed witness and imposes no source-problem constraint.",
            node=root,
        ))
    if claimed_answer and not _contains_answer(root_conclusion, claimed_answer):
        issues.append(_issue(
            "ROOT_MISSING_CLAIMED_ANSWER",
            "The root does not retain the simple claimed final answer.",
            node=root,
            category="grounding",
        ))
    proof_ancestors = {
        name for name in reachable - {root.name}
        if (candidate := node_map.get(name)) is not None
        and candidate.kind in {"lemma", "theorem"}
    }
    substantive_steps = [step for step in contract.steps if step.requires_formalization]
    if substantive_steps and len(substantive_steps) > 1 and not proof_ancestors:
        issues.append(_issue(
            "ROOT_NOT_GROUNDED",
            "The root has no proof-step ancestor from the multi-step source COT.",
            node=root,
            category="grounding",
        ))
    return issues


def semantic_audit_risk_reasons(
    blueprint: Blueprint,
    manifest: CotManifest | Any,
    *,
    claimed_answer: str = "",
) -> list[str]:
    """Return cheap reasons for routing a sample to the optional LLM audit.

    Absence of a reason is not a proof of faithfulness; it only means the local
    structure and the minimal numeric/relation IR found no ambiguity worth an
    additional 397B request.
    """
    contract = parse_cot_manifest(manifest)
    steps_by_id = {step.step_id: step for step in contract.steps}
    nodes_by_step: dict[str, list[BlueprintNode]] = {}
    for node in blueprint.nodes:
        base_id = _base_step_id(node.source_step_id)
        if base_id:
            nodes_by_step.setdefault(base_id, []).append(node)
    reasons: set[str] = set()
    for node in blueprint.nodes:
        if node.kind != "definition":
            continue
        base_id = _base_step_id(node.source_step_id)
        source_step = steps_by_id.get(base_id)
        if source_step and source_step.role not in {
            "setup", "given", "object_definition", "computation",
        }:
            reasons.add(
                "DERIVED_STEP_AS_DEFINITION:"
                f"step={base_id}:node={node.name}:role={source_step.role}"
            )
    relation_tokens = {
        "eq": "=", "ne": "≠", "le": "≤", "ge": "≥",
        "lt": "<", "gt": ">", "implication": "→",
    }
    for step in contract.steps:
        if not step.requires_formalization:
            continue
        nodes = nodes_by_step.get(step.step_id, [])
        if len(nodes) > 1:
            reasons.add(f"MULTI_NODE_STEP:{step.step_id}")
        lean_text = "\n".join(
            node.full_declaration() if node.kind == "definition" else node.signature()
            for node in nodes
        )
        normalized = _normalize_expr(lean_text)
        for number in step.numbers:
            literal = number.replace(" ", "")
            if literal and re.search(rf"(?<!\d){re.escape(literal)}(?!\d)", normalized) is None:
                reasons.add(f"NUMBER_NOT_VISIBLE:{step.step_id}:{number}")
        for relation in step.relations:
            token = relation_tokens.get(relation)
            if token and token not in lean_text:
                # Lean often spells <=/>= in ASCII.
                if not (token == "≤" and "<=" in lean_text) and not (
                    token == "≥" and ">=" in lean_text
                ):
                    reasons.add(f"RELATION_NOT_VISIBLE:{step.step_id}:{relation}")

    # Textual COT order is useful provenance, but it is not necessarily the
    # logical dependency order.  For example, an opening overview can use an
    # object whose detailed setup appears in the next numbered paragraph.
    # Such edges are therefore audit-routing signals rather than deterministic
    # fidelity failures.
    ordinal_by_id = {step.step_id: step.ordinal for step in contract.steps}
    node_map = blueprint.nodes_by_name()
    for node in blueprint.nodes:
        source_step = _base_step_id(node.source_step_id)
        source_ordinal = ordinal_by_id.get(source_step, -1)
        if source_ordinal < 0:
            continue
        for dependency_name in node.dependencies:
            dependency = node_map.get(dependency_name)
            if dependency is None:
                continue
            dependency_step = _base_step_id(dependency.source_step_id)
            dependency_ordinal = ordinal_by_id.get(dependency_step, -1)
            if dependency_ordinal > source_ordinal:
                reasons.add(
                    "FUTURE_STEP_DEPENDENCY:"
                    f"node={node.name}:source={source_step}:"
                    f"dependency={dependency.name}:dependency_source={dependency_step}"
                )
    answer = _simple_claimed_answer(claimed_answer)
    if answer:
        for node in blueprint.nodes:
            if node.kind != "definition":
                continue
            _prefix, body = _definition_parts(node)
            if _contains_answer(body, answer):
                reasons.add(f"ANSWER_IN_DEFINITION:{node.name}")
    root = blueprint.node_by_name(blueprint.target_theorem)
    if root is not None and "∃" in _node_conclusion(root):
        reasons.add("EXISTENTIAL_ROOT")
    return sorted(reasons)


def snapshot_blueprint_semantics(
    blueprint: Blueprint,
    manifest: CotManifest | Any = None,
) -> SemanticSnapshot:
    # Parsing validates that a persisted manifest has not silently changed.
    parse_cot_manifest(manifest)
    nodes: list[NodeSemanticSnapshot] = []
    grouped: dict[str, list[str]] = {}
    for node in blueprint.nodes:
        shape = _semantic_shape(node)
        step_id = _base_step_id(node.source_step_id)
        nodes.append(NodeSemanticSnapshot(
            name=node.name,
            kind=node.kind,
            source_step_id=node.source_step_id,
            semantic_shape=shape,
            declaration_hash=hashlib.sha256(
                node.full_declaration().encode("utf-8")
            ).hexdigest(),
            dependencies=tuple(node.dependencies),
        ))
        if step_id:
            grouped.setdefault(step_id, []).append(f"{node.kind}:{shape}")
    root = blueprint.node_by_name(blueprint.target_theorem)
    return SemanticSnapshot(
        root_name=blueprint.target_theorem,
        root_signature=_normalize_expr(root.signature()) if root is not None else "",
        root_step_id=root.source_step_id if root is not None else "",
        nodes=tuple(nodes),
        step_shapes={key: tuple(sorted(values)) for key, values in sorted(grouped.items())},
    )


def semantic_snapshot_from_dict(value: Any) -> SemanticSnapshot:
    """Restore a JSON/dataclasses.asdict snapshot from a checkpoint."""
    if isinstance(value, SemanticSnapshot):
        return value
    if not isinstance(value, dict):
        raise ValueError("semantic snapshot must be an object")
    raw_nodes = value.get("nodes") or []
    nodes = tuple(
        NodeSemanticSnapshot(
            name=str(item.get("name") or ""),
            kind=str(item.get("kind") or ""),
            source_step_id=str(item.get("source_step_id") or ""),
            semantic_shape=str(item.get("semantic_shape") or ""),
            declaration_hash=str(item.get("declaration_hash") or ""),
            dependencies=tuple(str(dep) for dep in (item.get("dependencies") or [])),
        )
        for item in raw_nodes
        if isinstance(item, dict)
    )
    raw_shapes = value.get("step_shapes") or {}
    if not isinstance(raw_shapes, dict):
        raise ValueError("semantic snapshot step_shapes must be an object")
    return SemanticSnapshot(
        root_name=str(value.get("root_name") or ""),
        root_signature=str(value.get("root_signature") or ""),
        root_step_id=str(value.get("root_step_id") or ""),
        nodes=nodes,
        step_shapes={
            str(key): tuple(str(shape) for shape in (shapes or []))
            for key, shapes in raw_shapes.items()
        },
    )


def check_semantic_freeze(
    baseline: SemanticSnapshot,
    revised: Blueprint,
    manifest: CotManifest | Any = None,
) -> list[SemanticIssue]:
    """Reject iter-N candidates that change the accepted iter-0 semantics."""
    parse_cot_manifest(manifest)
    issues: list[SemanticIssue] = []
    revised_snapshot = snapshot_blueprint_semantics(revised, manifest)
    if revised.target_theorem != baseline.root_name:
        issues.append(SemanticIssue(
            "ROOT_NAME_DRIFT",
            f"Root changed from {baseline.root_name} to {revised.target_theorem}.",
            node_name=revised.target_theorem,
            category="drift",
        ))
    if revised_snapshot.root_signature != baseline.root_signature:
        issues.append(SemanticIssue(
            "ROOT_SIGNATURE_DRIFT",
            "The refinement changed the root binders, assumptions, or conclusion.",
            node_name=revised.target_theorem,
            step_id=revised_snapshot.root_step_id,
            category="drift",
        ))
    if _base_step_id(revised_snapshot.root_step_id) != _base_step_id(baseline.root_step_id):
        issues.append(SemanticIssue(
            "ROOT_STEP_BINDING_DRIFT",
            "The refinement moved the root to a different source step.",
            node_name=revised.target_theorem,
            step_id=revised_snapshot.root_step_id,
            category="drift",
        ))

    revised_shapes = {
        key: list(values) for key, values in revised_snapshot.step_shapes.items()
    }
    for step_id, old_shapes in baseline.step_shapes.items():
        available = revised_shapes.get(step_id, [])
        for shape in old_shapes:
            if shape in available:
                available.remove(shape)
            else:
                issues.append(SemanticIssue(
                    "STEP_SEMANTIC_DRIFT",
                    "An iter-0 declaration was removed or its mathematical type/body changed.",
                    step_id=step_id,
                    category="drift",
                ))
                break
    return issues


def format_semantic_issues(issues: Iterable[SemanticIssue], limit: int = 20) -> str:
    values = list(issues)
    shown = values[:limit]
    lines: list[str] = []
    for issue in shown:
        location = "/".join(
            value for value in (issue.step_id, issue.node_name) if value
        )
        lines.append(
            f"- {issue.code}{f' [{location}]' if location else ''}: {issue.message}"
        )
    if len(values) > limit:
        lines.append(f"... and {len(values) - limit} more")
    return "\n".join(lines)

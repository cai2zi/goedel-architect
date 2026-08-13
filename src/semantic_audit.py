"""Strict semantic audit for complete Whole-COT Blueprint generation.

Separate mode keeps the blind decompiler and comparator as independent
requests. Joint mode asks for the literal translation first and its audit
second in one response. The runner, rather than the model, computes PASS.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from llm_client import chat_completion_with_retry
from tracer import TraceEvent


WHOLE_COT_PROMPT_VERSION = "whole-cot-comparator-v1"
JOINT_WHOLE_COT_PROMPT_VERSION = "whole-cot-joint-audit-v2"
SEMANTIC_EFFECTS = {"objectDefinition", "proposition", "vacuous"}
JOINT_SEMANTIC_EFFECT_ALIASES = {
    "assertsproperty": "proposition",
    "assumption": "proposition",
    "claim": "proposition",
    "claimassertion": "proposition",
    "claimstatement": "proposition",
    "deduction": "proposition",
    "lemma": "proposition",
    "lemmastatement": "proposition",
    "logicalassertion": "proposition",
    "logicalclaim": "proposition",
    "logicalstatement": "proposition",
    "mathematicalassertion": "proposition",
    "mathematicalstatement": "proposition",
    "propertyassertion": "proposition",
    "propertyclaim": "proposition",
    "propertystatement": "proposition",
    "propositionstatement": "proposition",
    "statementassertion": "proposition",
    "statesproperty": "proposition",
    "theoremassertion": "proposition",
    "theoremclaim": "proposition",
    "theoremstatement": "proposition",
    "definition": "objectDefinition",
    "definesconstant": "objectDefinition",
    "propertydefinition": "objectDefinition",
}


DECOMPILER_SYSTEM_PROMPT = r"""You are a literal Lean semantic decompiler.
You receive only sanitized Lean declarations and their graph metadata. You do
not receive the source problem, chain-of-thought, claimed answer, Blueprint
comments, or proof bodies. Translate exactly what each formal declaration
means. Never infer intended mathematics from identifier names.

`semantic_effect` must be:
- `objectDefinition` for a non-vacuous object/function/set/relation definition;
- `proposition` for a non-vacuous proposition;
- `vacuous` only for True shells, literal reflexive/answer aliases such as
  `let x := 64; x = 64`, unconstrained witnesses, or declarations that impose
  no substantive mathematical constraint. A theorem which repeats the same
  non-vacuous proposition as an earlier node is still `proposition`: graph
  redundancy is not semantic vacuity, and a final COT answer restatement may
  legitimately repeat its supporting lemma's proposition.

Return exactly one JSON object with this shape and no Markdown:
{"nodes":[{"node_name":"n","kind":"definition","translation":"...",
"semantic_effect":"objectDefinition","introduced_objects":[],
"referenced_objects":[]}]}

Include every supplied node exactly once and in supplied order. Keep every
translation under 80 words. Object arrays contain only identifiers literally
introduced or referenced by the sanitized formal declaration; graph
dependencies are not formal references. For every implication or quantified
theorem, explicitly distinguish assumptions from the conclusion using the
words "Assuming ..., concludes ...". Never promote a hypothesis into the
conclusion, and never infer a geometric/counting meaning from the node name.
"""


WHOLE_COT_COMPARATOR_SYSTEM_PROMPT = r"""You are a strict semantic-translation
comparator, not a truth judge. Compare the complete original chain-of-thought
with frozen literal translations of sanitized Lean declarations. A faithful
formalization of a mathematically wrong COT must pass. Reject only omissions,
weakenings, added claims, object replacement, unbound objects, wrong relation
or direction, answer hard-coding, dependency breaks, and an unrelated root.
Names and comments carry no semantic credit.

Audit every relation and target object mechanically. In particular, an
equality describing a boundary or surface does not formalize an enclosed
region, solid, volume, or interior, which normally requires a membership or
inequality constraint. The root must preserve the target object requested by
the original problem, not merely repeat a final equation from the COT. Root
`reasons` list defects only and must be empty when both root booleans are true.

Return one JSON object with exactly `cot`, `root`, `unreachable_nodes`, and
`dependency_issues`. `cot` has exactly `combined_formal_translation`,
`missing_clauses`, `weakened_clauses`, `unbound_objects`, `wrong_relations`,
and `added_clauses`. Each clause issue has exactly `clause`, `node_names`, and
`reason`. `root` has exactly `translation`, `target_object_preserved`,
`answer_grounded`, and `reasons`. Copy the supplied unreachable-node inventory
exactly and in order; each item has `node_name`, `justified_side_branch`, and
`reason`. Each dependency issue has exactly `node_name` and `reason`. Use only
supplied node names. Return JSON only, no Markdown or extra keys. Be terse.
"""


JOINT_WHOLE_COT_SYSTEM_PROMPT = r"""You perform a two-part semantic audit in
one response. First literally translate every sanitized Lean declaration.
Second compare the complete original chain-of-thought with those translations.

For the `formal_decompiler` part, translate only the supplied Lean. Names,
comments, the source problem, and the COT give no semantic credit. Use exactly
the required node order and the same node schema as a literal formal
decompiler. Complete this part before deciding the comparator part.
`semantic_effect` is a closed enum: every node must use exactly one of
`objectDefinition`, `proposition`, or `vacuous` with identical spelling and
capitalization. Never emit alternatives such as `theoremStatement`,
`propertyAssertion`, `lemmaStatement`, `propertyDefinition`, or `definition`.

For the `whole_cot_comparator` part, treat your completed node translations as
frozen evidence. A mathematically wrong COT must pass when formalized exactly.
Reject omissions, weakenings, added claims, object replacement, unbound
objects, wrong relation/direction, answer hard-coding, dependency breaks, and
an unrelated root. A boundary equality does not represent an enclosed volume
or interior. Use only supplied node names. The runner recomputes PASS.

Return one JSON object and no Markdown. Its top-level keys must occur exactly
in this order: `formal_decompiler`, then `whole_cot_comparator`. The first value
has exactly `nodes`. The second has exactly `cot`, `root`,
`unreachable_nodes`, and `dependency_issues`, using the supplied schemas and
inventories. Keep each node translation under 40 words, the combined formal
translation under 60 words, and each clause or reason under 20 words. Do not
repeat the Lean declarations. Completing valid JSON is more important than
explanation.
"""


@dataclass(frozen=True)
class FormalNodeView:
    node_name: str
    kind: str
    dependencies: tuple[str, ...]
    declaration: str
    is_root: bool
    in_root_closure: bool


@dataclass(frozen=True)
class FormalView:
    nodes: tuple[FormalNodeView, ...]
    root_name: str
    root_closure: tuple[str, ...]
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [asdict(node) for node in self.nodes],
            "root_name": self.root_name,
            "root_closure": list(self.root_closure),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class FormalNodeTranslation:
    node_name: str
    kind: str
    translation: str
    semantic_effect: str
    introduced_objects: tuple[str, ...]
    referenced_objects: tuple[str, ...]


@dataclass(frozen=True)
class FormalDecompilerResult:
    nodes: tuple[FormalNodeTranslation, ...]
    raw_content: str
    reasoning_content: str
    finish_reason: str | None
    request_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    attempts: tuple[dict[str, Any], ...] = ()

    @property
    def vacuous_nodes(self) -> tuple[str, ...]:
        return tuple(node.node_name for node in self.nodes if node.semantic_effect == "vacuous")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WholeCotComparatorResult:
    cot: dict[str, Any]
    root: dict[str, Any]
    unreachable_nodes: tuple[dict[str, Any], ...]
    dependency_issues: tuple[dict[str, Any], ...]
    passed: bool
    raw_content: str
    reasoning_content: str
    finish_reason: str | None
    request_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    attempts: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JointWholeCotAuditResult:
    decompiler: FormalDecompilerResult
    comparator: WholeCotComparatorResult
    raw_content: str
    reasoning_content: str
    finish_reason: str | None
    request_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    attempts: tuple[dict[str, Any], ...] = ()
    semantic_effect_normalizations: tuple[dict[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SemanticAuditFormatError(ValueError):
    def __init__(
        self,
        reason: str,
        *,
        raw_content: str = "",
        attempts: Sequence[dict[str, Any]] = (),
    ) -> None:
        super().__init__(f"semantic audit format error: {reason}")
        self.reason = reason
        self.raw_content = raw_content
        self.attempts = tuple(attempts)


def _strip_lean_comments(text: str) -> str:
    """Strip nested Lean comments without corrupting strings."""
    output: list[str] = []
    index = 0
    block_depth = 0
    line_comment = False
    in_string = False
    while index < len(text):
        ch = text[index]
        nxt = text[index + 1] if index + 1 < len(text) else ""
        if line_comment:
            if ch == "\n":
                line_comment = False
                output.append(ch)
            index += 1
            continue
        if block_depth:
            if ch == "/" and nxt == "-":
                block_depth += 1
                index += 2
            elif ch == "-" and nxt == "/":
                block_depth -= 1
                index += 2
            else:
                if ch == "\n":
                    output.append(ch)
                index += 1
            continue
        if in_string:
            output.append(ch)
            if ch == "\\" and index + 1 < len(text):
                output.append(text[index + 1])
                index += 2
                continue
            if ch == '"':
                in_string = False
            index += 1
            continue
        if ch == '"':
            in_string = True
            output.append(ch)
            index += 1
        elif ch == "-" and nxt == "-":
            line_comment = True
            index += 2
        elif ch == "/" and nxt == "-":
            block_depth = 1
            index += 2
        else:
            output.append(ch)
            index += 1
    return "\n".join(line.rstrip() for line in "".join(output).splitlines() if line.strip()).strip()


def build_formal_view(blueprint: Any) -> FormalView:
    node_names = {node.name for node in blueprint.nodes}
    declarations: dict[str, str] = {}
    effective: dict[str, tuple[str, ...]] = {}
    for node in blueprint.nodes:
        raw = node.full_declaration() if node.kind == "definition" else node.signature()
        declaration = _strip_lean_comments(raw)
        declarations[node.name] = declaration
        effective[node.name] = tuple(
            dep for dep in node.dependencies if dep in node_names
        )
    closure: set[str] = set()
    stack = [blueprint.target_theorem]
    while stack:
        name = stack.pop()
        if name in closure or name not in node_names:
            continue
        closure.add(name)
        stack.extend(effective.get(name, ()))
    views = tuple(FormalNodeView(
        node.name,
        node.kind,
        effective[node.name],
        declarations[node.name],
        node.name == blueprint.target_theorem,
        node.name in closure,
    ) for node in blueprint.nodes)
    payload = json.dumps(
        {"root": blueprint.target_theorem, "nodes": [asdict(node) for node in views]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return FormalView(
        views,
        blueprint.target_theorem,
        tuple(node.node_name for node in views if node.in_root_closure),
        hashlib.sha256(payload.encode()).hexdigest(),
    )


def formal_decompiler_messages(view: FormalView) -> list[dict[str, str]]:
    inventory = []
    for node in view.nodes:
        item = {
            "node_name": node.node_name,
            "kind": node.kind,
            "dependencies": list(node.dependencies),
            "is_root": node.is_root,
            "in_root_closure": node.in_root_closure,
            "sanitized_formal_lean": node.declaration,
        }
        inventory.append(item)
    return [
        {"role": "system", "content": DECOMPILER_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Formal View SHA-256: {view.sha256}\n"
            + json.dumps({"root": view.root_name, "nodes": inventory}, ensure_ascii=False)
        )},
    ]


def _json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.I)
    if fenced:
        text = fenced.group(1).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SemanticAuditFormatError(f"invalid JSON: {exc}", raw_content=content) from exc
    if not isinstance(value, dict):
        raise SemanticAuditFormatError("top-level value must be an object", raw_content=content)
    return value


def _string(value: Any, label: str, raw: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise SemanticAuditFormatError(f"{label} must be a string", raw_content=raw)
    return value.strip()


def _strings(value: Any, label: str, raw: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise SemanticAuditFormatError(f"{label} must be an array of strings", raw_content=raw)
    return tuple(item.strip() for item in value)


def _obvious_true_shell(declaration: str) -> bool:
    compact = re.sub(r"\s+", " ", declaration).strip()
    return bool(
        re.search(r":\s*True\s*$", compact)
        or re.search(r":\s*Prop\s*:=\s*True\s*$", compact)
    )


def _semantic_effect(
    value: Any,
    *,
    declaration: str,
    raw: str,
    allow_joint_aliases: bool,
) -> str:
    effect = _string(value, "semantic_effect", raw)
    if effect in SEMANTIC_EFFECTS:
        return effect
    if not allow_joint_aliases:
        raise SemanticAuditFormatError(
            f"invalid semantic_effect {effect}", raw_content=raw,
        )
    normalized = re.sub(r"[^a-z0-9]", "", effect.lower())
    canonical = JOINT_SEMANTIC_EFFECT_ALIASES.get(normalized)
    if canonical is None:
        raise SemanticAuditFormatError(
            f"invalid semantic_effect {effect}", raw_content=raw,
        )
    if _obvious_true_shell(declaration):
        return "vacuous"
    return canonical


def parse_formal_decompiler(
    content: str,
    *,
    view: FormalView,
    allow_joint_aliases: bool = False,
) -> tuple[FormalNodeTranslation, ...]:
    value = _json_object(content)
    if set(value) != {"nodes"} or not isinstance(value["nodes"], list):
        raise SemanticAuditFormatError("top-level key must be exactly nodes", raw_content=content)
    parsed: list[FormalNodeTranslation] = []
    for index, item in enumerate(value["nodes"]):
        expected_keys = {
            "node_name", "kind", "translation", "semantic_effect",
            "introduced_objects", "referenced_objects",
        }
        if not isinstance(item, dict) or set(item) != expected_keys:
            raise SemanticAuditFormatError(f"nodes[{index}] has invalid keys", raw_content=content)
        node_name = _string(item["node_name"], "node_name", content)
        formal_node = next(
            (node for node in view.nodes if node.node_name == node_name), None,
        )
        if formal_node is None:
            raise SemanticAuditFormatError(
                f"node inventory contains unknown node {node_name}", raw_content=content,
            )
        effect = _semantic_effect(
            item["semantic_effect"], declaration=formal_node.declaration,
            raw=content, allow_joint_aliases=allow_joint_aliases,
        )
        parsed.append(FormalNodeTranslation(
            node_name,
            _string(item["kind"], "kind", content),
            _string(item["translation"], "translation", content),
            effect,
            _strings(item["introduced_objects"], "introduced_objects", content),
            _strings(item["referenced_objects"], "referenced_objects", content),
        ))
    if [item.node_name for item in parsed] != [node.node_name for node in view.nodes]:
        raise SemanticAuditFormatError("node inventory is incomplete, duplicated, or reordered", raw_content=content)
    expected_kinds = {node.node_name: node.kind for node in view.nodes}
    if any(item.kind != expected_kinds[item.node_name] for item in parsed):
        raise SemanticAuditFormatError("node kind does not match Formal View", raw_content=content)
    return tuple(parsed)


_ISSUE_KEYS = {"clause", "node_names", "reason"}


def _parse_clause_issues(
    values: Any,
    *,
    label: str,
    known_nodes: set[str],
    raw: str,
) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        raise SemanticAuditFormatError(f"{label} must be an array", raw_content=raw)
    parsed = []
    for index, item in enumerate(values):
        if not isinstance(item, dict) or set(item) != _ISSUE_KEYS:
            raise SemanticAuditFormatError(
                f"{label}[{index}] has invalid keys", raw_content=raw,
            )
        nodes = _strings(item["node_names"], f"{label}.node_names", raw)
        if any(node not in known_nodes for node in nodes):
            raise SemanticAuditFormatError(
                f"{label} references unknown node", raw_content=raw,
            )
        parsed.append({
            "clause": _string(item["clause"], f"{label}.clause", raw),
            "node_names": list(nodes),
            "reason": _string(item["reason"], f"{label}.reason", raw),
        })
    return parsed


_WHOLE_COT_KEYS = {
    "combined_formal_translation", "missing_clauses", "weakened_clauses",
    "unbound_objects", "wrong_relations", "added_clauses",
}


def whole_cot_comparator_messages(
    informal_statement: str,
    informal_proof: str,
    claimed_answer: str,
    view: FormalView,
    decompiler: FormalDecompilerResult,
) -> list[dict[str, str]]:
    unreachable = [node.node_name for node in view.nodes if not node.in_root_closure]
    payload = {
        "problem": informal_statement,
        "claimed_answer": claimed_answer,
        "complete_original_cot": informal_proof,
        "formal_view": view.to_dict(),
        "frozen_node_translations": [asdict(node) for node in decompiler.nodes],
        "required_unreachable_node_names": unreachable,
        "required_output_shape": {
            "cot": {
                "combined_formal_translation": "...",
                "missing_clauses": [], "weakened_clauses": [],
                "unbound_objects": [], "wrong_relations": [], "added_clauses": [],
            },
            "root": {
                "translation": "...", "target_object_preserved": True,
                "answer_grounded": True, "reasons": [],
            },
            "unreachable_nodes": [{
                "node_name": "n", "justified_side_branch": False, "reason": "...",
            }],
            "dependency_issues": [{"node_name": "n", "reason": "..."}],
        },
    }
    return [
        {"role": "system", "content": WHOLE_COT_COMPARATOR_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def parse_whole_cot_comparator(
    content: str,
    *,
    view: FormalView,
    decompiler: FormalDecompilerResult,
) -> tuple[dict[str, Any], dict[str, Any], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], bool]:
    value = _json_object(content)
    if set(value) != {"cot", "root", "unreachable_nodes", "dependency_issues"}:
        raise SemanticAuditFormatError("invalid whole-COT comparator top-level keys", raw_content=content)
    known_nodes = {node.node_name for node in view.nodes}
    cot = value["cot"]
    if not isinstance(cot, dict) or set(cot) != _WHOLE_COT_KEYS:
        raise SemanticAuditFormatError("cot has invalid keys", raw_content=content)
    parsed_cot: dict[str, Any] = {
        "combined_formal_translation": _string(
            cot["combined_formal_translation"], "cot.combined_formal_translation", content,
        ),
    }
    for key in (
        "missing_clauses", "weakened_clauses", "unbound_objects",
        "wrong_relations", "added_clauses",
    ):
        parsed_cot[key] = _parse_clause_issues(
            cot[key], label=f"cot.{key}", known_nodes=known_nodes, raw=content,
        )

    root = value["root"]
    if not isinstance(root, dict) or set(root) != {
        "translation", "target_object_preserved", "answer_grounded", "reasons",
    }:
        raise SemanticAuditFormatError("root has invalid keys", raw_content=content)
    if not isinstance(root["target_object_preserved"], bool) or not isinstance(root["answer_grounded"], bool):
        raise SemanticAuditFormatError("root verdicts must be booleans", raw_content=content)
    parsed_root = {
        "translation": _string(root["translation"], "root.translation", content),
        "target_object_preserved": root["target_object_preserved"],
        "answer_grounded": root["answer_grounded"],
        "reasons": list(_strings(root["reasons"], "root.reasons", content)),
    }

    expected_unreachable = [node.node_name for node in view.nodes if not node.in_root_closure]
    raw_unreachable = value["unreachable_nodes"]
    if not isinstance(raw_unreachable, list):
        raise SemanticAuditFormatError("unreachable_nodes must be an array", raw_content=content)
    unreachable = []
    for item in raw_unreachable:
        if not isinstance(item, dict) or set(item) != {
            "node_name", "justified_side_branch", "reason",
        }:
            raise SemanticAuditFormatError("unreachable node has invalid keys", raw_content=content)
        if not isinstance(item["justified_side_branch"], bool):
            raise SemanticAuditFormatError("justified_side_branch must be boolean", raw_content=content)
        unreachable.append({
            "node_name": _string(item["node_name"], "unreachable node_name", content),
            "justified_side_branch": item["justified_side_branch"],
            "reason": _string(item["reason"], "unreachable reason", content),
        })
    if [item["node_name"] for item in unreachable] != expected_unreachable:
        raise SemanticAuditFormatError(
            "unreachable node inventory is incomplete or reordered", raw_content=content,
        )

    raw_dependencies = value["dependency_issues"]
    if not isinstance(raw_dependencies, list):
        raise SemanticAuditFormatError("dependency_issues must be an array", raw_content=content)
    dependencies = []
    for item in raw_dependencies:
        if not isinstance(item, dict) or set(item) != {"node_name", "reason"}:
            raise SemanticAuditFormatError("dependency issue has invalid keys", raw_content=content)
        node_name = _string(item["node_name"], "dependency node", content)
        if node_name not in known_nodes:
            raise SemanticAuditFormatError("dependency issue has unknown node", raw_content=content)
        dependencies.append({
            "node_name": node_name,
            "reason": _string(item["reason"], "dependency reason", content),
        })

    passed = (
        not any(parsed_cot[key] for key in (
            "missing_clauses", "weakened_clauses", "unbound_objects",
            "wrong_relations", "added_clauses",
        ))
        and parsed_root["target_object_preserved"]
        and parsed_root["answer_grounded"]
        and not parsed_root["reasons"]
        and all(item["justified_side_branch"] for item in unreachable)
        and not dependencies
        and not decompiler.vacuous_nodes
    )
    return parsed_cot, parsed_root, tuple(unreachable), tuple(dependencies), passed


def joint_whole_cot_audit_messages(
    informal_statement: str,
    informal_proof: str,
    claimed_answer: str,
    view: FormalView,
) -> list[dict[str, str]]:
    unreachable = [node.node_name for node in view.nodes if not node.in_root_closure]
    node_shape = {
        "node_name": "n", "kind": "definition", "translation": "...",
        "semantic_effect": "objectDefinition", "introduced_objects": [],
        "referenced_objects": [],
    }
    payload = {
        "problem": informal_statement,
        "claimed_answer": claimed_answer,
        "complete_original_cot": informal_proof,
        "formal_view": view.to_dict(),
        "required_node_names": [node.node_name for node in view.nodes],
        "required_semantic_effect_values": [
            "objectDefinition", "proposition", "vacuous",
        ],
        "required_unreachable_node_names": unreachable,
        "required_output_shape": {
            "formal_decompiler": {"nodes": [node_shape]},
            "whole_cot_comparator": {
                "cot": {
                    "combined_formal_translation": "...",
                    "missing_clauses": [], "weakened_clauses": [],
                    "unbound_objects": [], "wrong_relations": [],
                    "added_clauses": [],
                },
                "root": {
                    "translation": "...", "target_object_preserved": True,
                    "answer_grounded": True, "reasons": [],
                },
                "unreachable_nodes": [{
                    "node_name": "n", "justified_side_branch": False,
                    "reason": "...",
                }],
                "dependency_issues": [{"node_name": "n", "reason": "..."}],
            },
        },
    }
    return [
        {"role": "system", "content": JOINT_WHOLE_COT_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def parse_joint_whole_cot_audit(
    content: str,
    *,
    view: FormalView,
) -> tuple[
    tuple[FormalNodeTranslation, ...],
    dict[str, Any],
    dict[str, Any],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    bool,
]:
    value = _json_object(content)
    if list(value) != ["formal_decompiler", "whole_cot_comparator"]:
        raise SemanticAuditFormatError(
            "joint top-level keys must be formal_decompiler then whole_cot_comparator",
            raw_content=content,
        )
    translations = parse_formal_decompiler(
        json.dumps(value["formal_decompiler"], ensure_ascii=False), view=view,
        allow_joint_aliases=True,
    )
    decompiler = FormalDecompilerResult(
        translations, json.dumps(value["formal_decompiler"], ensure_ascii=False),
        "", None, "", 0, 0, 0,
    )
    cot, root, unreachable, dependencies, passed = parse_whole_cot_comparator(
        json.dumps(value["whole_cot_comparator"], ensure_ascii=False),
        view=view,
        decompiler=decompiler,
    )
    return translations, cot, root, unreachable, dependencies, passed


def semantic_audit_cache_key(
    model: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    version: str = WHOLE_COT_PROMPT_VERSION,
    request_params: Mapping[str, Any] | None = None,
) -> str:
    key_payload: dict[str, Any] = {
        "version": version, "model": model, "messages": list(messages),
    }
    if request_params:
        key_payload["request_params"] = dict(request_params)
    payload = json.dumps(
        key_payload,
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _response_parts(response: Any) -> tuple[str, str, str | None, str, tuple[int, int, int]]:
    choice = response.choices[0]
    message = choice.message
    content = str(getattr(message, "content", None) or "")
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning is None:
        # vLLM's current OpenAI-compatible response model exposes Qwen
        # thinking output as ``message.reasoning`` rather than
        # ``message.reasoning_content``.
        reasoning = getattr(message, "reasoning", None)
    if reasoning is None and getattr(message, "model_extra", None):
        reasoning = (
            message.model_extra.get("reasoning_content")
            or message.model_extra.get("reasoning")
        )
    usage = getattr(response, "usage", None)
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    completion = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
    total = int(getattr(usage, "total_tokens", 0) or prompt + completion) if usage else 0
    request_id = str(getattr(response, "id", None) or getattr(response, "_request_id", None) or "")
    return content, str(reasoning or ""), getattr(choice, "finish_reason", None), request_id, (prompt, completion, total)


def _run_stage(
    client: Any,
    model: str,
    *,
    messages: list[dict[str, str]],
    parser,
    parser_kwargs: dict[str, Any],
    max_tokens: int,
    max_attempts: int,
    tracer,
    thm_name: str,
    round_index: int,
    phase: str,
    operation: str,
    enable_thinking: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    min_p: float = 0.0,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
) -> tuple[Any, str, str, str | None, str, tuple[dict[str, Any], ...], tuple[int, int, int]]:
    base_messages = list(messages)
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        request_started_ns = time.monotonic_ns()
        response = chat_completion_with_retry(
            client, tracer=tracer, thm_name=thm_name, phase=phase,
            model_id=model, operation=operation,
            trace_args={
                "round": round_index, "attempt": attempt,
                "enable_thinking": enable_thinking,
                "temperature": temperature, "top_p": top_p,
                "top_k": top_k, "min_p": min_p,
                "presence_penalty": presence_penalty,
                "repetition_penalty": repetition_penalty,
            },
            model=model, messages=messages, temperature=temperature,
            top_p=top_p, presence_penalty=presence_penalty,
            max_completion_tokens=max_tokens,
            extra_body={
                "top_k": top_k,
                "min_p": min_p,
                "repetition_penalty": repetition_penalty,
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            },
        )
        content, reasoning, finish_reason, request_id, usage = _response_parts(response)
        latency_ms = (time.monotonic_ns() - request_started_ns) / 1_000_000
        truncated = str(finish_reason or "").lower() == "length"
        attempts.append({
            "attempt": attempt, "rawContent": content, "reasoningContent": reasoning,
            "finishReason": finish_reason, "requestId": request_id,
            "truncated": truncated, "promptTokens": usage[0],
            "completionTokens": usage[1], "totalTokens": usage[2],
            "latencyMs": latency_ms,
        })
        try:
            if truncated:
                raise SemanticAuditFormatError("response truncated", raw_content=content)
            parsed = parser(content, **parser_kwargs)
        except SemanticAuditFormatError as exc:
            if attempt == max_attempts:
                raise SemanticAuditFormatError(
                    exc.reason, raw_content=content, attempts=attempts,
                ) from exc
            if operation == "formal_decompiler":
                schema_guidance = (
                    "Every nodes item must have exactly node_name, kind, translation, "
                    "semantic_effect, introduced_objects, and referenced_objects. "
                    "Copy the required node inventory exactly in the supplied order."
                )
            elif operation == "whole_cot_comparator":
                schema_guidance = (
                    "Top-level keys are exactly cot, root, unreachable_nodes, and "
                    "dependency_issues. Every COT clause issue is exactly "
                    "{clause,node_names,reason}; copy the required unreachable node "
                    "inventory exactly and use node names only."
                )
            elif operation == "joint_whole_cot_audit":
                schema_guidance = (
                    "Top-level keys must occur exactly in this order: "
                    "formal_decompiler, whole_cot_comparator. The first has "
                    "exactly nodes in the supplied order. The second has exactly "
                    "cot, root, unreachable_nodes, dependency_issues. Clause "
                    "issues are exactly {clause,node_names,reason}; use node names only. "
                    "semantic_effect must be exactly objectDefinition, proposition, "
                    "or vacuous. Keep node translations under 30 words, the combined "
                    "translation under 40 words, and reasons under 15 words."
                )
            else:
                schema_guidance = (
                    "Copy all required node inventories exactly. Keep translations, "
                    "clauses, and reasons terse; omit all explanation outside JSON."
                )
            messages = [
                *base_messages,
                {"role": "user", "content": (
                    f"The previous JSON was rejected: {exc.reason}. Generate it again "
                    "from the original input. Return one corrected JSON object only. "
                    "Use no Markdown and no extra keys. " + schema_guidance
                )},
            ]
            continue
        aggregate = tuple(sum(item[key] for item in attempts) for key in (
            "promptTokens", "completionTokens", "totalTokens",
        ))
        return parsed, content, reasoning, finish_reason, request_id, tuple(attempts), aggregate
    raise AssertionError("unreachable")


def run_formal_decompiler(
    client: Any,
    model: str,
    *,
    view: FormalView,
    max_tokens: int,
    max_attempts: int,
    enable_thinking: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    min_p: float = 0.0,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    tracer=None,
    thm_name: str = "",
    round_index: int = 0,
) -> FormalDecompilerResult:
    if tracer is not None:
        tracer.emit(TraceEvent(
            kind="formalDecompileStart", thm_name=thm_name, turn=round_index,
            args={"round": round_index, "formalViewHash": view.sha256, "nodeCount": len(view.nodes)},
        ))
    parsed, content, reasoning, finish, request_id, attempts, usage = _run_stage(
        client, model, messages=formal_decompiler_messages(view),
        parser=parse_formal_decompiler, parser_kwargs={"view": view},
        max_tokens=max_tokens, max_attempts=max_attempts, tracer=tracer,
        thm_name=thm_name, round_index=round_index,
        phase="formalDecompiler", operation="formal_decompiler",
        enable_thinking=enable_thinking, temperature=temperature,
        top_p=top_p, top_k=top_k, min_p=min_p,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
    )
    result = FormalDecompilerResult(
        parsed, content, reasoning, finish, request_id,
        usage[0], usage[1], usage[2], attempts,
    )
    if tracer is not None:
        tracer.emit(TraceEvent(
            kind="formalDecompileResult", thm_name=thm_name, turn=round_index,
            args={"round": round_index, "formalViewHash": view.sha256,
                  "vacuousNodes": list(result.vacuous_nodes), "result": result.to_dict()},
            ok=not bool(result.vacuous_nodes),
        ))
        tracer.emit(TraceEvent(
            kind="formalDecompileEnd", thm_name=thm_name, turn=round_index,
            args={"round": round_index, "attemptCount": len(attempts),
                  "promptTokens": usage[0], "completionTokens": usage[1],
                  "totalTokens": usage[2], "requestId": request_id},
            ok=True,
        ))
    return result


def run_whole_cot_comparator(
    client: Any,
    model: str,
    *,
    informal_statement: str,
    informal_proof: str,
    claimed_answer: str,
    view: FormalView,
    decompiler: FormalDecompilerResult,
    max_tokens: int,
    max_attempts: int,
    enable_thinking: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    min_p: float = 0.0,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    tracer=None,
    thm_name: str = "",
    round_index: int = 0,
) -> WholeCotComparatorResult:
    messages = whole_cot_comparator_messages(
        informal_statement, informal_proof, claimed_answer, view, decompiler,
    )
    cache_key = semantic_audit_cache_key(model, messages, version=WHOLE_COT_PROMPT_VERSION)
    if tracer is not None:
        tracer.emit(TraceEvent(
            kind="wholeCotCompareStart", thm_name=thm_name, turn=round_index,
            args={"round": round_index, "formalViewHash": view.sha256,
                  "cacheKey": cache_key, "protocol": WHOLE_COT_PROMPT_VERSION},
        ))
    parsed, content, reasoning, finish, request_id, attempts, usage = _run_stage(
        client, model, messages=messages, parser=parse_whole_cot_comparator,
        parser_kwargs={"view": view, "decompiler": decompiler},
        max_tokens=max_tokens, max_attempts=max_attempts, tracer=tracer,
        thm_name=thm_name, round_index=round_index,
        phase="wholeCotComparator", operation="whole_cot_comparator",
        enable_thinking=enable_thinking, temperature=temperature,
        top_p=top_p, top_k=top_k, min_p=min_p,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
    )
    cot, root, unreachable, dependencies, passed = parsed
    result = WholeCotComparatorResult(
        cot, root, unreachable, dependencies, passed, content, reasoning, finish,
        request_id, usage[0], usage[1], usage[2], attempts,
    )
    if tracer is not None:
        tracer.emit(TraceEvent(
            kind="wholeCotCompareResult", thm_name=thm_name, turn=round_index,
            args={"round": round_index, "formalViewHash": view.sha256,
                  "cacheKey": cache_key, "protocol": WHOLE_COT_PROMPT_VERSION,
                  "passed": passed, "result": result.to_dict()}, ok=passed,
        ))
        tracer.emit(TraceEvent(
            kind="wholeCotCompareEnd", thm_name=thm_name, turn=round_index,
            args={"round": round_index, "passed": passed,
                  "attemptCount": len(attempts), "promptTokens": usage[0],
                  "completionTokens": usage[1], "totalTokens": usage[2],
                  "requestId": request_id}, ok=passed,
        ))
    return result


def run_joint_whole_cot_audit(
    client: Any,
    model: str,
    *,
    informal_statement: str,
    informal_proof: str,
    claimed_answer: str,
    view: FormalView,
    max_tokens: int,
    max_attempts: int,
    enable_thinking: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    min_p: float = 0.0,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    tracer=None,
    thm_name: str = "",
    round_index: int = 0,
) -> JointWholeCotAuditResult:
    messages = joint_whole_cot_audit_messages(
        informal_statement, informal_proof, claimed_answer, view,
    )
    request_params = {
        "enable_thinking": enable_thinking, "temperature": temperature,
        "top_p": top_p, "top_k": top_k, "min_p": min_p,
        "presence_penalty": presence_penalty,
        "repetition_penalty": repetition_penalty,
        "max_tokens": max_tokens,
    }
    cache_key = semantic_audit_cache_key(
        model, messages, version=JOINT_WHOLE_COT_PROMPT_VERSION,
        request_params=request_params,
    )
    if tracer is not None:
        tracer.emit(TraceEvent(
            kind="jointSemanticAuditStart", thm_name=thm_name, turn=round_index,
            args={
                "round": round_index, "formalViewHash": view.sha256,
                "cacheKey": cache_key, "protocol": JOINT_WHOLE_COT_PROMPT_VERSION,
                "nodeCount": len(view.nodes), "maxCompletionTokens": max_tokens,
                **request_params,
            },
        ))
    parsed, content, reasoning, finish, request_id, attempts, usage = _run_stage(
        client, model, messages=messages, parser=parse_joint_whole_cot_audit,
        parser_kwargs={"view": view}, max_tokens=max_tokens,
        max_attempts=max_attempts, tracer=tracer, thm_name=thm_name,
        round_index=round_index, phase="jointSemanticAudit",
        operation="joint_whole_cot_audit", enable_thinking=enable_thinking,
        temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
    )
    translations, cot, root, unreachable, dependencies, passed = parsed
    raw_value = _json_object(content)
    raw_nodes = raw_value["formal_decompiler"]["nodes"]
    effect_normalizations = tuple(
        {
            "node_name": translation.node_name,
            "reported": str(raw_node.get("semantic_effect") or ""),
            "canonical": translation.semantic_effect,
        }
        for raw_node, translation in zip(raw_nodes, translations, strict=True)
        if raw_node.get("semantic_effect") != translation.semantic_effect
    )
    decompiler = FormalDecompilerResult(
        translations,
        json.dumps(raw_value["formal_decompiler"], ensure_ascii=False),
        "", finish, "", 0, 0, 0, (),
    )
    comparator = WholeCotComparatorResult(
        cot, root, unreachable, dependencies, passed,
        json.dumps(raw_value["whole_cot_comparator"], ensure_ascii=False),
        "", finish, "", 0, 0, 0, (),
    )
    result = JointWholeCotAuditResult(
        decompiler, comparator, content, reasoning, finish, request_id,
        usage[0], usage[1], usage[2], attempts, effect_normalizations,
    )
    if tracer is not None:
        tracer.emit(TraceEvent(
            kind="jointSemanticAuditResult", thm_name=thm_name, turn=round_index,
            args={
                "round": round_index, "formalViewHash": view.sha256,
                "cacheKey": cache_key, "protocol": JOINT_WHOLE_COT_PROMPT_VERSION,
                "passed": passed, "vacuousNodes": list(decompiler.vacuous_nodes),
                "formalDecompiler": decompiler.to_dict(),
                "wholeCotComparator": comparator.to_dict(),
                "semanticEffectNormalizations": list(effect_normalizations),
            },
            ok=passed,
        ))
        tracer.emit(TraceEvent(
            kind="jointSemanticAuditEnd", thm_name=thm_name, turn=round_index,
            args={
                "round": round_index, "passed": passed,
                "attemptCount": len(attempts), "promptTokens": usage[0],
                "completionTokens": usage[1], "totalTokens": usage[2],
                "finishReason": finish, "requestId": request_id,
                "actualRequestCount": len(attempts),
            },
            ok=passed,
        ))
    return result


def whole_cot_comparator_defects(result: WholeCotComparatorResult) -> list[dict[str, Any]]:
    defects: list[dict[str, Any]] = []
    for category in (
        "missing_clauses", "weakened_clauses", "unbound_objects",
        "wrong_relations", "added_clauses",
    ):
        for item in result.cot[category]:
            defects.append({
                "category": category,
                "node_names": list(item["node_names"]),
                "requirement": item["clause"],
                "reason": item["reason"],
            })
    if not result.root["target_object_preserved"]:
        defects.append({
            "category": "rootTargetObject", "node_names": [],
            "requirement": "Preserve the source target object in root.",
            "reason": "; ".join(result.root["reasons"]) or "Root target object changed.",
        })
    if not result.root["answer_grounded"]:
        defects.append({
            "category": "rootAnswerGrounding", "node_names": [],
            "requirement": "Ground the answer in the source object.",
            "reason": "; ".join(result.root["reasons"]) or "Answer is ungrounded.",
        })
    if result.root["reasons"]:
        defects.append({
            "category": "rootSemanticReasons", "node_names": [],
            "requirement": "Resolve every root semantic defect and return no pass rationale.",
            "reason": "; ".join(result.root["reasons"]),
        })
    for item in result.unreachable_nodes:
        if not item["justified_side_branch"]:
            defects.append({
                "category": "dagDisconnected", "node_names": [item["node_name"]],
                "requirement": "Connect this required node to root.",
                "reason": item["reason"],
            })
    for item in result.dependency_issues:
        defects.append({
            "category": "dependencyFidelity", "node_names": [item["node_name"]],
            "requirement": "Repair the source use-chain dependency.",
            "reason": item["reason"],
        })
    return defects



__all__ = [
    "WHOLE_COT_PROMPT_VERSION", "JOINT_WHOLE_COT_PROMPT_VERSION",
    "JOINT_SEMANTIC_EFFECT_ALIASES", "JOINT_WHOLE_COT_SYSTEM_PROMPT",
    "FormalDecompilerResult", "FormalView", "JointWholeCotAuditResult",
    "SemanticAuditFormatError", "WholeCotComparatorResult",
    "build_formal_view",
    "formal_decompiler_messages", "parse_formal_decompiler", "run_formal_decompiler",
    "parse_whole_cot_comparator", "run_whole_cot_comparator",
    "joint_whole_cot_audit_messages", "parse_joint_whole_cot_audit",
    "run_joint_whole_cot_audit",
    "semantic_audit_cache_key", "strict_comparator_messages",
    "whole_cot_comparator_defects", "whole_cot_comparator_messages",
]

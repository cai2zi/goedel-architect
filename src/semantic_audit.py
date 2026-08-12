"""Strict two-stage semantic audit for complete Blueprint generation.

The formal decompiler never sees the problem or COT.  The strict comparator
receives its frozen literal translation and reports concrete omissions or
weakenings.  The runner, rather than the model, computes the final PASS bit.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from llm_client import chat_completion_with_retry
from tracer import TraceEvent


PROMPT_VERSION = "blueprint-semantic-audit-v1"
WHOLE_COT_PROMPT_VERSION = "whole-cot-comparator-v1"
SEMANTIC_EFFECTS = {"objectDefinition", "proposition", "vacuous"}


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


COMPARATOR_SYSTEM_PROMPT = r"""You are a strict semantic-translation
comparator, not a proof or truth judge. The source COT may be mathematically
wrong, contradictory, or unsupported. An exact formalization of that wrong
claim is faithful. Reject only formal omissions, weakenings, added content,
object replacement, unbound objects, wrong relation/direction, or a DAG that
does not preserve the source use-chain.

The formal translations were produced by a separate decompiler that could not
see the COT. Treat them as frozen evidence. A matching answer literal is never
sufficient: the root must constrain the same source object used by the COT.
Comments and names carry no semantic credit. `exists x, x = answer` is
unfaithful unless x is constrained as the source object. A Step mapped to
several nodes is covered only when the nodes jointly encode every mathematical
clause. A vacuous node never contributes coverage.

Dependencies are judged separately from node truth. Do not demand a formal
proof that a parent entails its child; decide whether the listed parents match
the earlier COT results the child claims to use. An unreachable Step is allowed
only when the COT explicitly abandons it, diagnoses it as an error, or keeps it
as a genuinely independent branch. Ordinary sequential calculations used in
the final answer are not side branches.

Perform a mechanical clause audit before writing JSON:
1. Split each source Step into its atomic mathematical assertions, including
   quantifier strength (`all`, `exists`, `unique`, `exactly`), object identity,
   restrictions, relation direction, and asserted conclusion.
2. Compare each assertion only with the literal frozen node translations and
   sanitized formal types mapped to that Step. A node identifier, dependency
   label, or suggestive English name is routing metadata and earns zero credit.
3. A property present only as a hypothesis is not a formalized conclusion. A
   theorem that assumes the desired relation and concludes an already-given
   fact is missing or direction-reversed.
4. Do not add source concepts while composing `combined_formal_translation`.
   It must be a faithful compression of the frozen translations. If the source
   says "diameters form a unique rectangle" while the formal conclusion only
   constructs a four-element endpoint set, report the rectangle and uniqueness
   clauses as missing; never insert them into the combined translation.
5. Existence does not encode uniqueness, a finite collection of stated size
   does not say it is the complete collection of valid objects, and arbitrary
   coordinates do not encode the original geometric object unless the formal
   constraints bind them to it.
6. Compare every conjunct and inequality. Boundary-only, one-sided, or
   restricted-family claims are weakened when the source includes an interior,
   converse, total-count bridge, or additional condition.
7. Give no credit to ex-falso encodings. A node with a literal `False`
   premise does not formalize the source clause merely because it can conclude
   any proposition.
8. A numeric value introduced by a definition is not a derived result unless
   its formal body is connected to the source objects and relations. Merely
   assigning the COT answer to a fresh constant is answer hard-coding.
9. Track object identity across Steps. Independent existential variables or
   freshly rebound coordinates do not represent the same source object just
   because their names or values resemble it.
10. The root must mention the shared target object and its relevant relation
    in its formal type. A dependency edge alone does not repair an unrelated
    root conclusion.

Never repair, reinterpret, or charitably complete the formalization. When in
doubt, describe the literal formal conclusion first and record the absent
source clause in the appropriate issue array.

Return exactly one JSON object and no Markdown. It must have exactly these
top-level keys: `steps`, `root`, `unreachable_steps`, `dependency_issues`, and
`obligation_results`. Every issue in the four Step issue arrays has keys
`clause`, `node_names`, and `reason`, with no extra keys. Every dependency issue
has exactly `node_name`, `step_id`, and `reason`. Copy the supplied Step,
unreachable-Step, and open-obligation inventories exactly and in order. Do not
invent obligation IDs: when `required_obligation_ids` is empty,
`obligation_results` must be `[]`. When there are no dependency issues, return
`dependency_issues: []`.

Be terse so the complete inventory fits the output budget. Each
`combined_formal_translation` and the root translation must be at most 25
words. Each clause must be at most 20 words and every reason at most 15 words.
An obligation reason may simply say "fixed" or name the still-missing formal
relation. Never restate the full source Step inside a reason, and do not
duplicate the same explanation across issue categories. Empty inventories
must be literal `[]`. Completeness of the JSON inventory is more important
than stylistic explanation.

An open obligation is a persistent repair question, not an immutable verdict.
Re-evaluate every obligation against the CURRENT Formal View. Set `resolved`
to true exactly when its stated defect is absent now. In particular, if an old
`vacuousNode` obligation names a node that is now a concrete non-vacuous
proposition/object definition and faithfully covers its source clause, return
`resolved:true`; do not keep it false merely because it appears in the open
inventory or because the former diagnosis was once valid. Conversely, an
obligation may not disappear by omission: copy its ID and explicitly report
the current evidence. The issue arrays describe current defects; do not
re-create a repaired historical defect just to justify keeping an obligation
open.
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


@dataclass(frozen=True)
class FormalNodeView:
    node_name: str
    kind: str
    step_id: str
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
    includes_step_ids: bool = True

    def to_dict(self) -> dict[str, Any]:
        nodes = []
        for node in self.nodes:
            item = asdict(node)
            if not self.includes_step_ids:
                item.pop("step_id", None)
            nodes.append(item)
        return {
            "nodes": nodes,
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
class StrictComparatorResult:
    steps: tuple[dict[str, Any], ...]
    root: dict[str, Any]
    unreachable_steps: tuple[dict[str, Any], ...]
    dependency_issues: tuple[dict[str, Any], ...]
    obligation_results: tuple[dict[str, Any], ...]
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


def _base_step_id(value: str) -> str:
    return str(value or "").split(".", 1)[0]


def build_formal_view(blueprint: Any, *, include_step_ids: bool = True) -> FormalView:
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
        _base_step_id(node.source_step_id),
        effective[node.name],
        declarations[node.name],
        node.name == blueprint.target_theorem,
        node.name in closure,
    ) for node in blueprint.nodes)
    payload_nodes = []
    for node in views:
        item = asdict(node)
        if not include_step_ids:
            item.pop("step_id", None)
        payload_nodes.append(item)
    payload = json.dumps(
        {"root": blueprint.target_theorem, "nodes": payload_nodes},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return FormalView(
        views,
        blueprint.target_theorem,
        tuple(node.node_name for node in views if node.in_root_closure),
        hashlib.sha256(payload.encode()).hexdigest(),
        include_step_ids,
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
        if view.includes_step_ids:
            item["step_id"] = node.step_id
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


def parse_formal_decompiler(content: str, *, view: FormalView) -> tuple[FormalNodeTranslation, ...]:
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
        effect = _string(item["semantic_effect"], "semantic_effect", content)
        if effect not in SEMANTIC_EFFECTS:
            raise SemanticAuditFormatError(f"invalid semantic_effect {effect}", raw_content=content)
        parsed.append(FormalNodeTranslation(
            _string(item["node_name"], "node_name", content),
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


def _manifest_steps(manifest: Any) -> list[Any]:
    return list(getattr(manifest, "steps", ()) or ())


def _unreachable_inventory(view: FormalView, manifest: Any) -> list[dict[str, Any]]:
    nodes_by_step: dict[str, list[FormalNodeView]] = {}
    for node in view.nodes:
        nodes_by_step.setdefault(node.step_id, []).append(node)
    inventory = []
    for step in _manifest_steps(manifest):
        nodes = nodes_by_step.get(step.step_id, [])
        if nodes and not any(node.in_root_closure for node in nodes):
            inventory.append({
                "step_id": step.step_id,
                "node_names": [node.node_name for node in nodes],
            })
    return inventory


def strict_comparator_messages(
    informal_statement: str,
    claimed_answer: str,
    manifest: Any,
    view: FormalView,
    decompiler: FormalDecompilerResult,
    open_obligations: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    steps = [{"step_id": step.step_id, "source": step.source_text} for step in _manifest_steps(manifest)]
    payload = {
        "problem": informal_statement,
        "claimed_answer": claimed_answer,
        "cot_steps": steps,
        "formal_view": view.to_dict(),
        "frozen_node_translations": [asdict(node) for node in decompiler.nodes],
        "unreachable_step_inventory": _unreachable_inventory(view, manifest),
        "unreachable_node_inventory": [
            node.node_name for node in view.nodes if not node.in_root_closure
        ],
        "open_obligations": [dict(item) for item in open_obligations],
        "required_unreachable_step_ids": [
            item["step_id"] for item in _unreachable_inventory(view, manifest)
        ],
        "required_obligation_ids": [
            str(item.get("obligation_id") or "") for item in open_obligations
        ],
        "required_output_shape": {
            "steps": [{
                "step_id": "S001",
                "combined_formal_translation": "...",
                "missing_clauses": [{"clause": "...", "node_names": ["n"], "reason": "..."}],
                "weakened_clauses": [{"clause": "...", "node_names": ["n"], "reason": "..."}],
                "unbound_objects": [{"clause": "...", "node_names": ["n"], "reason": "..."}],
                "wrong_relations": [{"clause": "...", "node_names": ["n"], "reason": "..."}],
            }],
            "root": {
                "translation": "...",
                "target_object_preserved": True,
                "answer_grounded": True,
                "reasons": [],
            },
            "unreachable_steps": [{
                "step_id": "S003", "justified_side_branch": False, "reason": "...",
            }],
            "dependency_issues": [{"node_name": "n", "step_id": "S001", "reason": "..."}],
            "obligation_results": [{"obligation_id": "semantic:...", "resolved": False, "reason": "..."}],
        },
    }
    return [
        {"role": "system", "content": COMPARATOR_SYSTEM_PROMPT},
        {"role": "user", "content": (
            json.dumps(payload, ensure_ascii=False)
            + "\n\nNON-NEGOTIABLE OUTPUT INVENTORY:\n"
            + "- Step IDs in order: " + json.dumps([step["step_id"] for step in steps])
            + "\n- unreachable_steps IDs in order: "
            + json.dumps(payload["required_unreachable_step_ids"])
            + "\n- obligation_results IDs in order: "
            + json.dumps(payload["required_obligation_ids"])
            + "\nCopy these three inventories exactly. Every item in missing_clauses, "
              "weakened_clauses, unbound_objects, and wrong_relations must use "
              "exactly {clause, node_names, reason}; even an unbound object issue "
              "must use `clause` and `node_names`, never an `object` key."
        )},
    ]


_STEP_KEYS = {
    "step_id", "combined_formal_translation", "missing_clauses",
    "weakened_clauses", "unbound_objects", "wrong_relations",
}
_ISSUE_KEYS = {"clause", "node_names", "reason"}


def _parse_step_issues(
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
            raise SemanticAuditFormatError(f"{label}[{index}] has invalid keys", raw_content=raw)
        nodes = _strings(item["node_names"], f"{label}.node_names", raw)
        if any(node not in known_nodes for node in nodes):
            raise SemanticAuditFormatError(f"{label} references unknown node", raw_content=raw)
        parsed.append({
            "clause": _string(item["clause"], f"{label}.clause", raw),
            "node_names": list(nodes),
            "reason": _string(item["reason"], f"{label}.reason", raw),
        })
    return parsed


def parse_strict_comparator(
    content: str,
    *,
    manifest: Any,
    view: FormalView,
    decompiler: FormalDecompilerResult,
    open_obligations: Sequence[Mapping[str, Any]],
) -> tuple[
    tuple[dict[str, Any], ...], dict[str, Any], tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], bool,
]:
    value = _json_object(content)
    top_keys = {"steps", "root", "unreachable_steps", "dependency_issues", "obligation_results"}
    if set(value) != top_keys:
        raise SemanticAuditFormatError("invalid comparator top-level keys", raw_content=content)
    if not isinstance(value["steps"], list):
        raise SemanticAuditFormatError("steps must be an array", raw_content=content)
    known_nodes = {node.node_name for node in view.nodes}
    known_steps = [step.step_id for step in _manifest_steps(manifest)]
    parsed_steps = []
    for index, item in enumerate(value["steps"]):
        if not isinstance(item, dict) or set(item) != _STEP_KEYS:
            raise SemanticAuditFormatError(f"steps[{index}] has invalid keys", raw_content=content)
        step_id = _string(item["step_id"], "step_id", content)
        parsed = {
            "step_id": step_id,
            "combined_formal_translation": _string(
                item["combined_formal_translation"], "combined_formal_translation", content,
            ),
        }
        for key in ("missing_clauses", "weakened_clauses", "unbound_objects", "wrong_relations"):
            parsed[key] = _parse_step_issues(
                item[key], label=f"{step_id}.{key}", known_nodes=known_nodes, raw=content,
            )
        parsed_steps.append(parsed)
    if [item["step_id"] for item in parsed_steps] != known_steps:
        raise SemanticAuditFormatError("Step inventory is incomplete, duplicated, or reordered", raw_content=content)

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

    expected_unreachable = [item["step_id"] for item in _unreachable_inventory(view, manifest)]
    raw_unreachable = value["unreachable_steps"]
    if not isinstance(raw_unreachable, list):
        raise SemanticAuditFormatError("unreachable_steps must be an array", raw_content=content)
    unreachable = []
    for item in raw_unreachable:
        if not isinstance(item, dict) or set(item) != {"step_id", "justified_side_branch", "reason"}:
            raise SemanticAuditFormatError("unreachable_steps item has invalid keys", raw_content=content)
        if not isinstance(item["justified_side_branch"], bool):
            raise SemanticAuditFormatError("justified_side_branch must be boolean", raw_content=content)
        unreachable.append({
            "step_id": _string(item["step_id"], "unreachable step_id", content),
            "justified_side_branch": item["justified_side_branch"],
            "reason": _string(item["reason"], "unreachable reason", content),
        })
    if [item["step_id"] for item in unreachable] != expected_unreachable:
        raise SemanticAuditFormatError(
            "unreachable Step inventory is incomplete or reordered; "
            f"expected={expected_unreachable} got={[item['step_id'] for item in unreachable]}",
            raw_content=content,
        )

    raw_dependencies = value["dependency_issues"]
    if not isinstance(raw_dependencies, list):
        raise SemanticAuditFormatError("dependency_issues must be an array", raw_content=content)
    dependencies = []
    for item in raw_dependencies:
        if not isinstance(item, dict) or set(item) != {"node_name", "step_id", "reason"}:
            raise SemanticAuditFormatError("dependency issue has invalid keys", raw_content=content)
        node_name = _string(item["node_name"], "dependency node", content)
        step_id = _string(item["step_id"], "dependency step", content)
        if node_name not in known_nodes or step_id not in known_steps:
            raise SemanticAuditFormatError("dependency issue has unknown node/Step", raw_content=content)
        dependencies.append({
            "node_name": node_name, "step_id": step_id,
            "reason": _string(item["reason"], "dependency reason", content),
        })

    expected_obligations = [str(item.get("obligation_id") or "") for item in open_obligations]
    raw_results = value["obligation_results"]
    if not isinstance(raw_results, list):
        raise SemanticAuditFormatError("obligation_results must be an array", raw_content=content)
    obligation_results = []
    for item in raw_results:
        if not isinstance(item, dict) or set(item) != {"obligation_id", "resolved", "reason"}:
            raise SemanticAuditFormatError("obligation result has invalid keys", raw_content=content)
        if not isinstance(item["resolved"], bool):
            raise SemanticAuditFormatError("obligation resolved must be boolean", raw_content=content)
        obligation_results.append({
            "obligation_id": _string(item["obligation_id"], "obligation_id", content),
            "resolved": item["resolved"],
            "reason": _string(item["reason"], "obligation reason", content),
        })
    if [item["obligation_id"] for item in obligation_results] != expected_obligations:
        raise SemanticAuditFormatError(
            "obligation inventory is incomplete or reordered; "
            f"expected={expected_obligations} "
            f"got={[item['obligation_id'] for item in obligation_results]}",
            raw_content=content,
        )

    has_step_issues = any(
        any(step[key] for key in ("missing_clauses", "weakened_clauses", "unbound_objects", "wrong_relations"))
        for step in parsed_steps
    )
    passed = (
        not has_step_issues
        and parsed_root["target_object_preserved"]
        and parsed_root["answer_grounded"]
        and not parsed_root["reasons"]
        and all(item["justified_side_branch"] for item in unreachable)
        and not dependencies
        and all(item["resolved"] for item in obligation_results)
        and not decompiler.vacuous_nodes
    )
    return (
        tuple(parsed_steps), parsed_root, tuple(unreachable), tuple(dependencies),
        tuple(obligation_results), passed,
    )


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
        parsed_cot[key] = _parse_step_issues(
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


def semantic_audit_cache_key(
    model: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    version: str = PROMPT_VERSION,
) -> str:
    payload = json.dumps(
        {"version": version, "model": model, "messages": list(messages)},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _response_parts(response: Any) -> tuple[str, str, str | None, str, tuple[int, int, int]]:
    choice = response.choices[0]
    message = choice.message
    content = str(getattr(message, "content", None) or "")
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning is None and getattr(message, "model_extra", None):
        reasoning = message.model_extra.get("reasoning_content")
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
        truncated = str(finish_reason or "").lower() == "length"
        attempts.append({
            "attempt": attempt, "rawContent": content, "reasoningContent": reasoning,
            "finishReason": finish_reason, "requestId": request_id,
            "truncated": truncated, "promptTokens": usage[0],
            "completionTokens": usage[1], "totalTokens": usage[2],
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
                    "inventory exactly and use no Step IDs."
                )
            else:
                schema_guidance = (
                    "A Step issue is exactly {\"clause\":string,"
                    "\"node_names\":[string],\"reason\":string}; a dependency issue "
                    "is exactly {\"node_name\":string,\"step_id\":string,"
                    "\"reason\":string}; an obligation result is exactly "
                    "{\"obligation_id\":string,\"resolved\":boolean,"
                    "\"reason\":string}. Copy all required inventories exactly and "
                    "do not invent obligation IDs. This is a compact retry: keep "
                    "translations at most 15 words and every clause/reason at "
                    "most 10 words; omit all explanation outside the JSON."
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


def run_strict_comparator(
    client: Any,
    model: str,
    *,
    informal_statement: str,
    claimed_answer: str,
    manifest: Any,
    view: FormalView,
    decompiler: FormalDecompilerResult,
    open_obligations: Sequence[Mapping[str, Any]],
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
) -> StrictComparatorResult:
    messages = strict_comparator_messages(
        informal_statement, claimed_answer, manifest, view, decompiler, open_obligations,
    )
    cache_key = semantic_audit_cache_key(model, messages)
    if tracer is not None:
        tracer.emit(TraceEvent(
            kind="strictCompareStart", thm_name=thm_name, turn=round_index,
            args={"round": round_index, "formalViewHash": view.sha256,
                  "cacheKey": cache_key, "stepCount": len(_manifest_steps(manifest)),
                  "openObligationCount": len(open_obligations)},
        ))
    parsed, content, reasoning, finish, request_id, attempts, usage = _run_stage(
        client, model, messages=messages, parser=parse_strict_comparator,
        parser_kwargs={
            "manifest": manifest, "view": view, "decompiler": decompiler,
            "open_obligations": open_obligations,
        },
        max_tokens=max_tokens, max_attempts=max_attempts, tracer=tracer,
        thm_name=thm_name, round_index=round_index,
        phase="strictComparator", operation="strict_comparator",
        enable_thinking=enable_thinking, temperature=temperature,
        top_p=top_p, top_k=top_k, min_p=min_p,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
    )
    steps, root, unreachable, dependencies, obligation_results, passed = parsed
    result = StrictComparatorResult(
        steps, root, unreachable, dependencies, obligation_results, passed,
        content, reasoning, finish, request_id, usage[0], usage[1], usage[2], attempts,
    )
    if tracer is not None:
        tracer.emit(TraceEvent(
            kind="strictCompareResult", thm_name=thm_name, turn=round_index,
            args={"round": round_index, "formalViewHash": view.sha256,
                  "cacheKey": cache_key, "passed": passed, "result": result.to_dict()},
            ok=passed,
        ))
        tracer.emit(TraceEvent(
            kind="strictCompareEnd", thm_name=thm_name, turn=round_index,
            args={"round": round_index, "passed": passed, "attemptCount": len(attempts),
                  "promptTokens": usage[0], "completionTokens": usage[1],
                  "totalTokens": usage[2], "requestId": request_id},
            ok=passed,
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


def comparator_defects(result: StrictComparatorResult) -> list[dict[str, Any]]:
    defects: list[dict[str, Any]] = []
    for step in result.steps:
        for category in ("missing_clauses", "weakened_clauses", "unbound_objects", "wrong_relations"):
            for item in step[category]:
                defects.append({
                    "category": category,
                    "step_id": step["step_id"],
                    "node_names": list(item["node_names"]),
                    "requirement": item["clause"],
                    "reason": item["reason"],
                })
    if not result.root["target_object_preserved"]:
        defects.append({
            "category": "rootTargetObject", "step_id": "",
            "node_names": [], "requirement": "Preserve the source target object in root.",
            "reason": "; ".join(result.root["reasons"]) or "Root target object changed.",
        })
    if not result.root["answer_grounded"]:
        defects.append({
            "category": "rootAnswerGrounding", "step_id": "",
            "node_names": [], "requirement": "Ground the answer in the source object.",
            "reason": "; ".join(result.root["reasons"]) or "Answer is ungrounded.",
        })
    for item in result.unreachable_steps:
        if not item["justified_side_branch"]:
            defects.append({
                "category": "dagDisconnected", "step_id": item["step_id"],
                "node_names": [], "requirement": "Connect this final-path Step to root.",
                "reason": item["reason"],
            })
    for item in result.dependency_issues:
        defects.append({
            "category": "dependencyFidelity", "step_id": item["step_id"],
            "node_names": [item["node_name"]],
            "requirement": "Repair the source use-chain dependency.",
            "reason": item["reason"],
        })
    return defects


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


def format_semantic_audit_feedback(
    decompiler: FormalDecompilerResult,
    comparator: StrictComparatorResult,
    open_obligations: Sequence[Mapping[str, Any]],
) -> str:
    if comparator.passed and not open_obligations and not decompiler.vacuous_nodes:
        return "Strict semantic audit PASSED."
    lines = ["Strict semantic audit FAILED."]
    if decompiler.vacuous_nodes:
        lines.append("Vacuous formal nodes: " + ", ".join(decompiler.vacuous_nodes))
    if open_obligations:
        lines.append("Open persistent semantic obligations:")
        for item in open_obligations:
            lines.append(
                f"- {item.get('obligation_id')} step={item.get('step_id') or '<root>'} "
                f"nodes={','.join(item.get('node_names') or []) or '<joint/root>'}: "
                f"{item.get('requirement')} Reason: {item.get('reason')}"
            )
    else:
        for defect in comparator_defects(comparator):
            nodes = ",".join(defect["node_names"]) or "<joint/root>"
            lines.append(
                f"- {defect['category']} step={defect['step_id'] or '<root>'} "
                f"nodes={nodes}: {defect['requirement']} Reason: {defect['reason']}"
            )
    return "\n".join(lines)


__all__ = [
    "PROMPT_VERSION", "WHOLE_COT_PROMPT_VERSION", "FormalDecompilerResult", "FormalView",
    "SemanticAuditFormatError", "StrictComparatorResult", "WholeCotComparatorResult",
    "build_formal_view",
    "comparator_defects", "formal_decompiler_messages", "format_semantic_audit_feedback",
    "parse_formal_decompiler", "parse_strict_comparator", "run_formal_decompiler",
    "parse_whole_cot_comparator", "run_strict_comparator", "run_whole_cot_comparator",
    "semantic_audit_cache_key", "strict_comparator_messages",
    "whole_cot_comparator_defects", "whole_cot_comparator_messages",
]

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


FORMAL_DECOMPILER_PROMPT_VERSION = "formal-decompiler-v2-product-metric"
WHOLE_COT_PROMPT_VERSION = "whole-cot-comparator-v3-semantic-repair"
COMPACT_WHOLE_COT_PROMPT_VERSION = "whole-cot-compact-separate-v4-semantic-repair"
DIRECT_WHOLE_COT_PROMPT_VERSION = "whole-cot-direct-comparator-v4-semantic-repair"
JOINT_WHOLE_COT_PROMPT_VERSION = "whole-cot-joint-audit-v4-semantic-repair"
CANONICAL_COMPACT_WHOLE_COT_PROMPT_VERSION = "whole-cot-compact-separate-canonical-v2-r4"
CANONICAL_DIRECT_WHOLE_COT_PROMPT_VERSION = "whole-cot-direct-canonical-v2-r4"
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

Apply Lean's metric instances literally. For `P Q : α × β`, an unqualified
`dist P Q` is the maximum of the two component distances, and the default
product norm is likewise max/sup, not Euclidean. Do not translate it as
Euclidean because of names, comments, or geometry context. Translate an
explicit sum of squared coordinate differences as squared Euclidean distance,
not ordinary distance; only an explicit square root of that sum is ordinary
Euclidean distance. `EuclideanSpace ℝ (Fin 2)` norm/dist is Euclidean.
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

Treat `ℝ × ℝ` and `ℚ × ℚ` as coordinate carriers, which is not itself a defect.
But their default product `dist`/`norm` is max/sup, not Euclidean. When the COT
requires Euclidean length, circle, or angle and the formalization uses that
product metric, report the mismatch in `cot.wrong_relations`; if it affects the
target/root, set the corresponding root boolean false. Do not equate explicit
squared Euclidean distance with ordinary distance unless the COT use-chain
preserves the square relation. `EuclideanSpace ℝ (Fin 2)` uses Euclidean metric.

Return one JSON object with exactly `cot`, `root`, `unreachable_nodes`,
`dependency_issues`, and `repair_issues`. `cot` has exactly `combined_formal_translation`,
`missing_clauses`, `weakened_clauses`, `unbound_objects`, `wrong_relations`,
and `added_clauses`. Each clause issue has exactly `clause`, `node_names`, and
`reason`. `root` has exactly `translation`, `target_object_preserved`,
`answer_grounded`, and `reasons`. Copy the supplied unreachable-node inventory
exactly and in order; each item has `node_name`, `justified_side_branch`, and
`reason`. Each dependency issue has exactly `node_name` and `reason`. Use only
supplied node names. Return JSON only, no Markdown or extra keys. Be terse.
"""


COMPACT_WHOLE_COT_COMPARATOR_SYSTEM_PROMPT = r"""You are a strict
semantic-translation comparator, not a truth judge. Compare the complete
original chain-of-thought with frozen literal translations produced by a
blind Lean decompiler. A faithful formalization of a mathematically wrong COT
must pass. Reject omissions, weakenings, added claims, object replacement,
unbound objects, wrong relation or direction, answer hard-coding, material
dependency breaks, and an unrelated root. Identifier names are opaque handles
and carry no semantic credit.

All supplied definitions are global context available to every proof node.
Definitions need not occur in `sorry_using`; an existing definition entry in
`sorry_using` is harmless and is not a proof-graph edge. Never report a
dependency issue because a definition is absent from `sorry_using`. Mere global
availability gives no semantic credit: a required source object or relation is
grounded only when the root or a root-reachable proposition actually references,
constrains, or relates it.

Treat `ℝ × ℝ` and `ℚ × ℚ` as valid coordinate carriers. Their default product
`dist`/`norm`, however, is max/sup rather than Euclidean. If the COT requires
Euclidean length, circle, or angle and the formalization uses the product
metric, report it in `cot.wrong_relations` and set an affected root boolean
false. An explicit coordinate square-sum is squared Euclidean distance, not
ordinary distance unless the COT use-chain preserves that square relation;
an explicit square root or `EuclideanSpace ℝ (Fin 2)` may represent Euclidean
distance.

Audit in this order: probability and quantifiers; target object and relation;
root grounding; then material dependency use-chains. Do not report a
dependency issue merely because a proof node is outside `proof_root_closure`.
Verification, abandoned derivations, and legitimate side branches need not
support the root. Report one
only when the COT materially requires that translated object or proposition to
support the root and the supplied dependency graph has no path carrying it to
the root. Do not duplicate one defect across categories.

The root must preserve the target object requested by the original problem,
not merely repeat a final equation. Root `reasons` list defects only and must
be empty when both root booleans are true. Return one JSON object with exactly
`cot`, `root`, `dependency_issues`, and `repair_issues`. `cot` has exactly
`combined_formal_translation`, `missing_clauses`, `weakened_clauses`,
`unbound_objects`, `wrong_relations`, and `added_clauses`. Each clause issue
has exactly `clause`, `node_names`, and `reason`. `root` has exactly
`translation`, `target_object_preserved`, `answer_grounded`, and `reasons`.
Each dependency issue has exactly `node_name` and `reason`. Use only supplied
proof-node names for dependency issues. Return JSON only, no Markdown or extra
keys. Be terse: at most six
clause issues total, and keep each clause and reason under 50 words.
"""


DIRECT_WHOLE_COT_COMPARATOR_SYSTEM_PROMPT = r"""You are a strict literal Lean
semantic decompiler and semantic-translation comparator in one request. Read
only the sanitized Lean declarations for formal meaning; identifier names are
opaque handles and carry no semantic credit. Then compare that meaning with
the complete original chain-of-thought. A faithful formalization of a
mathematically wrong COT must pass.

Reject omissions, weakenings, added claims, object replacement, unbound
objects, wrong relation or direction, answer hard-coding, vacuous True or
reflexive shells replacing substantive claims, material dependency breaks,
and an unrelated root. Audit in this order: probability and quantifiers;
target object and relation; root grounding; then material dependency
use-chains.

All supplied definitions are global context available to every proof node.
Definitions need not occur in `sorry_using`; an existing definition entry in
`sorry_using` is harmless and is not a proof-graph edge. Never report a
dependency issue because a definition is absent from `sorry_using`. Mere global
availability gives no semantic credit: a required source object or relation is
grounded only when the root or a root-reachable proposition actually references,
constrains, or relates it.

Apply Lean's metric instances literally while auditing. A product carrier is
allowed, but default `dist`/`norm` on `ℝ × ℝ` or `ℚ × ℚ` is max/sup, not
Euclidean. Report its use for a COT Euclidean length, circle, or angle in
`cot.wrong_relations`, and set an affected root boolean false. A coordinate
square-sum is squared Euclidean distance and cannot silently stand for ordinary
distance; an explicit square root or `EuclideanSpace ℝ (Fin 2)` can be
Euclidean.

Do not report a dependency issue merely because a proof node is outside
`proof_root_closure`. Verification, abandoned derivations, and legitimate side
branches need not support the root. Report one only when material COT content
cannot reach the root. Do not infer intended mathematics from names and do not
duplicate one defect across categories.

Return one JSON object with exactly `cot`, `root`, `dependency_issues`, and
`repair_issues`.
`cot` has exactly `combined_formal_translation`, `missing_clauses`,
`weakened_clauses`, `unbound_objects`, `wrong_relations`, and `added_clauses`.
Each clause issue has exactly `clause`, `node_names`, and `reason`. `root` has
exactly `translation`, `target_object_preserved`, `answer_grounded`, and
`reasons`; reasons must be empty when both booleans are true. Each dependency
issue has exactly `node_name` and `reason`. Use only supplied proof-node names
for dependency issues. Return JSON only, no Markdown or extra keys. Be terse: at most six clause
issues total, and keep each clause and reason under 50 words.
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
For `P Q : α × β`, translate default product `dist`/`norm` as componentwise
max/sup, never Euclidean. A coordinate square-sum is squared Euclidean distance;
only an explicit square root (or EuclideanSpace norm/dist) is Euclidean.

For the `whole_cot_comparator` part, treat your completed node translations as
frozen evidence. A mathematically wrong COT must pass when formalized exactly.
Reject omissions, weakenings, added claims, object replacement, unbound
objects, wrong relation/direction, answer hard-coding, dependency breaks, and
an unrelated root. A boundary equality does not represent an enclosed volume
or interior. Use only supplied node names. The runner recomputes PASS.
When a COT requires Euclidean length, circle, or angle but the formalization
uses product max/sup `dist`/`norm`, report `cot.wrong_relations` and make an
affected root boolean false. A tuple carrier alone is not a defect, and squared
distance is not ordinary distance without a preserved square relation.

Return one JSON object and no Markdown. Its top-level keys must occur exactly
in this order: `formal_decompiler`, then `whole_cot_comparator`. The first value
has exactly `nodes`. The second has exactly `cot`, `root`,
`unreachable_nodes`, `dependency_issues`, and `repair_issues`, using the supplied schemas and
inventories. Keep each node translation under 40 words, the combined formal
translation under 60 words, and each clause or reason under 20 words. Do not
repeat the Lean declarations. Completing valid JSON is more important than
explanation.
"""


REPAIR_ISSUE_CLASSIFICATION_PROMPT = r"""
`repair_issues` is an array of actionable labels. Every item has exactly
`code`, `node_names`, and `reason`; `code` is exactly one of
`answerPreassigned` or `targetCoverageIncomplete`. Use `answerPreassigned`
only when an object that the COT must compute is fixed by a definition to the
claimed answer or to an arbitrary placeholder, making the proof verification,
tautological, or circular. Never use it for a constant supplied by the source
problem. Use `targetCoverageIncomplete` only when the task requires all
solutions, an exact set, an extremum, or exhaustiveness but the formalization
proves only a witness, one-way inclusion, or one candidate. Every repair issue
must be supported by at least one ordinary `cot` defect or a false root verdict;
the repair label does not replace that semantic evidence. Use supplied node
names only. Return an empty array when neither pattern applies.
"""


CANONICAL_WHOLE_COT_COMPARATOR_SYSTEM_PROMPT = r"""You are a strict semantic
translation comparator, not a truth judge. A faithful formalization of a
mathematically wrong source COT must pass. Identifier names and comments carry
no semantic credit. Compare the complete source requirement with the literal
formal meaning and report independent material root causes, not every affected
node and not the same defect under multiple labels.

Use exactly three issue families:
- `semanticMismatch`: source content is omitted, weakened, strengthened,
  quantified incorrectly, attached to the wrong object, uses the wrong
  relation/direction, or fails the requested answer scope.
- `derivationShortcut`: an object that the source must derive is encoded
  upstream so the formal derivation becomes verification, circular, or
  vacuous. Its `shortcut` is exactly
  {"pattern":"preassigned|derivedAssumption|vacuous",
  "object_role":"target|materialIntermediate"}.
- `dependencyBreak`: the required formal propositions exist, but the explicit
  proof dependency graph does not carry a material source use-chain to root.

`derivationShortcut` applies to the target or any materially required
intermediate. Do not use it for source-given constants, source-given
relations, ordinary object definitions, or a claimed answer appearing only in
a theorem conclusion. `preassigned` means a computed object is fixed to its
answer or placeholder by a definition. `derivedAssumption` means a required
derived conclusion is moved into an explicit hypothesis/binder or a relation
that the root assumes instead of derives. A proposition node is itself the
formal representation of a COT claim: never call it `derivedAssumption` merely
because its explicit proof dependency list is empty or its proof body is not
shown. A definition may faithfully encode a source-given functional relation
such as y = 2*x; that is not `preassigned` unless the source must compute y as
an answer or derived intermediate. Source-given constraints may be explicit
root or lemma binders; a conditional theorem deriving the target from those
constraints is faithful and need not separately prove their existence. Only a
conclusion that the COT itself derives, when moved into a hypothesis, is a
`derivedAssumption`. `vacuous`
means a substantive step is replaced by True, reflexivity, an unconstrained
witness, or an equivalent shell.

Definitions are global context and are not proof-graph vertices. A definition
need not appear in `sorry_using`; its availability alone gives no semantic
credit. Do not reject an unreachable proof node unless a material source
use-chain is actually broken. Verification or abandoned side branches may be
irrelevant and never need to reach root merely because the COT mentions a
check. Use `dependencyBreak` only when required propositions are already
represented faithfully, but an explicit proof-to-proof edge needed to carry
them into an otherwise faithful downstream conclusion is absent. Never use it
for a missing edge from a definition, and never emit it when the same root
cause is already a `semanticMismatch` or `derivationShortcut`.

Important binder pattern: if a source asks to compute `u` from stated data and
relations, a formal root of the form `(u) (h : source_constraints u) : u = k`
is faithful when `source_constraints` expresses those data and relations. Do
not demand an existential theorem, do not label `h` a `derivedAssumption`, and
do not require optional verification nodes to support root. If a separate
intermediate theorem incorrectly claims the relation for every `u`, report
that quantifier error alone; do not duplicate it as a shortcut or dependency
break at root. This remains true when the COT obtains the bound equation by
applying a general principle to source data: binding that resulting equation
is the accepted formal interface for the source constraints, while separate
proposition nodes may preserve the explanatory derivation.

Set `source_contract.answer_scope` to exactly one of `single`, `witness`,
`exhaustive`, `extremum`, or `proof`. Then state the source target object and
required relation. Scope describes the whole semantic obligation, not merely
the final output cardinality: use `exhaustive` when every solution or object
must be covered before computing a single aggregate such as a sum. One issue
represents one independent root cause and may
name multiple affected nodes. If fixing issue A would remove apparent issue B,
merge them; if B would remain, emit a separate issue. At most six issues.

Return JSON only with exactly `source_contract`, `formal_root`, and `issues`.
`source_contract` has exactly `answer_scope`, `target_object`, and
`required_relation`. `formal_root` has exactly `translation`. Every issue has
exactly `issue_id`, `family`, `shortcut`, `node_names`, `source_requirement`,
`observed_formal_behavior`, and `reason`. IDs are unique contiguous I1..In.
Use supplied node names only; `semanticMismatch` may use an empty node list
when required source content is wholly absent. `shortcut` is non-null only for
`derivationShortcut`; it is null for the other families. Every text field is
required and at most 50 words. Return an empty `issues` array only for a full
semantic match.

Mandatory scope scan before inspecting the formalization: find source phrases
such as `all solutions`, `every`, `exactly the set`, `only solutions`, or a sum
over `all` intersections/objects. These require exhaustive coverage even when
the requested final answer is one scalar aggregate. A finite hardcoded list
plus proofs that each listed item is valid gives only one-way membership; it
is a `semanticMismatch` unless the formal propositions also say every valid
item belongs to that list. Do not relabel this omission as `preassigned`.

Apply Lean semantics literally. Product `dist`/`norm` on `R x R` is max/sup,
not Euclidean; a coordinate square-sum is squared Euclidean distance. Audit in
this order: source answer scope and quantifiers; target object and relation;
derivation grounding; material dependency use-chain. In reasoning, make one
linear pass: write a short source contract, inspect root and declarations
once, create a provisional root-cause ledger, merge counterfactually dependent
entries once, then emit JSON. Do not restate the full problem, COT,
declaration inventory, or schema. Do not reopen an issue after classifying it
unless a directly conflicting declaration is found. Keep all internal
reasoning under 1200 words; spend tokens on final decisions, not repeated
self-questioning or mathematical truth-checking of the source COT.
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
    global_definition_names: tuple[str, ...]
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [asdict(node) for node in self.nodes],
            "root_name": self.root_name,
            "global_definition_names": list(self.global_definition_names),
            "proof_root_closure": list(self.root_closure),
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
    repair_issues: tuple[dict[str, Any], ...]
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
class CanonicalWholeCotComparatorResult:
    source_contract: dict[str, Any]
    formal_root: dict[str, Any]
    issues: tuple[dict[str, Any], ...]
    passed: bool
    raw_content: str
    reasoning_content: str
    finish_reason: str | None
    request_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    attempts: tuple[dict[str, Any], ...] = ()

    @property
    def unreachable_nodes(self) -> tuple[dict[str, Any], ...]:
        """Compatibility shim for generation warning handling."""
        return ()

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
    node_map = {node.name: node for node in blueprint.nodes}
    proof_names = {
        node.name for node in blueprint.nodes if node.kind != "definition"
    }
    global_definition_names = tuple(
        node.name for node in blueprint.nodes if node.kind == "definition"
    )
    declarations: dict[str, str] = {}
    proof_dependencies: dict[str, tuple[str, ...]] = {}
    for node in blueprint.nodes:
        raw = node.full_declaration() if node.kind == "definition" else node.signature()
        declaration = _strip_lean_comments(raw)
        declarations[node.name] = declaration
        proof_dependencies[node.name] = tuple(
            dep for dep in node.dependencies
            if dep in proof_names and node_map[dep].kind != "definition"
        )
    closure: set[str] = set()
    stack = [blueprint.target_theorem]
    while stack:
        name = stack.pop()
        if name in closure or name not in proof_names:
            continue
        closure.add(name)
        stack.extend(proof_dependencies.get(name, ()))
    views = tuple(FormalNodeView(
        node.name,
        node.kind,
        proof_dependencies[node.name],
        declarations[node.name],
        node.name == blueprint.target_theorem,
        node.kind != "definition" and node.name in closure,
    ) for node in blueprint.nodes)
    payload = json.dumps(
        {
            "root": blueprint.target_theorem,
            "global_definition_names": global_definition_names,
            "proof_root_closure": sorted(closure),
            "nodes": [asdict(node) for node in views],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return FormalView(
        views,
        blueprint.target_theorem,
        tuple(
            node.node_name for node in views
            if node.kind != "definition" and node.in_root_closure
        ),
        global_definition_names,
        hashlib.sha256(payload.encode()).hexdigest(),
    )


def unreachable_proof_node_names(view: FormalView) -> tuple[str, ...]:
    """Return only non-root proof nodes outside the explicit proof closure."""
    return tuple(
        node.node_name for node in view.nodes
        if node.kind != "definition" and not node.is_root and not node.in_root_closure
    )


def formal_decompiler_messages(view: FormalView) -> list[dict[str, str]]:
    inventory = []
    for node in view.nodes:
        item: dict[str, Any] = {
            "node_name": node.node_name,
            "kind": node.kind,
            "is_root": node.is_root,
            "sanitized_formal_lean": node.declaration,
        }
        if node.kind == "definition":
            item["scope"] = "global_definition"
        else:
            item["dependencies"] = list(node.dependencies)
            item["in_root_closure"] = node.in_root_closure
        inventory.append(item)
    return [
        {"role": "system", "content": DECOMPILER_SYSTEM_PROMPT},
        {"role": "user", "content": (
            json.dumps({"root": view.root_name, "nodes": inventory}, ensure_ascii=False)
        )},
    ]


def compact_formal_decompiler_messages(view: FormalView) -> list[dict[str, str]]:
    """Blind decompiler request without cache identity or graph-only fields."""
    inventory = [{
        "node_name": node.node_name,
        "kind": node.kind,
        "is_root": node.is_root,
        "sanitized_formal_lean": node.declaration,
    } for node in view.nodes]
    return [
        {"role": "system", "content": DECOMPILER_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({
            "root": view.root_name, "nodes": inventory,
        }, ensure_ascii=False)},
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
_REPAIR_ISSUE_KEYS = {"code", "node_names", "reason"}
_REPAIR_CODES = {"answerPreassigned", "targetCoverageIncomplete"}


def _parse_repair_issues(
    value: Any,
    *,
    known_nodes: set[str],
    raw: str,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        raise SemanticAuditFormatError("repair_issues must be an array", raw_content=raw)
    parsed = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != _REPAIR_ISSUE_KEYS:
            raise SemanticAuditFormatError(
                f"repair_issues[{index}] has invalid keys", raw_content=raw,
            )
        code = _string(item["code"], "repair issue code", raw)
        if code not in _REPAIR_CODES:
            raise SemanticAuditFormatError(
                f"unsupported repair issue code: {code}", raw_content=raw,
            )
        node_names = _strings(
            item["node_names"], "repair issue node_names", raw,
        )
        if any(node_name not in known_nodes for node_name in node_names):
            raise SemanticAuditFormatError(
                "repair issue references unknown node", raw_content=raw,
            )
        parsed.append({
            "code": code,
            "node_names": list(node_names),
            "reason": _string(item["reason"], "repair issue reason", raw),
        })
    return tuple(parsed)


def _has_ordinary_semantic_defect(
    cot: Mapping[str, Any], root: Mapping[str, Any],
) -> bool:
    return (
        any(cot[key] for key in (
            "missing_clauses", "weakened_clauses", "unbound_objects",
            "wrong_relations", "added_clauses",
        ))
        or not root["target_object_preserved"]
        or not root["answer_grounded"]
        or bool(root["reasons"])
    )


def whole_cot_comparator_messages(
    informal_statement: str,
    informal_proof: str,
    claimed_answer: str,
    view: FormalView,
    decompiler: FormalDecompilerResult,
) -> list[dict[str, str]]:
    unreachable = list(unreachable_proof_node_names(view))
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
            "repair_issues": [{
                "code": "answerPreassigned", "node_names": ["n"],
                "reason": "...",
            }],
        },
    }
    return [
        {"role": "system", "content": (
            WHOLE_COT_COMPARATOR_SYSTEM_PROMPT + REPAIR_ISSUE_CLASSIFICATION_PROMPT
        )},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _compact_graph(view: FormalView) -> dict[str, Any]:
    return {
        "root": view.root_name,
        "global_definition_names": list(view.global_definition_names),
        "proof_dependencies": {
            node.node_name: list(node.dependencies)
            for node in view.nodes if node.kind != "definition"
        },
        "proof_root_closure": list(view.root_closure),
    }


def compact_whole_cot_comparator_messages(
    informal_statement: str,
    informal_proof: str,
    claimed_answer: str,
    view: FormalView,
    decompiler: FormalDecompilerResult,
) -> list[dict[str, str]]:
    payload = {
        "problem": informal_statement,
        "claimed_answer": claimed_answer,
        "complete_original_cot": informal_proof,
        "frozen_node_translations": [asdict(node) for node in decompiler.nodes],
        "graph": _compact_graph(view),
        "required_output_shape": {
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
            "dependency_issues": [{"node_name": view.root_name, "reason": "..."}],
            "repair_issues": [{
                "code": "answerPreassigned", "node_names": [view.root_name],
                "reason": "...",
            }],
        },
    }
    return [
        {"role": "system", "content": (
            COMPACT_WHOLE_COT_COMPARATOR_SYSTEM_PROMPT
            + REPAIR_ISSUE_CLASSIFICATION_PROMPT
        )},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def direct_whole_cot_comparator_messages(
    informal_statement: str,
    informal_proof: str,
    claimed_answer: str,
    view: FormalView,
) -> list[dict[str, str]]:
    formal_view = {
        "root_name": view.root_name,
        "global_definitions": [
            {
                "node_name": node.node_name,
                "kind": node.kind,
                "declaration": node.declaration,
            }
            for node in view.nodes if node.kind == "definition"
        ],
        "proof_nodes": [asdict(node) for node in view.nodes if node.kind != "definition"],
        "proof_root_closure": list(view.root_closure),
    }
    payload = {
        "problem": informal_statement,
        "claimed_answer": claimed_answer,
        "complete_original_cot": informal_proof,
        "formal_view": formal_view,
        "required_output_shape": {
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
            "dependency_issues": [{"node_name": view.root_name, "reason": "..."}],
            "repair_issues": [{
                "code": "answerPreassigned", "node_names": [view.root_name],
                "reason": "...",
            }],
        },
    }
    return [
        {"role": "system", "content": (
            DIRECT_WHOLE_COT_COMPARATOR_SYSTEM_PROMPT
            + REPAIR_ISSUE_CLASSIFICATION_PROMPT
        )},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _canonical_required_output_shape(root_name: str) -> dict[str, Any]:
    return {
        "source_contract": {
            "answer_scope": "single",
            "target_object": "...",
            "required_relation": "...",
        },
        "formal_root": {"translation": "..."},
        "issues": [{
            "issue_id": "I1",
            "family": "semanticMismatch",
            "shortcut": None,
            "node_names": [root_name],
            "source_requirement": "...",
            "observed_formal_behavior": "...",
            "reason": "...",
        }],
    }


def canonical_compact_whole_cot_comparator_messages(
    informal_statement: str,
    informal_proof: str,
    claimed_answer: str,
    view: FormalView,
    decompiler: FormalDecompilerResult,
) -> list[dict[str, str]]:
    payload = {
        "problem": informal_statement,
        "claimed_answer": claimed_answer,
        "complete_original_cot": informal_proof,
        "frozen_node_translations": [asdict(node) for node in decompiler.nodes],
        "graph": _compact_graph(view),
        "required_output_shape": _canonical_required_output_shape(view.root_name),
    }
    return [
        {"role": "system", "content": CANONICAL_WHOLE_COT_COMPARATOR_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def canonical_direct_whole_cot_comparator_messages(
    informal_statement: str,
    informal_proof: str,
    claimed_answer: str,
    view: FormalView,
) -> list[dict[str, str]]:
    formal_view = {
        "root_name": view.root_name,
        "global_definitions": [{
            "node_name": node.node_name,
            "kind": node.kind,
            "declaration": node.declaration,
        } for node in view.nodes if node.kind == "definition"],
        "proof_nodes": [asdict(node) for node in view.nodes if node.kind != "definition"],
        "proof_root_closure": list(view.root_closure),
    }
    payload = {
        "problem": informal_statement,
        "claimed_answer": claimed_answer,
        "complete_original_cot": informal_proof,
        "formal_view": formal_view,
        "required_output_shape": _canonical_required_output_shape(view.root_name),
    }
    return [
        {"role": "system", "content": CANONICAL_WHOLE_COT_COMPARATOR_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


_CANONICAL_SOURCE_CONTRACT_KEYS = {
    "answer_scope", "target_object", "required_relation",
}
_CANONICAL_ISSUE_KEYS = {
    "issue_id", "family", "shortcut", "node_names", "source_requirement",
    "observed_formal_behavior", "reason",
}
_CANONICAL_FAMILIES = {
    "semanticMismatch", "derivationShortcut", "dependencyBreak",
}
_CANONICAL_ANSWER_SCOPES = {"single", "witness", "exhaustive", "extremum", "proof"}
_CANONICAL_SHORTCUT_PATTERNS = {"preassigned", "derivedAssumption", "vacuous"}
_CANONICAL_OBJECT_ROLES = {"target", "materialIntermediate"}


def _canonical_text(value: Any, label: str, raw: str) -> str:
    text = _string(value, label, raw)
    if len(text.split()) > 50:
        raise SemanticAuditFormatError(
            f"{label} must contain at most 50 words", raw_content=raw,
        )
    return text


def parse_canonical_whole_cot_comparator(
    content: str,
    *,
    view: FormalView,
    decompiler: FormalDecompilerResult | None = None,
) -> tuple[dict[str, Any], dict[str, Any], tuple[dict[str, Any], ...], bool]:
    value = _json_object(content)
    if set(value) != {"source_contract", "formal_root", "issues"}:
        raise SemanticAuditFormatError(
            "canonical comparator top-level keys must be exactly source_contract, "
            "formal_root, and issues", raw_content=content,
        )
    source = value["source_contract"]
    if not isinstance(source, dict) or set(source) != _CANONICAL_SOURCE_CONTRACT_KEYS:
        raise SemanticAuditFormatError("source_contract has invalid keys", raw_content=content)
    answer_scope = _string(source["answer_scope"], "source_contract.answer_scope", content)
    if answer_scope not in _CANONICAL_ANSWER_SCOPES:
        raise SemanticAuditFormatError("invalid answer_scope", raw_content=content)
    parsed_source = {
        "answer_scope": answer_scope,
        "target_object": _canonical_text(
            source["target_object"], "source_contract.target_object", content,
        ),
        "required_relation": _canonical_text(
            source["required_relation"], "source_contract.required_relation", content,
        ),
    }
    formal_root = value["formal_root"]
    if not isinstance(formal_root, dict) or set(formal_root) != {"translation"}:
        raise SemanticAuditFormatError("formal_root has invalid keys", raw_content=content)
    parsed_root = {
        "translation": _canonical_text(
            formal_root["translation"], "formal_root.translation", content,
        ),
    }
    raw_issues = value["issues"]
    if not isinstance(raw_issues, list):
        raise SemanticAuditFormatError("issues must be an array", raw_content=content)
    if len(raw_issues) > 6:
        raise SemanticAuditFormatError("canonical comparator returned more than six issues", raw_content=content)
    known_nodes = {node.node_name for node in view.nodes}
    parsed_issues: list[dict[str, Any]] = []
    for index, item in enumerate(raw_issues, start=1):
        if not isinstance(item, dict) or set(item) != _CANONICAL_ISSUE_KEYS:
            raise SemanticAuditFormatError(f"issues[{index - 1}] has invalid keys", raw_content=content)
        issue_id = _string(item["issue_id"], "issue_id", content)
        if issue_id != f"I{index}":
            raise SemanticAuditFormatError("issue IDs must be contiguous I1..In", raw_content=content)
        family = _string(item["family"], "issue family", content)
        if family not in _CANONICAL_FAMILIES:
            raise SemanticAuditFormatError("invalid issue family", raw_content=content)
        node_names = _strings(item["node_names"], "issue node_names", content)
        if len(set(node_names)) != len(node_names) or any(
            node_name not in known_nodes for node_name in node_names
        ):
            raise SemanticAuditFormatError("issue contains duplicate or unknown node", raw_content=content)
        if family != "semanticMismatch" and not node_names:
            raise SemanticAuditFormatError(
                f"{family} must name at least one node", raw_content=content,
            )
        shortcut = item["shortcut"]
        parsed_shortcut = None
        if family == "derivationShortcut":
            if not isinstance(shortcut, dict) or set(shortcut) != {"pattern", "object_role"}:
                raise SemanticAuditFormatError("derivationShortcut requires shortcut", raw_content=content)
            pattern = _string(shortcut["pattern"], "shortcut.pattern", content)
            object_role = _string(shortcut["object_role"], "shortcut.object_role", content)
            if pattern not in _CANONICAL_SHORTCUT_PATTERNS or object_role not in _CANONICAL_OBJECT_ROLES:
                raise SemanticAuditFormatError("invalid shortcut pattern or object_role", raw_content=content)
            parsed_shortcut = {"pattern": pattern, "object_role": object_role}
        elif shortcut is not None:
            raise SemanticAuditFormatError(
                "shortcut must be null outside derivationShortcut", raw_content=content,
            )
        parsed_issues.append({
            "issue_id": issue_id,
            "family": family,
            "shortcut": parsed_shortcut,
            "node_names": list(node_names),
            "source_requirement": _canonical_text(
                item["source_requirement"], "issue.source_requirement", content,
            ),
            "observed_formal_behavior": _canonical_text(
                item["observed_formal_behavior"], "issue.observed_formal_behavior", content,
            ),
            "reason": _canonical_text(item["reason"], "issue.reason", content),
        })
    passed = not parsed_issues and (decompiler is None or not decompiler.vacuous_nodes)
    return parsed_source, parsed_root, tuple(parsed_issues), passed


def parse_compact_whole_cot_comparator(
    content: str,
    *,
    view: FormalView,
    decompiler: FormalDecompilerResult | None = None,
) -> tuple[
    dict[str, Any], dict[str, Any], tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], bool,
]:
    value = _json_object(content)
    if set(value) != {"cot", "root", "dependency_issues", "repair_issues"}:
        raise SemanticAuditFormatError(
            "invalid compact comparator top-level keys", raw_content=content,
        )
    known_nodes = {node.node_name for node in view.nodes}
    proof_nodes = {
        node.node_name for node in view.nodes if node.kind != "definition"
    }
    cot = value["cot"]
    if not isinstance(cot, dict) or set(cot) != _WHOLE_COT_KEYS:
        raise SemanticAuditFormatError("cot has invalid keys", raw_content=content)
    parsed_cot: dict[str, Any] = {
        "combined_formal_translation": _string(
            cot["combined_formal_translation"],
            "cot.combined_formal_translation", content,
        ),
    }
    issue_count = 0
    for key in (
        "missing_clauses", "weakened_clauses", "unbound_objects",
        "wrong_relations", "added_clauses",
    ):
        parsed_cot[key] = _parse_clause_issues(
            cot[key], label=f"cot.{key}", known_nodes=known_nodes, raw=content,
        )
        issue_count += len(parsed_cot[key])
    if issue_count > 6:
        raise SemanticAuditFormatError(
            "compact comparator returned more than six clause issues",
            raw_content=content,
        )

    root = value["root"]
    if not isinstance(root, dict) or set(root) != {
        "translation", "target_object_preserved", "answer_grounded", "reasons",
    }:
        raise SemanticAuditFormatError("root has invalid keys", raw_content=content)
    if not isinstance(root["target_object_preserved"], bool) or not isinstance(
        root["answer_grounded"], bool,
    ):
        raise SemanticAuditFormatError("root verdicts must be booleans", raw_content=content)
    parsed_root = {
        "translation": _string(root["translation"], "root.translation", content),
        "target_object_preserved": root["target_object_preserved"],
        "answer_grounded": root["answer_grounded"],
        "reasons": list(_strings(root["reasons"], "root.reasons", content)),
    }

    raw_dependencies = value["dependency_issues"]
    if not isinstance(raw_dependencies, list):
        raise SemanticAuditFormatError("dependency_issues must be an array", raw_content=content)
    dependencies = []
    for item in raw_dependencies:
        if not isinstance(item, dict) or set(item) != {"node_name", "reason"}:
            raise SemanticAuditFormatError("dependency issue has invalid keys", raw_content=content)
        node_name = _string(item["node_name"], "dependency node", content)
        if node_name in known_nodes and node_name not in proof_nodes:
            # A dependency defect is a broken material path into the proof
            # graph.  Comparators sometimes attach that defect to the global
            # definition whose semantics failed to reach the proof spine.
            # Definitions are global context rather than graph vertices, so
            # retain the rejecting issue but anchor it at the root proof node.
            node_name = view.root_name
        if node_name not in proof_nodes:
            raise SemanticAuditFormatError(
                "dependency issue must name a proof node", raw_content=content,
            )
        dependencies.append({
            "node_name": node_name,
            "reason": _string(item["reason"], "dependency reason", content),
        })

    repair_issues = _parse_repair_issues(
        value["repair_issues"], known_nodes=known_nodes, raw=content,
    )
    if repair_issues and not _has_ordinary_semantic_defect(parsed_cot, parsed_root):
        raise SemanticAuditFormatError(
            "repair issue requires an ordinary cot/root defect", raw_content=content,
        )

    passed = (
        not any(parsed_cot[key] for key in (
            "missing_clauses", "weakened_clauses", "unbound_objects",
            "wrong_relations", "added_clauses",
        ))
        and parsed_root["target_object_preserved"]
        and parsed_root["answer_grounded"]
        and not parsed_root["reasons"]
        and not dependencies
        and not repair_issues
        and (decompiler is None or not decompiler.vacuous_nodes)
    )
    return parsed_cot, parsed_root, (), tuple(dependencies), repair_issues, passed


def parse_whole_cot_comparator(
    content: str,
    *,
    view: FormalView,
    decompiler: FormalDecompilerResult,
) -> tuple[
    dict[str, Any], dict[str, Any], tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], bool,
]:
    value = _json_object(content)
    if set(value) != {
        "cot", "root", "unreachable_nodes", "dependency_issues", "repair_issues",
    }:
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

    expected_unreachable = list(unreachable_proof_node_names(view))
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

    repair_issues = _parse_repair_issues(
        value["repair_issues"], known_nodes=known_nodes, raw=content,
    )
    if repair_issues and not _has_ordinary_semantic_defect(parsed_cot, parsed_root):
        raise SemanticAuditFormatError(
            "repair issue requires an ordinary cot/root defect", raw_content=content,
        )

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
        and not repair_issues
        and not decompiler.vacuous_nodes
    )
    return (
        parsed_cot, parsed_root, tuple(unreachable), tuple(dependencies),
        repair_issues, passed,
    )


def joint_whole_cot_audit_messages(
    informal_statement: str,
    informal_proof: str,
    claimed_answer: str,
    view: FormalView,
) -> list[dict[str, str]]:
    unreachable = list(unreachable_proof_node_names(view))
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
                "repair_issues": [{
                    "code": "answerPreassigned", "node_names": ["n"],
                    "reason": "...",
                }],
            },
        },
    }
    return [
        {"role": "system", "content": (
            JOINT_WHOLE_COT_SYSTEM_PROMPT + REPAIR_ISSUE_CLASSIFICATION_PROMPT
        )},
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
    cot, root, unreachable, dependencies, repair_issues, passed = parse_whole_cot_comparator(
        json.dumps(value["whole_cot_comparator"], ensure_ascii=False),
        view=view,
        decompiler=decompiler,
    )
    return translations, cot, root, unreachable, dependencies, repair_issues, passed


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
                    "Top-level keys are exactly cot, root, unreachable_nodes, "
                    "dependency_issues, and repair_issues. Every COT clause issue is exactly "
                    "{clause,node_names,reason}; copy the required unreachable node "
                    "inventory exactly and use node names only."
                )
            elif operation in {
                "compact_whole_cot_comparator", "direct_whole_cot_comparator",
            }:
                schema_guidance = (
                    "Top-level keys are exactly cot, root, dependency_issues, and repair_issues. "
                    "Every COT clause issue is exactly {clause,node_names,reason}; "
                    "dependency_issues.node_name must be a proof-node name (a key "
                    "of graph.proof_dependencies for compact input); use the root "
                    "proof node for a material chain that fails to reach the root. "
                    "Use supplied node names only and return at most six clause issues."
                )
            elif operation in {
                "canonical_compact_whole_cot_comparator",
                "canonical_direct_whole_cot_comparator",
            }:
                schema_guidance = (
                    "Top-level keys are exactly source_contract, formal_root, and issues. "
                    "source_contract is exactly {answer_scope,target_object,required_relation}; "
                    "formal_root is exactly {translation}. Every issue is exactly "
                    "{issue_id,family,shortcut,node_names,source_requirement," 
                    "observed_formal_behavior,reason}; IDs must be contiguous I1..In and "
                    "there may be at most six issues. shortcut is null except for "
                    "derivationShortcut, where it is exactly {pattern,object_role}."
                )
            elif operation == "joint_whole_cot_audit":
                schema_guidance = (
                    "Top-level keys must occur exactly in this order: "
                    "formal_decompiler, whole_cot_comparator. The first has "
                    "exactly nodes in the supplied order. The second has exactly "
                    "cot, root, unreachable_nodes, dependency_issues, repair_issues. Clause "
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
    compact: bool = False,
) -> FormalDecompilerResult:
    if tracer is not None:
        tracer.emit(TraceEvent(
            kind="formalDecompileStart", thm_name=thm_name, turn=round_index,
            args={"round": round_index, "formalViewHash": view.sha256,
                  "nodeCount": len(view.nodes),
                  "protocol": FORMAL_DECOMPILER_PROMPT_VERSION},
        ))
    parsed, content, reasoning, finish, request_id, attempts, usage = _run_stage(
        client, model, messages=(
            compact_formal_decompiler_messages(view)
            if compact else formal_decompiler_messages(view)
        ),
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
                  "protocol": FORMAL_DECOMPILER_PROMPT_VERSION,
                  "vacuousNodes": list(result.vacuous_nodes), "result": result.to_dict()},
            ok=not bool(result.vacuous_nodes),
        ))
        tracer.emit(TraceEvent(
            kind="formalDecompileEnd", thm_name=thm_name, turn=round_index,
            args={"round": round_index, "attemptCount": len(attempts),
                  "protocol": FORMAL_DECOMPILER_PROMPT_VERSION,
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
    cot, root, unreachable, dependencies, repair_issues, passed = parsed
    result = WholeCotComparatorResult(
        cot, root, unreachable, dependencies, repair_issues, passed,
        content, reasoning, finish,
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


def run_compact_whole_cot_comparator(
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
    messages = compact_whole_cot_comparator_messages(
        informal_statement, informal_proof, claimed_answer, view, decompiler,
    )
    cache_key = semantic_audit_cache_key(
        model, messages, version=COMPACT_WHOLE_COT_PROMPT_VERSION,
    )
    if tracer is not None:
        tracer.emit(TraceEvent(
            kind="wholeCotCompareStart", thm_name=thm_name, turn=round_index,
            args={"round": round_index, "formalViewHash": view.sha256,
                  "cacheKey": cache_key, "protocol": COMPACT_WHOLE_COT_PROMPT_VERSION},
        ))
    parsed, content, reasoning, finish, request_id, attempts, usage = _run_stage(
        client, model, messages=messages, parser=parse_compact_whole_cot_comparator,
        parser_kwargs={"view": view, "decompiler": decompiler},
        max_tokens=max_tokens, max_attempts=max_attempts, tracer=tracer,
        thm_name=thm_name, round_index=round_index,
        phase="wholeCotComparator", operation="compact_whole_cot_comparator",
        enable_thinking=enable_thinking, temperature=temperature,
        top_p=top_p, top_k=top_k, min_p=min_p,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
    )
    cot, root, unreachable, dependencies, repair_issues, passed = parsed
    result = WholeCotComparatorResult(
        cot, root, unreachable, dependencies, repair_issues, passed,
        content, reasoning, finish,
        request_id, usage[0], usage[1], usage[2], attempts,
    )
    if tracer is not None:
        tracer.emit(TraceEvent(
            kind="wholeCotCompareResult", thm_name=thm_name, turn=round_index,
            args={"round": round_index, "formalViewHash": view.sha256,
                  "cacheKey": cache_key, "protocol": COMPACT_WHOLE_COT_PROMPT_VERSION,
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


def run_direct_whole_cot_comparator(
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
) -> WholeCotComparatorResult:
    messages = direct_whole_cot_comparator_messages(
        informal_statement, informal_proof, claimed_answer, view,
    )
    cache_key = semantic_audit_cache_key(
        model, messages, version=DIRECT_WHOLE_COT_PROMPT_VERSION,
    )
    if tracer is not None:
        tracer.emit(TraceEvent(
            kind="wholeCotCompareStart", thm_name=thm_name, turn=round_index,
            args={"round": round_index, "formalViewHash": view.sha256,
                  "cacheKey": cache_key, "protocol": DIRECT_WHOLE_COT_PROMPT_VERSION},
        ))
    parsed, content, reasoning, finish, request_id, attempts, usage = _run_stage(
        client, model, messages=messages, parser=parse_compact_whole_cot_comparator,
        parser_kwargs={"view": view, "decompiler": None},
        max_tokens=max_tokens, max_attempts=max_attempts, tracer=tracer,
        thm_name=thm_name, round_index=round_index,
        phase="wholeCotComparator", operation="direct_whole_cot_comparator",
        enable_thinking=enable_thinking, temperature=temperature,
        top_p=top_p, top_k=top_k, min_p=min_p,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
    )
    cot, root, unreachable, dependencies, repair_issues, passed = parsed
    result = WholeCotComparatorResult(
        cot, root, unreachable, dependencies, repair_issues, passed,
        content, reasoning, finish,
        request_id, usage[0], usage[1], usage[2], attempts,
    )
    if tracer is not None:
        tracer.emit(TraceEvent(
            kind="wholeCotCompareResult", thm_name=thm_name, turn=round_index,
            args={"round": round_index, "formalViewHash": view.sha256,
                  "cacheKey": cache_key, "protocol": DIRECT_WHOLE_COT_PROMPT_VERSION,
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


def _run_canonical_whole_cot_comparator(
    client: Any,
    model: str,
    *,
    messages: list[dict[str, str]],
    protocol: str,
    operation: str,
    view: FormalView,
    decompiler: FormalDecompilerResult | None,
    max_tokens: int,
    max_attempts: int,
    enable_thinking: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    presence_penalty: float,
    repetition_penalty: float,
    tracer: Any,
    thm_name: str,
    round_index: int,
) -> CanonicalWholeCotComparatorResult:
    cache_key = semantic_audit_cache_key(model, messages, version=protocol)
    if tracer is not None:
        tracer.emit(TraceEvent(
            kind="wholeCotCompareStart", thm_name=thm_name, turn=round_index,
            args={
                "round": round_index, "formalViewHash": view.sha256,
                "cacheKey": cache_key, "protocol": protocol,
            },
        ))
    parsed, content, reasoning, finish, request_id, attempts, usage = _run_stage(
        client, model, messages=messages, parser=parse_canonical_whole_cot_comparator,
        parser_kwargs={"view": view, "decompiler": decompiler},
        max_tokens=max_tokens, max_attempts=max_attempts, tracer=tracer,
        thm_name=thm_name, round_index=round_index,
        phase="wholeCotComparator", operation=operation,
        enable_thinking=enable_thinking, temperature=temperature,
        top_p=top_p, top_k=top_k, min_p=min_p,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
    )
    source_contract, formal_root, issues, passed = parsed
    result = CanonicalWholeCotComparatorResult(
        source_contract, formal_root, issues, passed, content, reasoning, finish,
        request_id, usage[0], usage[1], usage[2], attempts,
    )
    if tracer is not None:
        tracer.emit(TraceEvent(
            kind="wholeCotCompareResult", thm_name=thm_name, turn=round_index,
            args={
                "round": round_index, "formalViewHash": view.sha256,
                "cacheKey": cache_key, "protocol": protocol,
                "passed": passed, "sourceContract": source_contract,
                "canonicalIssues": list(issues), "result": result.to_dict(),
            }, ok=passed,
        ))
        tracer.emit(TraceEvent(
            kind="wholeCotCompareEnd", thm_name=thm_name, turn=round_index,
            args={
                "round": round_index, "passed": passed,
                "attemptCount": len(attempts), "promptTokens": usage[0],
                "completionTokens": usage[1], "totalTokens": usage[2],
                "requestId": request_id,
            }, ok=passed,
        ))
    return result


def run_canonical_compact_whole_cot_comparator(
    client: Any, model: str, *, informal_statement: str, informal_proof: str,
    claimed_answer: str, view: FormalView, decompiler: FormalDecompilerResult,
    max_tokens: int, max_attempts: int, enable_thinking: bool = False,
    temperature: float = 0.0, top_p: float = 1.0, top_k: int = -1,
    min_p: float = 0.0, presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0, tracer=None, thm_name: str = "",
    round_index: int = 0,
) -> CanonicalWholeCotComparatorResult:
    return _run_canonical_whole_cot_comparator(
        client, model,
        messages=canonical_compact_whole_cot_comparator_messages(
            informal_statement, informal_proof, claimed_answer, view, decompiler,
        ),
        protocol=CANONICAL_COMPACT_WHOLE_COT_PROMPT_VERSION,
        operation="canonical_compact_whole_cot_comparator", view=view,
        decompiler=decompiler, max_tokens=max_tokens, max_attempts=max_attempts,
        enable_thinking=enable_thinking, temperature=temperature, top_p=top_p,
        top_k=top_k, min_p=min_p, presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty, tracer=tracer,
        thm_name=thm_name, round_index=round_index,
    )


def run_canonical_direct_whole_cot_comparator(
    client: Any, model: str, *, informal_statement: str, informal_proof: str,
    claimed_answer: str, view: FormalView, max_tokens: int, max_attempts: int,
    enable_thinking: bool = False, temperature: float = 0.0,
    top_p: float = 1.0, top_k: int = -1, min_p: float = 0.0,
    presence_penalty: float = 0.0, repetition_penalty: float = 1.0,
    tracer=None, thm_name: str = "", round_index: int = 0,
) -> CanonicalWholeCotComparatorResult:
    return _run_canonical_whole_cot_comparator(
        client, model,
        messages=canonical_direct_whole_cot_comparator_messages(
            informal_statement, informal_proof, claimed_answer, view,
        ),
        protocol=CANONICAL_DIRECT_WHOLE_COT_PROMPT_VERSION,
        operation="canonical_direct_whole_cot_comparator", view=view,
        decompiler=None, max_tokens=max_tokens, max_attempts=max_attempts,
        enable_thinking=enable_thinking, temperature=temperature, top_p=top_p,
        top_k=top_k, min_p=min_p, presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty, tracer=tracer,
        thm_name=thm_name, round_index=round_index,
    )


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
    translations, cot, root, unreachable, dependencies, repair_issues, passed = parsed
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
        cot, root, unreachable, dependencies, repair_issues, passed,
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


def whole_cot_comparator_defects(
    result: WholeCotComparatorResult | CanonicalWholeCotComparatorResult,
) -> list[dict[str, Any]]:
    if isinstance(result, CanonicalWholeCotComparatorResult):
        return [{
            "category": item["family"],
            "node_names": list(item["node_names"]),
            "requirement": item["source_requirement"],
            "observed_formal_behavior": item["observed_formal_behavior"],
            "reason": item["reason"],
            "issue_id": item["issue_id"],
            "shortcut": item["shortcut"],
        } for item in result.issues]
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
    repair_requirements = {
        "answerPreassigned": (
            "Bind the computed object and derive the claimed answer from source constraints."
        ),
        "targetCoverageIncomplete": (
            "Formalize the complete target set, extremum, or exhaustiveness claim."
        ),
    }
    for item in result.repair_issues:
        defects.append({
            "category": item["code"],
            "node_names": list(item["node_names"]),
            "requirement": repair_requirements[item["code"]],
            "reason": item["reason"],
        })
    return defects



__all__ = [
    "FORMAL_DECOMPILER_PROMPT_VERSION", "WHOLE_COT_PROMPT_VERSION",
    "JOINT_WHOLE_COT_PROMPT_VERSION",
    "JOINT_SEMANTIC_EFFECT_ALIASES", "JOINT_WHOLE_COT_SYSTEM_PROMPT",
    "COMPACT_WHOLE_COT_PROMPT_VERSION", "DIRECT_WHOLE_COT_PROMPT_VERSION",
    "CANONICAL_COMPACT_WHOLE_COT_PROMPT_VERSION",
    "CANONICAL_DIRECT_WHOLE_COT_PROMPT_VERSION",
    "CANONICAL_WHOLE_COT_COMPARATOR_SYSTEM_PROMPT",
    "CanonicalWholeCotComparatorResult",
    "FormalDecompilerResult", "FormalView", "JointWholeCotAuditResult",
    "SemanticAuditFormatError", "WholeCotComparatorResult",
    "build_formal_view",
    "compact_formal_decompiler_messages", "formal_decompiler_messages",
    "parse_formal_decompiler", "run_formal_decompiler",
    "unreachable_proof_node_names",
    "compact_whole_cot_comparator_messages",
    "direct_whole_cot_comparator_messages",
    "canonical_compact_whole_cot_comparator_messages",
    "canonical_direct_whole_cot_comparator_messages",
    "parse_canonical_whole_cot_comparator",
    "parse_compact_whole_cot_comparator",
    "run_compact_whole_cot_comparator", "run_direct_whole_cot_comparator",
    "run_canonical_compact_whole_cot_comparator",
    "run_canonical_direct_whole_cot_comparator",
    "parse_whole_cot_comparator", "run_whole_cot_comparator",
    "joint_whole_cot_audit_messages", "parse_joint_whole_cot_audit",
    "run_joint_whole_cot_audit",
    "semantic_audit_cache_key", "strict_comparator_messages",
    "whole_cot_comparator_defects", "whole_cot_comparator_messages",
]

"""Phase 1: Blueprint generation.

Calls the LLM with the verbatim system prompt from the paper (prompts/blueprint_system.md)
and validates the resulting @[blueprint]-annotated Lean file via LeanArchitect.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import lru_cache
import hashlib
import json
import os
import re
from pathlib import Path

from blueprint_text import (
    BLUEPRINT_DECL_KW as _BLUEPRINT_DECL_KW,
    BLUEPRINT_PROOF_RE,
    extract_current_node_decl,
    extract_blueprint_signature,
    lemma_to_theorem,
    proof_body_to_decl_suffix,
    strip_blueprint_attr,
)
from kimina_lean_compiler import (
    CompileRequest,
    CompilerResult,
    KiminaInfrastructureError,
    KiminaLeanCompiler,
    MATHLIB_HEADER,
)
from llm_client import chat_completion_with_retry, make_client
from semantic_audit import SemanticAuditFormatError, run_semantic_audit
from semantic_fidelity import (
    SemanticIssue,
    format_semantic_issues,
    parse_cot_manifest,
    semantic_audit_risk_reasons,
    validate_blueprint_fidelity,
)
from goedel_prompts import load, render
from tracer import TraceEvent
from cot_blueprint_refine.claim_scope_manifest import decode_claim_scope_manifest


def _reasoning_kwargs(model: str) -> dict:
    """Return reasoning_effort kwarg for models that support it (gpt-5.x series)."""
    if model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3") or model.startswith("o4"):
        return {"reasoning_effort": "low"}
    return {}

BLUEPRINT_SYSTEM_PROMPT = load("blueprint_system")
BLUEPRINT_USER_TEMPLATE = load("blueprint_user")
ROBUSTPA_BLUEPRINT_SYSTEM_PROMPT = load("robustpa_blueprint_system")
ROBUSTPA_BLUEPRINT_USER_TEMPLATE = load("robustpa_blueprint_user")


_SEMANTIC_BLUEPRINT_SYSTEM_SUFFIX = r"""

## Immutable COT Claim/Scope translation contract

For this request the informal proof is divided into immutable source claims and shared scopes.
Translate every supplied claim, including a wrong
or unsupported claim; never delete, weaken, repair, or replace it merely to
make Lean accept the graph.

Every `@[blueprint]` node, including definitions, MUST contain exactly one
native LeanArchitect title binding naming one supplied claim, of this form:

    (title := "COT_CLAIM:C002")

Use only one of the supplied `COT_CLAIM` identifiers. A claim may have several
nodes, but a node has one primary source claim. The root theorem must bind to a
final supplied claim, and every supplied `COT_CLAIM` must occur on the root's
dependency closure. `[COT_CONTEXT]` is narration/layout only and gets no node.
Claim order records provenance, not a forced dependency chain: infer the
actual logical dependencies from the COT and never invent edges merely to
make claim numbers monotone.
For legacy manifests, Step identifiers record provenance, not a forced
chronology; the same dependency rule applies.

`COT_SCOPE` blocks are exact source prefixes, section labels, or case
conditions shared by the listed `applies_to` claims.  Read a scope and each
target claim as one semantic unit.  The scope does not receive its own theorem
node, but every mathematical object, qualifier, quantifier, definition,
assertion, or branch condition in it MUST appear in the target proposition or
its faithful dependencies.  Do not blindly turn every scope into a new
hypothesis: preserve whether the source presents it as setup, notation,
derived content, or a case assumption.  In particular, never translate a case
body while dropping the condition printed in its `COT_SCOPE` block.
An inclusive compact range such as `C002..C006` means every consecutively
numbered claim from `C002` through `C006`.

Do not encode an asserted or derived step as an executable definition.  Do not
replace a claim with `True`, a reflexive equality, an unconstrained existential
witness, a constant `Prop := True`/`Bool := true`, or a nullary definition that
hard-codes the claimed answer.  If a source step is false or has a gap, keep
its actual proposition as a lemma with `sorry_using [...]`; proving or
diagnosing that lemma belongs to the later phase.

Only the formal Lean type and definition body count as semantic coverage.
Comments, docstrings, and natural-language `statement` fields do not encode a
mathematical constraint.  Put every critical source object, assumption,
quantifier and its polarity, candidate branch, case split, and filtering
condition in the formal proposition itself.  A derived equation must be a
conclusion reached from its source dependencies, never a fresh premise that is
assumed by the node which is supposed to derive it.

When the COT makes a wrong or unsupported jump, state that exact jump while
keeping the same formal quantity connected across the adjacent nodes.  Do not
replace it with disconnected local definitions or a new quantity that merely
reuses the same informal name.  Introduce a coordinate choice or normalization
only at the source step that introduces it, or in a descendant of that step;
never hard-code its consequences into an earlier setup node or root premise.

When a Mathlib-native encoding would be disproportionately elaborate, use a
typed abstract relational model instead of weakening the claim.  Introduce
explicit typed binders for the source objects, relation or function binders for
the operations used by the COT, and hypotheses for exactly the source givens.
Reuse those same formal objects in every later step and in the root.  This is
preferable to `True`, comments standing in for constraints, arbitrary concrete
coordinates, or unrelated existential witnesses.  Instantiate coordinates
only when the source COT itself does so.

Coverage is clause-level, not merely paragraph-level.  If a source step counts
a restricted family and then asserts that the original problem's total `N`
equals that count `K`, define the original `N` once and emit the exact bridge
`lemma cot_total_jump : N = K := by sorry_using [...]`.  Do not replace it by
the weaker statement `restrictedCount = K`, and do not invent a converse or
set equivalence to justify it.  The explicit bridge is how an unsupported COT
jump is preserved for later diagnosis.  Before returning, check every
mathematical clause in each numbered step against a formal type or definition
body; prose in `statement`/`proof` fields does not satisfy this check.

Do not merge distinct clauses by strengthening one of them. For example, a
source claim `3^7 ∣ x^3 ↔ 27 ∣ x` about one variable and a separate claim
`(27 ∣ a ∧ 27 ∣ b ∧ 27 ∣ c) → 3^7 ∣ a^3+b^3+c^3` require two propositions.
The stronger aggregate statement
`3^7 ∣ a^3+b^3+c^3 ↔ (27 ∣ a ∧ 27 ∣ b ∧ 27 ∣ c)` is not a faithful substitute
because it drops the single-variable claim and invents an aggregate converse.

Before emitting Lean, perform an internal step-by-step coverage check.  For
every computation, derived claim, verification, or conclusion step, include a
Lemma/Theorem anchor whose proposition states the source claim itself; helper
Definitions do not replace that anchor.  Reserve Definitions for objects,
functions, sets, and notation actually introduced by the source.  A quantity
computed by the COT must be expressed as an equality proposition, not merely
defined to equal the result.

Keep every claimed quantity tied to the original problem object and givens.
Never make a geometric or counting conclusion easy by choosing arbitrary
coordinates, adding inconsistent assumptions, or replacing a determined
quantity by `∃ (x : T), x = c`.  In particular, the root must state the answer
about the modeled object under the original conditions, rather than only
asserting that an unrelated witness with the answer value exists.
"""


def _decode_manifest_rows(value: str) -> list[dict] | dict:
    if not value:
        return []
    try:
        rows = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid cot_manifest_json: {exc}") from exc
    if isinstance(rows, list):
        return [dict(row) for row in rows]
    if isinstance(rows, dict):
        return dict(rows)
    raise ValueError("cot_manifest_json must contain a JSON list or object")


def _compact_scope_targets(values: list[str]) -> str:
    if len(values) < 2:
        return values[0] if values else "none"
    compact_claims = [re.fullmatch(r"C(\d{3,})", value) for value in values]
    if all(compact_claims):
        ordinals = [int(match.group(1)) for match in compact_claims if match is not None]
        if ordinals == list(range(ordinals[0], ordinals[0] + len(ordinals))):
            return f"{values[0]}..C{ordinals[-1]:03d}"
    parsed = [re.fullmatch(r"(S\d{3})\.C(\d{3,})", value) for value in values]
    if all(parsed):
        step_ids = [match.group(1) for match in parsed if match is not None]
        ordinals = [int(match.group(2)) for match in parsed if match is not None]
        if (
            len(set(step_ids)) == 1
            and ordinals == list(range(ordinals[0], ordinals[0] + len(ordinals)))
        ):
            return f"{values[0]}..C{ordinals[-1]:03d}"
    return ",".join(values)


def _render_claim_scope_proof(manifest_value: dict) -> str:
    manifest = decode_claim_scope_manifest(manifest_value)
    source = str(manifest["source_text"])
    semantic = sorted([
        (int(claim["source_start"]), int(claim["source_end"]), "claim", claim)
        for claim in manifest["claims"]
    ] + [
        (int(scope["source_start"]), int(scope["source_end"]), "scope", scope)
        for scope in manifest["scopes"]
    ])
    blocks: list[str] = []
    cursor = 0
    for start, end, kind, row in semantic:
        if cursor < start:
            blocks.append(f"[COT_CONTEXT]\n{source[cursor:start]}\n[/COT_CONTEXT]")
        if kind == "claim":
            claim_id = str(row["claim_id"])
            scope_ids = ",".join(str(value) for value in row.get("scope_ids", [])) or "none"
            blocks.append(
                f"[COT_CLAIM {claim_id} scopes={scope_ids}]\n"
                f"{source[start:end]}\n[/COT_CLAIM {claim_id}]"
            )
        else:
            scope_id = str(row["scope_id"])
            targets = _compact_scope_targets([
                str(value) for value in row["applies_to_claim_ids"]
            ])
            blocks.append(
                f"[COT_SCOPE {scope_id} type={row['scope_type']} applies_to={targets}]\n"
                f"{source[start:end]}\n[/COT_SCOPE {scope_id}]"
            )
        cursor = end
    if cursor < len(source):
        blocks.append(f"[COT_CONTEXT]\n{source[cursor:]}\n[/COT_CONTEXT]")
    return "\n\n".join(blocks)


def _render_step_grounded_proof(cot_manifest_json: str, *, include_ir: bool) -> str:
    decoded = _decode_manifest_rows(cot_manifest_json)
    if isinstance(decoded, dict):
        return _render_claim_scope_proof(decoded)
    blocks: list[str] = []
    for row in decoded:
        step_id = str(row.get("step_id") or "")
        role = str(row.get("role") or "derived_claim")
        dependencies = ",".join(str(item) for item in (row.get("depends_on") or [])) or "none"
        requires_formalization = bool(row.get("requires_formalization", True))
        stored_claims = [
            dict(claim) for claim in (row.get("claims") or [])
            if isinstance(claim, dict)
        ]
        claim_ids_by_hash: dict[str, list[str]] = {}
        for claim in stored_claims:
            claim_hash = str(claim.get("source_sha256") or "")
            claim_id = str(claim.get("claim_id") or "")
            if claim_hash and claim_id:
                claim_ids_by_hash.setdefault(claim_hash, []).append(claim_id)
        claim_hash_uses: Counter[str] = Counter()
        metadata = (
            f"role={role} depends_on={dependencies} "
            f"requires_formalization={str(requires_formalization).lower()} "
            f"claim_count={len(stored_claims)}"
        )
        if include_ir:
            numbers = ",".join(str(item) for item in (row.get("numbers") or [])) or "none"
            relations = ",".join(str(item) for item in (row.get("relations") or [])) or "none"
            metadata += f" numbers={numbers} relations={relations}"
        source_text = str(row.get("source_text") or "")
        explicit_segments = row.get("segments")
        if explicit_segments is not None:
            if not isinstance(explicit_segments, list) or not explicit_segments:
                raise ValueError(f"COT step {step_id} has invalid explicit segments")
            stored_by_id = {
                str(claim.get("claim_id") or ""): claim
                for claim in stored_claims
                if claim.get("claim_id")
            }
            cursor = 0
            seen_segment_claims: set[str] = set()
            rendered_parts: list[str] = []
            for segment_index, raw_segment in enumerate(explicit_segments, start=1):
                if not isinstance(raw_segment, dict):
                    raise ValueError(
                        f"COT step {step_id} segment {segment_index} is not an object"
                    )
                start = int(raw_segment.get("source_start", -1))
                end = int(raw_segment.get("source_end", -1))
                if start != cursor or end <= start or end > len(source_text):
                    raise ValueError(
                        f"COT step {step_id} segments do not exactly cover source text "
                        f"at segment {segment_index}: expected_start={cursor} start={start} end={end}"
                    )
                segment_text = source_text[start:end]
                kind = str(raw_segment.get("kind") or "")
                if kind == "context":
                    scope_id = str(raw_segment.get("scope_id") or "")
                    if scope_id:
                        scope_type = str(raw_segment.get("scope_type") or "context")
                        target_ids = [
                            str(value)
                            for value in (raw_segment.get("applies_to_claim_ids") or [])
                        ]
                        applies_to = _compact_scope_targets(target_ids)
                        rendered_parts.append(
                            f"[COT_SCOPE {scope_id} type={scope_type} "
                            f"applies_to={applies_to}]\n{segment_text}\n"
                            "[/COT_SCOPE]"
                        )
                    else:
                        rendered_parts.append(f"[COT_CONTEXT {step_id}] {segment_text}")
                elif kind == "claim":
                    claim_id = str(raw_segment.get("claim_id") or "")
                    claim = stored_by_id.get(claim_id)
                    if claim is None or claim_id in seen_segment_claims:
                        raise ValueError(
                            f"COT step {step_id} has invalid segment claim id {claim_id!r}"
                        )
                    if str(claim.get("source_text") or "") != segment_text:
                        raise ValueError(
                            f"COT step {step_id} segment text mismatch for {claim_id}"
                        )
                    seen_segment_claims.add(claim_id)
                    rendered_parts.append(
                        f"[COT_CLAIM {claim_id}]\n{segment_text}\n[/COT_CLAIM {claim_id}]"
                    )
                else:
                    raise ValueError(
                        f"COT step {step_id} segment {segment_index} has invalid kind {kind!r}"
                    )
                cursor = end
            if cursor != len(source_text):
                raise ValueError(
                    f"COT step {step_id} explicit segments leave source text uncovered"
                )
            if seen_segment_claims != set(stored_by_id):
                missing = sorted(set(stored_by_id) - seen_segment_claims)
                raise ValueError(
                    f"COT step {step_id} explicit segments omit claims: {missing}"
                )
            rendered_source = "\n\n".join(rendered_parts)
            blocks.append(
                f"[COT_STEP {step_id} {metadata}]\n"
                f"{rendered_source}\n"
                f"[/COT_STEP {step_id}]"
            )
            continue
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n+", source_text)
            if paragraph.strip()
        ] or [source_text]
        claim_index = 0
        rendered_parts: list[str] = []
        for paragraph in paragraphs:
            if not requires_formalization:
                rendered_parts.append(f"[COT_CONTEXT {step_id}] {paragraph}")
                continue
            if stored_claims:
                paragraph_hash = hashlib.sha256(paragraph.encode("utf-8")).hexdigest()
                candidates = claim_ids_by_hash.get(paragraph_hash, [])
                use_index = claim_hash_uses[paragraph_hash]
                claim_id = candidates[use_index] if use_index < len(candidates) else ""
                claim_hash_uses[paragraph_hash] += 1
                if not claim_id:
                    rendered_parts.append(f"[COT_CONTEXT {step_id}] {paragraph}")
                    continue
                rendered_parts.append(
                    f"[COT_CLAIM {claim_id}]\n{paragraph}\n[/COT_CLAIM {claim_id}]"
                )
                continue
            if re.fullmatch(r"\s*#{1,6}\s+[^\n]+", paragraph):
                rendered_parts.append(f"[COT_CONTEXT {step_id}] {paragraph}")
                continue
            claim_index += 1
            rendered_parts.append(
                f"[COT_CLAIM {step_id}.C{claim_index:03d}]\n{paragraph}\n"
                f"[/COT_CLAIM {step_id}.C{claim_index:03d}]"
            )
        rendered_source = "\n\n".join(rendered_parts)
        blocks.append(
            f"[COT_STEP {step_id} {metadata}]\n"
            f"{rendered_source}\n"
            f"[/COT_STEP {step_id}]"
        )
    return "\n\n".join(blocks)


def _enabled_semantic_issues(
    issues: list[SemanticIssue],
    *,
    require_step_ids: bool,
    static_gate: bool,
) -> list[SemanticIssue]:
    """Apply the additive E1 (binding) / E2 (static) feature boundary."""
    if static_gate:
        return issues
    if require_step_ids:
        return [issue for issue in issues if issue.category == "binding"]
    return []


def _emit_semantic_check(
    tracer,
    *,
    thm_name: str,
    phase: str,
    attempt: int,
    issues: list[SemanticIssue],
) -> None:
    if tracer is None:
        return
    tracer.emit(TraceEvent(
        kind="blueprint_semantic_check",
        thm_name=thm_name,
        ok=not issues,
        args={
            "phase": phase,
            "attempt": attempt,
            "issue_count": len(issues),
            "issues": [issue.to_dict() for issue in issues],
        },
    ))

# Appendix A specifies 262,144 (matches DeepSeek-V4-Flash's completion budget).
# Read at call time so experiment YAML environment settings apply after import.
def _max_tokens() -> int:
    return int(os.environ.get("GOEDEL_BLUEPRINT_MAX_TOKENS", "262144"))


MAX_RETRIES = 8


class BlueprintGenerationError(RuntimeError):
    """Terminal Phase-1 failure with the last unusable candidate attached."""

    def __init__(
        self,
        message: str,
        *,
        last_candidate: str = "",
        diagnostics: list[str] | None = None,
        attempt: int = 0,
        finish_reason: str | None = None,
        failure_stage: str = "model_output",
    ) -> None:
        super().__init__(message)
        self.last_candidate = last_candidate
        self.diagnostics = list(diagnostics or [])
        self.attempt = attempt
        self.finish_reason = finish_reason
        self.failure_stage = failure_stage


@lru_cache(maxsize=2)
def _load_phase1_tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def phase1_request_max_tokens(messages: list[dict[str, str]]) -> int:
    """Fit every initial/retry Blueprint completion in the model context."""
    cap = int(os.environ.get("GOEDEL_PHASE1_MAX_OUTPUT_CAP", str(_max_tokens())))
    context = int(os.environ.get("GOEDEL_PHASE1_MODEL_MAX_CONTEXT", "0"))
    tokenizer_path = os.environ.get("GOEDEL_TOKENIZER_PATH", "").strip()
    if context <= 0 or not tokenizer_path:
        return cap
    margin = int(os.environ.get("GOEDEL_PHASE1_CONTEXT_SAFETY_MARGIN", "512"))
    minimum = int(os.environ.get("GOEDEL_PHASE1_MIN_OUTPUT_TOKENS", "512"))
    encoded = _load_phase1_tokenizer(tokenizer_path).apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    prompt_tokens = len(encoded)
    available = context - prompt_tokens - margin
    if available < minimum:
        raise BlueprintGenerationError(
            "insufficient_context: Phase-1 prompt leaves "
            f"{available} output tokens (prompt={prompt_tokens}, context={context}, "
            f"margin={margin}, minimum={minimum})",
            failure_stage="phase1_context_budget",
        )
    return min(cap, available)


def _emit_usage(tracer, thm_name: str, phase: str, model: str, response) -> None:
    """Log token usage from a chat.completions/responses API response, if a
    tracer was given. `response.usage` is present on both APIs but with
    different field names, so normalize to prompt/completion/total."""
    if tracer is None:
        return
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    if prompt is None:
        prompt = getattr(usage, "input_tokens", 0)
    if completion is None:
        completion = getattr(usage, "output_tokens", 0)
    total = getattr(usage, "total_tokens", None) or (prompt + completion)
    tracer.emit(TraceEvent(
        kind="llm_usage",
        thm_name=thm_name,
        args={
            "phase": phase, "model": model,
            "prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": total,
        },
    ))


def _tool_calls_payload(message) -> list[dict]:
    return [
        {
            "id": tc.id,
            "type": getattr(tc, "type", "function"),
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            },
        }
        for tc in message.tool_calls or []
    ]


def _emit_llm_response(
    tracer,
    *,
    thm_name: str,
    phase: str,
    model: str,
    response,
    attempt: int,
    turn: int,
) -> None:
    if tracer is None:
        return
    choice = response.choices[0]
    msg = choice.message
    tracer.emit(TraceEvent(
        kind="llm_response",
        thm_name=thm_name,
        turn=turn,
        result=msg.content or "",
        args={
            "phase": phase,
            "model": model,
            "attempt": attempt,
            "finish_reason": getattr(choice, "finish_reason", None),
            "tool_calls": _tool_calls_payload(msg),
        },
    ))


def _emit_lean_check_result(
    tracer,
    *,
    thm_name: str,
    phase: str,
    attempt: int,
    target: str,
    result: CompilerResult,
) -> None:
    if tracer is None:
        return
    tracer.emit(TraceEvent(
        kind="lean_check_result",
        thm_name=thm_name,
        args={
            "phase": phase,
            "attempt": attempt,
            "target": target,
            "errors": result.errors,
            "warnings": result.warnings,
            "goals": result.goals,
            "raw_output": result.raw_output,
            "failure_kind": result.failure_kind,
            "timings": result.timings,
        },
        ok=result.success,
    ))


def _set_latest_blueprint_retry(
    messages: list[dict],
    base_messages: tuple[dict, ...],
    lean_code: str,
    feedback: str,
    *,
    finish_reason: str | None = None,
) -> None:
    """Keep one bounded, useful repair turn without replaying model reasoning.

    Blueprint responses from reasoning models can spend the entire completion
    budget before or around the Lean block.  Replaying that raw response on
    every repair attempt makes the prompt grow monotonically and can leave no
    room for the configured completion budget.  The compiler only repairs the
    latest Lean candidate, so retain the immutable original prompt plus that
    candidate and its latest diagnostic.
    """
    candidate = lean_code.strip()
    candidate_content = (
        f"```lean\n{candidate}\n```"
        if candidate
        else "No valid Lean blueprint was emitted."
    )
    retry_feedback = feedback
    if finish_reason == "length":
        retry_feedback = (
            "The previous response reached its output limit. Emit only one concise, "
            "complete Lean file with no reasoning outside the code block.\n\n"
            + retry_feedback
        )
    messages[:] = [dict(message) for message in base_messages]
    messages.extend([
        {"role": "assistant", "content": candidate_content},
        {"role": "user", "content": retry_feedback},
    ])


def _semantic_repair_guidance(issues: list[SemanticIssue]) -> str:
    """Give bounded, issue-specific guidance without adding another LLM call."""
    codes = {issue.code for issue in issues}
    guidance: list[str] = []
    if codes.intersection({"STEP_MAPPING_ABSENT", "CLAIM_MAPPING_ABSENT"}):
        guidance.append(
            "Create a substantive formal node for every listed absent source step/claim, "
            "using its exact `COT_CLAIM:CNNN` (or legacy `COT_STEP:SNNN.CNNN`) title, and then "
            "include it in a downstream `sorry_using` chain to the root; never replace "
            "the missing translation with `True`, a reflexive equality, or a placeholder."
        )
    if codes.intersection({
        "MISSING_STEP_MAPPING", "MALFORMED_STEP_MAPPING", "MISSING_CLAIM_MAPPING",
    }):
        guidance.append(
            "Repair each listed node's provenance title so it names exactly the source "
            "claim that its existing formal declaration translates."
        )
    if codes.intersection({"STEP_NOT_ROOT_REACHABLE", "CLAIM_NOT_ROOT_REACHABLE"}):
        guidance.append(
            "The corresponding formal nodes already exist: preserve their declarations "
            "and connect them to the root through downstream `sorry_using` dependencies."
        )
    if any(code.startswith("VACUOUS_") for code in codes):
        guidance.append(
            "Replace every vacuous type/body with the exact source proposition over the "
            "same shared objects, then connect that substantive node to the root."
        )
    if any(code.startswith("UNCONSTRAINED_EXISTS") for code in codes):
        guidance.append(
            "Model the original constrained object and givens explicitly; the root must "
            "answer the original question, not merely choose a closed witness."
        )
    if "ROOT_MISSING_CLAIMED_ANSWER" in codes:
        guidance.append(
            "Keep the original COT's claimed answer literally in the root proposition "
            "about the original modeled object."
        )
    if not guidance:
        guidance.append(
            "Fix every listed issue while preserving all existing substantive source claims."
        )
    return "\n".join(f"- {item}" for item in guidance)


def _call_blueprint_model(
    client,
    model: str,
    messages: list[dict],
    reasoning_kwargs: dict,
    max_tokens: int,
    tracer=None,
    thm_name: str = "",
    phase: str = "",
    attempt: int = 0,
):
    response = chat_completion_with_retry(
        client,
        tracer=tracer,
        thm_name=thm_name,
        phase=phase,
        model_id=model,
        operation="blueprint_generate",
        trace_args={"attempt": attempt, "turn": 1, "max_tokens": max_tokens},
        model=model, messages=messages, max_completion_tokens=max_tokens, **reasoning_kwargs,
    )
    _emit_usage(tracer, thm_name, phase, model, response)
    _emit_llm_response(
        tracer,
        thm_name=thm_name,
        phase=phase,
        model=model,
        response=response,
        attempt=attempt,
        turn=1,
    )
    return response


@dataclass
class BlueprintNode:
    name: str
    kind: str  # "definition" | "lemma" | "theorem"
    statement: str
    proof_sketch: str
    dependencies: list[str] = field(default_factory=list)
    lean_declaration: str = ""
    # LeanArchitect already persists ``title`` as part of the native
    # ``@[blueprint]`` attribute.  Semantic-fidelity experiments encode the
    # immutable source binding as ``title := \"COT_STEP:S001\"`` so the
    # mapping survives parsing, checkpointing, and export without inventing a
    # custom Lean attribute.
    title: str = ""
    source_step_id: str = ""

    def signature(self) -> str:  # 针对单个 node，提取为类似 theorem l1 (n : ℕ) : n + 0 = n
        """Strip the @[blueprint ...] attribute and sorry_using proof body,
        returning just the declaration up to (not including) ':='.

        Used to re-declare an already-proved dependency as a real, standalone
        lemma (signature + its actual proof) so sibling nodes can reference it
        by name instead of hitting "unknown identifier" - proven dependencies
        are otherwise only ever shown to the model as prompt text, never
        actually compiled into scope.
        """
        return extract_blueprint_signature(self.lean_declaration)

    def full_declaration(self) -> str:
        """Return this node's complete declaration without ``@[blueprint]``.

        Unlike :meth:`signature`, this deliberately preserves a definition's
        outer ``:=`` and its entire right-hand side.  Attribute removal happens
        before declaration detection, so declaration-like words in blueprint
        comments cannot become part of the returned Lean code.
        """
        return extract_current_node_decl(self.lean_declaration)

    def cache_key(self) -> str:
        """Declaration shape plus dependencies for cache staleness checks.

        Proof nodes use their signature; definitions use the complete
        declaration so an RHS change is visible. `signature()` alone only
        covers the text before `:=`, so a node
        whose sorry_using [...] dependency list changes (text AFTER `:=`)
        while its exposed statement stays byte-identical would otherwise be
        invisible to staleness checks, even though its cached proof was
        spliced together with the OLD set of sibling declarations in scope.
        """
        declaration_shape = (
            self.full_declaration() if self.kind == "definition" else self.signature()
        )
        return (
            declaration_shape
            + "\x00deps:"
            + ",".join(sorted(self.dependencies))
            + "\x00source_step_id:"
            + self.source_step_id
        )


def render_solved_declaration(node: BlueprintNode, proof_body: str) -> str:
    """Render a solved node as Lean code suitable for dependency context.

    Definitions already carry their executable body in the validated
    blueprint, so cached proof text is intentionally ignored.  Proof nodes
    are reconstructed from their signature and the proof accepted by Lean.
    """
    if node.kind == "definition":
        return node.full_declaration()
    return f"{node.signature()} {proof_body_to_decl_suffix(proof_body)}"


@dataclass
class Blueprint:
    nodes: list[BlueprintNode]
    lean_file: str  # full compilable @[blueprint]-annotated Lean file
    target_theorem: str
    phase2_header: str = MATHLIB_HEADER
    semantic_gate_results: list[dict] = field(default_factory=list)
    semantic_audit_result: dict = field(default_factory=dict)

    def node_by_name(self, name: str) -> BlueprintNode | None:
        return next((n for n in self.nodes if n.name == name), None)

    def nodes_by_name(self) -> dict[str, BlueprintNode]:
        return {n.name: n for n in self.nodes}

    def dependency_order(self) -> list[BlueprintNode]:
        """Topological order (definitions first, theorem last)."""
        node_map = self.nodes_by_name()
        ordered: list[BlueprintNode] = []
        visited: set[str] = set()
        visiting: set[str] = set()
        stack: list[str] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            if name in visiting:
                cycle_start = stack.index(name) if name in stack else 0
                cycle = stack[cycle_start:] + [name]
                raise ValueError(f"Blueprint dependency cycle: {' -> '.join(cycle)}")
            node = node_map.get(name)
            if node is None:
                return
            visiting.add(name)
            stack.append(name)
            for dep in node.dependencies:
                if dep in node_map:
                    visit(dep)
            stack.pop()
            visiting.remove(name)
            visited.add(name)
            ordered.append(node)

        for node in self.nodes:
            visit(node.name)
        return ordered


def _safe_phase2_header(lean_code: str) -> str:
    """Extract the leading commands Phase 2 may safely preserve."""
    imports: list[str] = []
    other: list[str] = []
    in_block_comment = False
    for raw_line in lean_code.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if in_block_comment:
            if "-/" in stripped:
                in_block_comment = False
            continue
        if stripped.startswith("/-"):
            if "-/" not in stripped[2:]:
                in_block_comment = True
            continue
        if stripped.startswith("--"):
            continue
        if stripped.startswith("import "):
            imports.append(stripped)
            continue
        if stripped.startswith("open ") or stripped.startswith("open scoped "):
            other.append(stripped)
            continue
        if stripped.startswith("set_option "):
            other.append(stripped)
            continue
        break

    if not any(line == "import Mathlib" for line in imports):
        imports.insert(0, "import Mathlib")
    if not any(line == "import Architect" for line in imports):
        insert_at = 1 if imports and imports[0] == "import Mathlib" else len(imports)
        imports.insert(insert_at, "import Architect")
    if not any(line.startswith("set_option autoImplicit ") for line in other):
        other.insert(0, "set_option autoImplicit false")
    return "\n".join(imports + other).rstrip() + "\n\n"


def _transitive_node_deps(node: BlueprintNode, blueprint: Blueprint) -> set[str]:
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


def _phase2_preflight_file(blueprint: Blueprint, node: BlueprintNode) -> str:
    parts = [blueprint.phase2_header.rstrip()]
    parts.extend(
        definition.full_declaration()
        for definition in blueprint.nodes
        if definition.kind == "definition" and definition.name != node.name
    )
    ancestor_deps = _transitive_node_deps(node, blueprint)
    parts.extend(
        dep_node.full_declaration()
        for dep_node in blueprint.dependency_order()
        if dep_node.kind != "definition"
        and dep_node.name in ancestor_deps
    )
    parts.append(node.full_declaration())
    return "\n\n".join(part.strip() for part in parts if part.strip()) + "\n"


def phase2_standalone_contract_errors(
    blueprint: Blueprint,
    compiler: KiminaLeanCompiler,
    *,
    limit: int = 12,
    concurrency: int = 1,
) -> list[str]:
    """Compile proof nodes as Phase 2 would see them before accepting a blueprint."""
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")

    nodes = [
        node for node in blueprint.nodes
        if node.kind in {"lemma", "theorem"}
    ]
    results = compiler.check_many(
        [
            CompileRequest(
                _phase2_preflight_file(blueprint, node),
                allow_sorry=True,
                request_id=f"phase2-contract-{index}-{node.name}",
            )
            for index, node in enumerate(nodes)
        ],
        batch_concurrency=concurrency,
    )
    errors: list[str] = []
    for node, result in zip(nodes, results, strict=True):
        if result.failure_kind == "infra":
            message = "\n".join(result.diagnostics) or result.raw_output[-2000:]
            raise KiminaInfrastructureError(message)
        if result.success:
            continue
        message = "\n".join(result.diagnostics) or result.raw_output[-2000:]
        errors.append(
            f"phase2_standalone_failed: node `{node.name}` does not compile when "
            f"assembled as a standalone Phase 2 goal.\n{message}"
        )
    if limit:
        errors = errors[:limit]
    return errors


def phase2_contract_errors(blueprint: Blueprint) -> list[str]:
    """Return structural errors that would make Phase 2 node proving invalid."""
    errors: list[str] = []
    for node in blueprint.nodes:
        if node.kind not in {"lemma", "theorem"}:
            continue
        current_decl = extract_current_node_decl(node.lean_declaration)
        placeholder_count = len(BLUEPRINT_PROOF_RE.findall(current_decl))
        if placeholder_count == 0:
            errors.append(
                f"missing_sorry_using_placeholder: proof node `{node.name}` must contain "
                "`:= by sorry_using [...]`, not a completed proof or plain `sorry`."
            )
        elif placeholder_count > 1:
            errors.append(
                f"multiple_sorry_using_placeholders: proof node `{node.name}` contains "
                f"{placeholder_count} `sorry_using` placeholders; expected exactly one."
            )
    try:
        blueprint.dependency_order()
    except ValueError as exc:
        errors.append(f"dependency_cycle: {exc}")
    return errors


def phase2_contract_error_counts(errors: list[str]) -> dict[str, int]:
    return dict(Counter(error.split(":", 1)[0] for error in errors))


_REACHABILITY_ONLY_ISSUES = {
    "CLAIM_NOT_ROOT_REACHABLE",
    "STEP_NOT_ROOT_REACHABLE",
}


def _repair_root_reachability_only(
    blueprint: Blueprint,
    issues: list[SemanticIssue],
) -> tuple[str, list[str]] | None:
    """Connect already-formalized source claims to the root without an LLM.

    This repair is deliberately narrow: it is available only when *every*
    enabled semantic issue says that an existing claim/step node is outside
    the root dependency closure.  It changes only the root theorem's
    ``sorry_using`` metadata; declarations and propositions remain byte-for-
    byte identical.  Candidates which also contain a vacuous, malformed, or
    missing translation must go through the normal semantic repair path.

    Every existing node mapped to a missing source ID is connected.  A single
    claim may legitimately require several formal nodes, so choosing one
    representative would risk making the other translated clauses dead code.
    """
    if not issues or any(
        issue.code not in _REACHABILITY_ONLY_ISSUES
        or issue.category != "binding"
        for issue in issues
    ):
        return None

    root = blueprint.node_by_name(blueprint.target_theorem)
    if root is None or root.kind not in {"lemma", "theorem"}:
        return None
    if len(blueprint.nodes_by_name()) != len(blueprint.nodes):
        return None
    root_decl = root.lean_declaration
    current_root_decl = extract_current_node_decl(root_decl)
    proof_matches = list(BLUEPRINT_PROOF_RE.finditer(current_root_decl))
    if (
        len(proof_matches) != 1
        or root_decl.count(current_root_decl) != 1
        or blueprint.lean_file.count(root_decl) != 1
    ):
        return None

    missing_ids = list(dict.fromkeys(issue.step_id for issue in issues if issue.step_id))
    if len(missing_ids) != len({issue.step_id for issue in issues}):
        return None

    def node_covers(node: BlueprintNode, source_id: str) -> bool:
        if source_id.count("."):
            return node.source_step_id == source_id
        return node.source_step_id.split(".", 1)[0] == source_id

    selected = [
        node.name
        for node in blueprint.nodes
        if node.name != root.name
        and any(node_covers(node, source_id) for source_id in missing_ids)
    ]
    for source_id in missing_ids:
        if not any(
            node_covers(node, source_id)
            for node in blueprint.nodes
            if node.name != root.name
        ):
            return None

    dependencies = list(dict.fromkeys([*root.dependencies, *selected]))
    proof_match = proof_matches[0]
    replacement = f":= by sorry_using [{', '.join(dependencies)}]"
    repaired_current_decl = (
        current_root_decl[:proof_match.start()]
        + replacement
        + current_root_decl[proof_match.end():]
    )
    repaired_root = root_decl.replace(current_root_decl, repaired_current_decl, 1)
    repaired_code = blueprint.lean_file.replace(root_decl, repaired_root, 1)
    if repaired_code == blueprint.lean_file:
        return None

    try:
        repaired_blueprint = _parse_blueprint(repaired_code, blueprint.target_theorem)
        repaired_blueprint.dependency_order()
    except (ValueError, RuntimeError):
        return None
    if [node.name for node in repaired_blueprint.nodes] != [
        node.name for node in blueprint.nodes
    ]:
        return None
    for before, after in zip(blueprint.nodes, repaired_blueprint.nodes, strict=True):
        if (
            before.kind,
            before.title,
            before.source_step_id,
            before.statement,
            before.proof_sketch,
        ) != (
            after.kind,
            after.title,
            after.source_step_id,
            after.statement,
            after.proof_sketch,
        ):
            return None
        if before.kind == "definition":
            if before.full_declaration() != after.full_declaration():
                return None
        elif before.signature() != after.signature():
            return None
        expected_dependencies = (
            dependencies if before.name == root.name else before.dependencies
        )
        if after.dependencies != expected_dependencies:
            return None
    return repaired_code, selected


def format_phase2_contract_errors(errors: list[str], limit: int = 12) -> str:
    shown = errors[:limit]
    suffix = "" if len(errors) <= limit else f"\n... and {len(errors) - limit} more"
    return "\n".join(f"- {error}" for error in shown) + suffix


def generate_blueprint(
    theorem_stmt: str,
    nl_proof: str | None = None,
    model: str = "labs-leanstral-1-5",
    *,
    compiler: KiminaLeanCompiler,
    tracer=None,
    thm_name: str = "",
    max_retries: int = MAX_RETRIES,
    phase2_contract_check_concurrency: int = 1,
) -> Blueprint:
    """
    Generate a @[blueprint]-annotated Lean dependency graph for `theorem_stmt`.

    Uses the verbatim system prompt from Appendix C.1 of the paper.
    Validates via lean_compile after each LLM attempt (up to max_retries).

    """
    client = make_client(model)
    user_content = _build_user_prompt(theorem_stmt, nl_proof)
    messages = [
        {"role": "system", "content": BLUEPRINT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    base_messages = tuple(dict(message) for message in messages)

    last_lean_code = None
    for attempt in range(max_retries):
        response = _call_blueprint_model(
            client, model, messages, _reasoning_kwargs(model),
            phase1_request_max_tokens(messages),
            tracer=tracer, thm_name=thm_name, phase="phase1", attempt=attempt + 1,
        )
        lean_code = _extract_lean_code(response.choices[0].message.content)
        last_lean_code = lean_code

        target = _extract_target_name(lean_code, theorem_stmt)
        result = compiler.check_blueprint(lean_code, target)
        _emit_lean_check_result(
            tracer,
            thm_name=thm_name,
            phase="phase1",
            attempt=attempt + 1,
            target=target,
            result=result,
        )
        if result.failure_kind == "infra":
            raise KiminaInfrastructureError(
                "\n".join(result.diagnostics) or result.raw_output[-2000:]
            )
        if result.success:
            parsed = _parse_blueprint(lean_code, target)
            if parsed.nodes:
                contract_errors = phase2_contract_errors(parsed)
                if not contract_errors:
                    contract_errors = phase2_standalone_contract_errors(
                        parsed,
                        compiler,
                        concurrency=phase2_contract_check_concurrency,
                    )
                if contract_errors:
                    feedback = (
                        f"The file compiled, but the blueprint is not usable by Phase 2 "
                        f"(attempt {attempt + 1}/{max_retries}):\n\n"
                        f"{format_phase2_contract_errors(contract_errors)}\n\n"
                        "Fix the blueprint contract and re-emit the whole file. Every "
                        "`lemma` and `theorem` blueprint node must end with exactly one "
                        "`:= by sorry_using [...]` placeholder; do not provide completed "
                        "proofs or plain `sorry` bodies in blueprint proof nodes. "
                        "Definitions may keep executable bodies. The dependency graph "
                        "must be acyclic."
                    )
                    _set_latest_blueprint_retry(
                        messages,
                        base_messages,
                        lean_code,
                        feedback,
                        finish_reason=getattr(response.choices[0], "finish_reason", None),
                    )
                    continue
                return parsed
            feedback = (
                f"The file compiled, but contains no `@[blueprint ...]`-annotated "
                f"declarations (attempt {attempt + 1}/{max_retries}). You must "
                "annotate the target theorem (and any helper lemmas) with "
                "`@[blueprint ...]` and give each a `sorry_using [...]` proof body. "
                "Re-emit the blueprint with proper annotations."
            )
            _set_latest_blueprint_retry(
                messages,
                base_messages,
                lean_code,
                feedback,
                finish_reason=getattr(response.choices[0], "finish_reason", None),
            )
            continue
        error_feedback = "\n".join(result.diagnostics) or result.raw_output[-2000:]
        feedback = (
            f"lean_compile reported errors (attempt {attempt + 1}/{max_retries}):\n\n"
            f"{error_feedback}\n\n"
            "Fix the issues and call lean_compile again."
        )
        _set_latest_blueprint_retry(
            messages,
            base_messages,
            lean_code,
            feedback,
            finish_reason=getattr(response.choices[0], "finish_reason", None),
        )
    raise RuntimeError(
        f"Blueprint generation failed after {max_retries} attempts "
        "without a validated blueprint"
    )


def generate_blueprint_from_informal(
    informal_statement: str,
    informal_proof: str | None,
    target_name: str,
    model: str = "labs-leanstral-1-5",
    *,
    compiler: KiminaLeanCompiler,
    cot_manifest_json: str = "",
    claimed_answer: str = "",
    tracer=None,
    thm_name: str = "",
    max_retries: int = MAX_RETRIES,
    phase2_contract_check_concurrency: int = 1,
    semantic_fidelity_enabled: bool = False,
    semantic_require_step_ids: bool = False,
    semantic_static_gate: bool = False,
    semantic_minimal_ir: bool = False,
    semantic_audit_mode: str = "none",
    semantic_max_repair_attempts: int = 1,
) -> Blueprint:
    """Generate and strictly validate a blueprint from informal text only.

    Unlike generate_blueprint(), this entry point has no formal Lean theorem
    signature to preserve. The model must formalize the main theorem itself,
    but the theorem identifier is fixed by target_name so downstream
    checkpointing, validation, and scoring remain stable.
    """
    if semantic_max_repair_attempts < 0:
        raise ValueError("semantic_max_repair_attempts must be non-negative")
    if semantic_audit_mode not in {"none", "risk", "full"}:
        raise ValueError("semantic_audit_mode must be one of: none, risk, full")
    if (
        semantic_require_step_ids
        or semantic_static_gate
        or semantic_minimal_ir
        or semantic_audit_mode != "none"
    ) and not semantic_fidelity_enabled:
        raise ValueError("semantic subfeatures require semantic_fidelity_enabled=true")
    manifest_rows = _decode_manifest_rows(cot_manifest_json) if semantic_fidelity_enabled else []
    semantic_manifest = (
        parse_cot_manifest(manifest_rows) if semantic_fidelity_enabled else None
    )
    if semantic_fidelity_enabled and not manifest_rows:
        raise ValueError("semantic_fidelity_enabled requires a non-empty cot_manifest_json")
    prompt_proof = informal_proof or ""
    if semantic_fidelity_enabled:
        prompt_proof = _render_step_grounded_proof(
            cot_manifest_json,
            include_ir=semantic_minimal_ir,
        )

    client = make_client(model)
    messages = [
        {
            "role": "system",
            "content": ROBUSTPA_BLUEPRINT_SYSTEM_PROMPT
            + (_SEMANTIC_BLUEPRINT_SYSTEM_SUFFIX if semantic_fidelity_enabled else ""),
        },
        {
            "role": "user",
            "content": render(
                ROBUSTPA_BLUEPRINT_USER_TEMPLATE,
                target_name=target_name,
                informal_statement=informal_statement,
                informal_proof=prompt_proof,
            ) + (
                "\n\nThe claimed final answer that must remain grounded in the original "
                f"problem object and final COT step is: `{claimed_answer}`."
                if semantic_fidelity_enabled else ""
            ),
        },
    ]
    base_messages = tuple(dict(message) for message in messages)

    last_error_feedback = ""
    last_candidate = ""
    last_diagnostics: list[str] = []
    last_finish_reason: str | None = None
    last_failure_stage = "model_output"
    # Local deterministic failures and model-audit failures are different
    # repair opportunities.  Sharing one counter meant that a candidate which
    # fixed a static issue could be rejected immediately by its first audit,
    # without ever receiving the audit's actionable feedback.
    local_semantic_repair_count = 0
    audit_semantic_repair_count = 0
    observed_semantic_issues: list[str] = []
    for attempt in range(max_retries):
        semantic_check_issues: list[SemanticIssue] = []
        deterministic_graph_repair_nodes: list[str] = []
        generation_kwargs = _reasoning_kwargs(model)
        if attempt > 0:
            # Repair turns should apply the supplied diagnostics, not spend a
            # second long completion re-solving or debating the source COT.
            generation_kwargs["temperature"] = 0
            if "qwen" in model.lower():
                generation_kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False},
                }
        response = _call_blueprint_model(
            client,
            model,
            messages,
            reasoning_kwargs=generation_kwargs,
            max_tokens=phase1_request_max_tokens(messages),
            tracer=tracer,
            thm_name=thm_name,
            phase="phase1",
            attempt=attempt + 1,
        )
        choice = response.choices[0]
        lean_code = _extract_lean_code(choice.message.content)
        last_candidate = lean_code
        last_finish_reason = getattr(choice, "finish_reason", None)
        emitted_target = _extract_target_name(lean_code, "")
        if emitted_target != target_name:
            last_error_feedback = (
                f"The main theorem must be named `{target_name}`, but the "
                f"latest output's final theorem is `{emitted_target or '<missing>'}`."
            )
            last_diagnostics = [last_error_feedback]
            last_failure_stage = "blueprint_contract"
            feedback = (
                f"{last_error_feedback}\n\n"
                "Re-emit the whole Lean file with exactly one main theorem "
                f"named `{target_name}`."
            )
            _set_latest_blueprint_retry(
                messages,
                base_messages,
                lean_code,
                feedback,
                finish_reason=last_finish_reason,
            )
            continue

        if semantic_fidelity_enabled:
            candidate = _parse_blueprint(lean_code, target_name)
            if candidate.nodes:
                all_semantic_check_issues = validate_blueprint_fidelity(
                    candidate,
                    semantic_manifest,
                    claimed_answer=claimed_answer,
                    require_step_bindings=semantic_require_step_ids,
                )
                semantic_check_issues = _enabled_semantic_issues(
                    all_semantic_check_issues,
                    require_step_ids=semantic_require_step_ids,
                    static_gate=semantic_static_gate,
                )
                _emit_semantic_check(
                    tracer,
                    thm_name=thm_name,
                    phase="phase1",
                    attempt=attempt + 1,
                    issues=semantic_check_issues,
                )
                graph_repair = _repair_root_reachability_only(
                    candidate,
                    semantic_check_issues,
                )
                if semantic_check_issues and graph_repair is not None:
                    repaired_code, repaired_nodes = graph_repair
                    repaired_candidate = _parse_blueprint(repaired_code, target_name)
                    repaired_issues = _enabled_semantic_issues(
                        validate_blueprint_fidelity(
                            repaired_candidate,
                            semantic_manifest,
                            claimed_answer=claimed_answer,
                            require_step_bindings=semantic_require_step_ids,
                        ),
                        require_step_ids=semantic_require_step_ids,
                        static_gate=semantic_static_gate,
                    )
                    if not repaired_issues:
                        lean_code = repaired_code
                        last_candidate = repaired_code
                        candidate = repaired_candidate
                        semantic_check_issues = []
                        deterministic_graph_repair_nodes = repaired_nodes
                        if tracer is not None:
                            tracer.emit(TraceEvent(
                                kind="blueprint_graph_repair",
                                thm_name=thm_name,
                                ok=True,
                                args={
                                    "phase": "phase1",
                                    "attempt": attempt + 1,
                                    "root": target_name,
                                    "added_dependencies": repaired_nodes,
                                    "changed_formal_declarations": False,
                                },
                            ))
                        _emit_semantic_check(
                            tracer,
                            thm_name=thm_name,
                            phase="phase1_graph_repair",
                            attempt=attempt + 1,
                            issues=[],
                        )
                if semantic_check_issues:
                    for issue in semantic_check_issues:
                        issue_key = ":".join(
                            part for part in (issue.code, issue.step_id, issue.node_name) if part
                        )
                        if issue_key not in observed_semantic_issues:
                            observed_semantic_issues.append(issue_key)
                    local_semantic_repair_count += 1
                    last_error_feedback = (
                        "The local semantic-fidelity gate rejected this candidate "
                        "before Lean execution:\n\n"
                        f"{format_semantic_issues(semantic_check_issues)}"
                    )
                    last_diagnostics = [last_error_feedback]
                    last_failure_stage = "semantic_gate"
                    if local_semantic_repair_count > semantic_max_repair_attempts:
                        raise BlueprintGenerationError(
                            last_error_feedback,
                            last_candidate=last_candidate,
                            diagnostics=last_diagnostics,
                            attempt=attempt + 1,
                            finish_reason=last_finish_reason,
                            failure_stage=last_failure_stage,
                        )
                    feedback = (
                        f"{last_error_feedback}\n\nCorrect the translation contract and "
                        "re-emit the entire file. Preserve the source COT exactly; a false "
                        "step must remain a proposition and explicit proof gap.\n\n"
                        f"Issue-specific repair rules:\n{_semantic_repair_guidance(semantic_check_issues)}\n\n"
                        "Previously observed issues that must not regress: "
                        f"{', '.join(observed_semantic_issues)}"
                    )
                    _set_latest_blueprint_retry(
                        messages,
                        base_messages,
                        lean_code,
                        feedback,
                        finish_reason=last_finish_reason,
                    )
                    continue

        result = compiler.check_blueprint(lean_code, target_name)
        _emit_lean_check_result(
            tracer,
            thm_name=thm_name,
            phase="phase1",
            attempt=attempt + 1,
            target=target_name,
            result=result,
        )
        if result.failure_kind == "infra":
            raise KiminaInfrastructureError(
                "\n".join(result.diagnostics) or result.raw_output[-2000:]
            )
        if result.success:
            try:
                parsed = _parse_blueprint(lean_code, target_name)
            except Exception as exc:  # noqa: BLE001
                parsed = None
                last_error_feedback = f"Blueprint parsing failed: {type(exc).__name__}: {exc}"
                last_failure_stage = "parse"
            if parsed is None:
                pass
            elif not parsed.nodes:
                last_error_feedback = (
                    "The file compiled, but contains no `@[blueprint ...]`-annotated declarations."
                )
                last_failure_stage = "parse"
            else:
                if semantic_fidelity_enabled:
                    parsed.semantic_gate_results.append({
                        "stage": "phase1_local_gate",
                        "passed": True,
                        "issues": [],
                        "require_step_ids": semantic_require_step_ids,
                        "static_gate": semantic_static_gate,
                    })
                    if deterministic_graph_repair_nodes:
                        parsed.semantic_gate_results.append({
                            "stage": "phase1_graph_repair",
                            "passed": True,
                            "root": target_name,
                            "added_dependencies": deterministic_graph_repair_nodes,
                            "changed_formal_declarations": False,
                        })
                contract_errors = phase2_contract_errors(parsed)
                if not contract_errors:
                    contract_errors = phase2_standalone_contract_errors(
                        parsed,
                        compiler,
                        concurrency=phase2_contract_check_concurrency,
                    )
                if not contract_errors:
                    audit_risk_reasons = (
                        semantic_audit_risk_reasons(
                            parsed,
                            semantic_manifest,
                            claimed_answer=claimed_answer,
                        )
                        if semantic_audit_mode == "risk"
                        else []
                    )
                    should_audit = (
                        semantic_audit_mode == "full"
                        or (semantic_audit_mode == "risk" and bool(audit_risk_reasons))
                    )
                    if semantic_audit_mode == "risk" and not should_audit:
                        parsed.semantic_gate_results.append({
                            "stage": "phase1_semantic_audit",
                            "passed": True,
                            "mode": "risk",
                            "routed": False,
                            "risk_reasons": [],
                        })
                    if should_audit:
                        try:
                            audit = run_semantic_audit(
                                model,
                                prompt_proof,
                                parsed.lean_file,
                                mode=semantic_audit_mode,
                                informal_statement=informal_statement,
                                claimed_answer=claimed_answer,
                                client=client,
                                tracer=tracer,
                                thm_name=thm_name,
                                phase="phase1_semantic_audit",
                            )
                            audit_feedback = audit.diagnostics
                            audit_passed = audit.passed
                        except SemanticAuditFormatError as exc:
                            audit_feedback = (
                                f"Audit response format was invalid ({exc.reason}). "
                                "The next candidate must still satisfy every source-step contract."
                            )
                            audit_passed = False
                        if audit_passed:
                            parsed.semantic_audit_result = asdict(audit)
                            parsed.semantic_gate_results.append({
                                "stage": "phase1_semantic_audit",
                                "passed": True,
                                "mode": semantic_audit_mode,
                                "routed": True,
                                "risk_reasons": audit_risk_reasons,
                                "diagnostics": audit.diagnostics,
                                "request_id": audit.request_id,
                                "total_tokens": audit.total_tokens,
                            })
                            return parsed
                        audit_semantic_repair_count += 1
                        last_error_feedback = (
                            "The semantic-fidelity audit rejected this candidate:\n\n"
                            f"{audit_feedback}"
                        )
                        last_diagnostics = [last_error_feedback]
                        last_failure_stage = "semantic_audit"
                        if audit_semantic_repair_count > semantic_max_repair_attempts:
                            raise BlueprintGenerationError(
                                last_error_feedback,
                                last_candidate=last_candidate,
                                diagnostics=last_diagnostics,
                                attempt=attempt + 1,
                                finish_reason=last_finish_reason,
                                failure_stage=last_failure_stage,
                            )
                        feedback = (
                            f"{last_error_feedback}\n\nRepair only the Lean translation. "
                            "Do not repair, weaken, or omit the original COT claims. "
                            "Re-emit the complete blueprint with the same step bindings."
                        )
                        _set_latest_blueprint_retry(
                            messages,
                            base_messages,
                            lean_code,
                            feedback,
                            finish_reason=last_finish_reason,
                        )
                        continue
                    return parsed
                last_error_feedback = (
                    "The file compiled, but the blueprint is not usable by Phase 2:\n\n"
                    f"{format_phase2_contract_errors(contract_errors)}"
                )
                last_failure_stage = "blueprint_contract"
        else:
            last_error_feedback = "\n".join(result.diagnostics) or result.raw_output[-2000:]
            last_failure_stage = "lean_check"
        last_diagnostics = list(result.diagnostics) or [last_error_feedback]

        feedback = (
            f"lean_compile reported errors (attempt {attempt + 1}/{max_retries}):\n\n"
            f"{last_error_feedback}\n\n"
            "Fix the issues and call lean_compile again."
        )
        _set_latest_blueprint_retry(
            messages,
            base_messages,
            lean_code,
            feedback,
            finish_reason=last_finish_reason,
        )

    message = (
        f"Informal blueprint generation failed after {max_retries} attempts. "
        f"Last error:\n{last_error_feedback[-2000:]}"
    )
    raise BlueprintGenerationError(
        message,
        last_candidate=last_candidate,
        diagnostics=last_diagnostics,
        attempt=max_retries,
        finish_reason=last_finish_reason,
        failure_stage=last_failure_stage,
    )


def _build_user_prompt(theorem_stmt: str, nl_proof: str | None) -> str:
    return render(BLUEPRINT_USER_TEMPLATE, theorem_stmt=theorem_stmt, nl_proof=nl_proof or "")


# Matches the first line that looks like real Lean source, used to strip a
# leaked non-Lean preamble (e.g. a model hallucinating a tool-call-style tag
# like `<lean_compile>` instead of a code fence) when there's no fence to
# delimit the code block.
_LEAN_START_RE = re.compile(
    r"^\s*(?:import\b|@\[blueprint\b|theorem\b|lemma\b|noncomputable\s+def\b|def\b|abbrev\b)",
    re.MULTILINE,
)


def _extract_lean_code(content: str | None) -> str:
    """Extract the Lean code block from the LLM response."""
    content = content or ""
    fenced_blocks = [
        block.strip()
        for block in re.findall(r"```(?:lean|lean4)?\s*\n(.*?)```", content, re.DOTALL)
        if block.strip()
    ]
    if fenced_blocks:
        blueprint_blocks = [
            block for block in fenced_blocks
            if "@[blueprint" in block and re.search(r"\btheorem\b", block)
        ]
        if blueprint_blocks:
            return blueprint_blocks[-1]
        leanish_blocks = [block for block in fenced_blocks if _LEAN_START_RE.search(block)]
        if leanish_blocks:
            return leanish_blocks[-1]
        return fenced_blocks[-1]
    # No fence - the model may still have prefixed its response with
    # non-Lean text (a leaked tag, an apology, etc.). Start at the first
    # line that looks like real Lean rather than treating the raw response
    # as Lean verbatim.
    start_match = _LEAN_START_RE.search(content)
    if start_match:
        return content[start_match.start():].strip()
    return content.strip()


def _parse_blueprint(lean_code: str, target_theorem: str) -> Blueprint:
    """
    Parse @[blueprint]-annotated Lean code into a Blueprint datastructure.

    Extracts node names, kinds, statements, proof sketches, and sorry_using deps.
    """
    nodes: list[BlueprintNode] = []
    # Match @[blueprint ...] blocks followed by a declaration
    pattern = re.compile(
        rf"@\[blueprint\s*(.*?)\]\s*\n\s*({_BLUEPRINT_DECL_KW})\s+"
        rf"(\w+)(.*?)(?=@\[blueprint|\Z)",
        re.DOTALL,
    )
    for m in pattern.finditer(lean_code):
        attrs_block = m.group(1)
        kind_kw = " ".join(m.group(2).split())
        name = m.group(3)
        rest = m.group(4)

        kind = "definition" if kind_kw in ("def", "noncomputable def", "abbrev") else kind_kw

        statement = _extract_attr(attrs_block, "statement")
        proof_sketch = _extract_attr(attrs_block, "proof")
        title = _extract_string_attr(attrs_block, "title")
        source_match = re.fullmatch(
            r"(?:COT_STEP:(S\d{3}(?:\.[A-Za-z0-9_-]+)?)|COT_CLAIM:(C\d{3,}))",
            title.strip(),
        )
        source_step_id = (
            (source_match.group(1) or source_match.group(2)) if source_match else ""
        )

        # Extract sorry_using [...] dependencies
        dep_match = re.search(r"sorry_using\s*\[([^\]]*)\]", rest)
        deps = [d.strip() for d in dep_match.group(1).split(",") if d.strip()] if dep_match else []

        nodes.append(BlueprintNode(
            name=name,
            kind=kind,
            statement=statement,
            proof_sketch=proof_sketch,
            dependencies=deps,
            lean_declaration=m.group(0),
            title=title,
            source_step_id=source_step_id,
        ))

    return Blueprint(
        nodes=nodes,
        lean_file=lean_code,
        target_theorem=target_theorem,
        phase2_header=_safe_phase2_header(lean_code),
    )


def _extract_attr(attrs: str, key: str) -> str:
    match = re.search(rf"\({key}\s*:=\s*/--\s*(.*?)\s*-/\)", attrs, re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_string_attr(attrs: str, key: str) -> str:
    """Extract a Lean string-valued blueprint attribute.

    ``title`` is deliberately kept machine-readable.  JSON's string decoder
    matches the escapes used by the subset of Lean strings emitted here and
    avoids silently retaining quotes/backslashes in step identifiers.
    """
    match = re.search(rf'\({key}\s*:=\s*("(?:\\.|[^"\\])*")\)', attrs, re.DOTALL)
    if not match:
        return ""
    try:
        return str(json.loads(match.group(1)))
    except json.JSONDecodeError:
        return ""


def _extract_target_name(lean_code: str, fallback: str) -> str:
    """Extract the main theorem name from the blueprint Lean code.

    The main theorem is the last `theorem` declaration (which must equal the
    targeted identifier per the blueprint system prompt).
    """
    matches = re.findall(r"\btheorem\s+(\w+)", lean_code)
    if matches:
        return matches[-1]
    # Fallback: extract identifier from the theorem statement
    m = re.search(r"\btheorem\s+(\w+)", fallback)
    return m.group(1) if m else "main_theorem"

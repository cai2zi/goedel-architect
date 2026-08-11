"""Phase 1: Blueprint generation.

Calls the LLM with the verbatim system prompt from the paper (prompts/blueprint_system.md)
and validates the resulting @[blueprint]-annotated Lean file via LeanArchitect.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

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
from mathlib_retrieval import MathlibRetrieval
from semantic_fidelity import (
    SemanticIssue,
    format_semantic_issues,
    parse_cot_manifest,
    validate_blueprint_fidelity,
)
from goedel_prompts import load, render
from tracer import TraceEvent


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

## Immutable formal Step translation contract

The COT is partitioned losslessly into `COT_STEP` blocks. A Step is one
formalization unit, not necessarily one Lean declaration. Translate every
Step collectively and preserve wrong, contradictory, or unsupported reasoning
exactly. Never solve, repair, weaken, omit, or reorder the source COT.

Every `@[blueprint]` node, including definitions, MUST have exactly one native
LeanArchitect title naming its source Step:

    (title := "COT_STEP:S003")

One Step MAY map to several connected Definitions/Lemmas/Theorems; this is the
normal way to express its objects, conditions, intermediate facts, and result.
Each supplied Step must map to at least one node. Every node must be in the
root theorem's transitive semantic dependency closure when it genuinely
supports the final conclusion. The root must map to the final Step. A faithful
but abandoned, erroneous, or diagnostic branch may remain outside the root
closure and will be reported as a warning. Infer real dependencies from the
mathematics; never add a fake edge merely to silence that warning.

The dependency graph is extracted ONLY from identifiers inside
`sorry_using [...]`. Merely mentioning a declaration in a type, theorem
hypothesis, comment, or definition body does not create a graph edge. Thus
each definition must be consumed by a genuinely downstream proof node's
`sorry_using` list. Before emitting, trace every node that supports the final
answer forward to the root, while leaving genuinely non-final branches
independent. The binding includes the colon: `COT_STEP:S003`, never
`COT_STEP S003`.

Do not encode an asserted or derived step as an executable definition.  Do not
replace a Step with `True`, a reflexive equality, an unconstrained existential
witness, a constant `Prop := True`/`Bool := true`, or a nullary definition that
hard-codes the claimed answer.  If a source step is false or has a gap, keep
its actual proposition as a lemma with `sorry_using [...]`; proving or
diagnosing that lemma belongs to the later phase.

The following are invalid semantic evasions, even when Lean accepts them:
`def P : Prop := True`, `lemma p : P → True`, `lemma p : x = x`, and
`lemma p : ∃ x, x = answer`. Preserve object binding, inference direction,
and real dependencies. If Mathlib lacks a convenient concrete encoding, use a
typed abstract relation over the same source objects rather than a `True`
shell.

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

Coverage is clause-level, not merely Step-level. If a source Step counts
a restricted family and then asserts that the original problem's total `N`
equals that count `K`, define the original `N` once and emit the exact bridge
`lemma cot_total_jump : N = K := by sorry_using [...]`.  Do not replace it by
the weaker statement `restrictedCount = K`, and do not invent a converse or
set equivalence to justify it.  The explicit bridge is how an unsupported COT
jump is preserved for later diagnosis.  Before returning, check every
mathematical clause in each numbered step against a formal type or definition
body; prose in `statement`/`proof` fields does not satisfy this check.

Do not merge distinct clauses by strengthening one of them. For example, a
source clause `3^7 ∣ x^3 ↔ 27 ∣ x` about one variable and a separate clause
`(27 ∣ a ∧ 27 ∣ b ∧ 27 ∣ c) → 3^7 ∣ a^3+b^3+c^3` require two propositions.
The stronger aggregate statement
`3^7 ∣ a^3+b^3+c^3 ↔ (27 ∣ a ∧ 27 ∣ b ∧ 27 ∣ c)` is not a faithful substitute
because it drops the single-variable claim and invents an aggregate converse.

Before emitting Lean, perform an internal Step-by-Step coverage check. For
every computation, derived assertion, verification, or conclusion, include a
Lemma/Theorem anchor whose proposition states the source assertion itself; helper
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

The root theorem itself must perform the COT's final inference. If the COT has
a distinct final answer-restatement Step, preserve its Step mapping without
inventing new mathematics; repeated conclusions are allowed when the source
itself repeats them.
"""


_WHOLE_COT_SEMANTIC_BLUEPRINT_SYSTEM_SUFFIX = r"""

## Whole-COT faithful translation contract

The informal proof is supplied as one uninterrupted COT. Translate its
mathematical reasoning faithfully without inventing numbered source-Step
bindings or `COT_STEP` titles. Preserve wrong, contradictory, unsupported, or
abandoned reasoning instead of silently solving, repairing, weakening, or
replacing it.

The absence of source-Step bindings is not permission to omit difficult
content. Every substantive computation, asserted intermediate result, case
condition, filtering constraint, and final inference in the COT must be
represented by the formal type or body of one or more Blueprint nodes. Purely
verbal transitions and final-answer restatements need no separate node.

Do not encode an asserted or derived proposition as an executable definition.
Never replace mathematical content with `True`, a proposition ending in
`→ True`, a conjunction/disjunction containing a `True` placeholder, a
reflexive equality, an unconstrained existential witness, a constant
`Prop := True`/`Bool := true`, or a nullary definition that hard-codes the
claimed answer. If a claim is false or unsupported, state that exact claim as
a lemma with `sorry_using [...]`; proving or diagnosing it belongs to the
later phase.

Only formal Lean types and definition bodies count as semantic coverage.
Comments, docstrings, and natural-language `statement` or `proof` fields do
not encode constraints. Preserve source objects, assumptions, quantifiers,
polarity, inference direction, and dependencies. Reuse the same formal
objects through the graph rather than introducing disconnected quantities
with similar names.

When a Mathlib-native encoding is disproportionately elaborate, use a typed
abstract relational model over the same source objects. Do not use `True`,
arbitrary coordinates, additional assumptions, or unrelated witnesses as an
escape hatch. A derived equation must be a conclusion from its actual source
dependencies, not a fresh premise assumed by the node meant to derive it.

Keep every claimed quantity tied to the original problem object and givens.
The root must state the claimed answer about that modeled object. If the COT
repeats the final answer in its own Step, formalize that Step normally; an
earlier node may legitimately have the same proposition as the root.
"""


SEMANTIC_SOURCE_MODES = {"step_grounded", "whole_cot"}


def _decode_manifest_rows(value: str) -> dict:
    from cot_blueprint_refine.formal_steps import decode_formal_step_manifest
    return decode_formal_step_manifest(value)


def _render_step_grounded_proof(cot_manifest_json: str, *, include_ir: bool) -> str:
    from cot_blueprint_refine.formal_steps import decode_formal_step_manifest
    del include_ir
    manifest = decode_formal_step_manifest(cot_manifest_json)
    return "\n\n".join(
        f"[COT_STEP {step['step_id']}]\n{step['source_text']}\n[/COT_STEP {step['step_id']}]"
        for step in manifest["steps"]
    )


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


_PHASE1A_IMMUTABLE_SEMANTIC_CODES = {
    "emptyCotManifest",
    "missingRoot",
    "missingStepMapping",
    "multipleStepMappings",
    "malformedStepMapping",
    "unknownStepMapping",
    "rootNotFinalStep",
    "malformedPendingClaim",
}


def _phase1a_blocking_semantic_issues(
    issues: list[SemanticIssue],
) -> list[SemanticIssue]:
    """Keep only errors that an editable Phase-1B DAG must not inherit."""
    return [
        issue for issue in issues
        if issue.severity == "error"
        and issue.code in _PHASE1A_IMMUTABLE_SEMANTIC_CODES
    ]


def _emit_semantic_check(
    tracer,
    *,
    thm_name: str,
    phase: str,
    attempt: int,
    issues: list[SemanticIssue],
    turn: int | None = None,
    blocking_issues: list[SemanticIssue] | None = None,
) -> None:
    if tracer is None:
        return
    all_errors = [issue for issue in issues if issue.severity == "error"]
    effective_blocking = all_errors if blocking_issues is None else blocking_issues
    blocking_count = len(effective_blocking)
    warning_count = sum(issue.severity == "warning" for issue in issues)
    tracer.emit(TraceEvent(
        kind="blueprint_semantic_check",
        thm_name=thm_name,
        ok=blocking_count == 0,
        args={
            "phase": phase,
            "attempt": attempt,
            "turn": turn,
            "issue_count": len(issues),
            "error_count": len(all_errors),
            "blocking_count": blocking_count,
            "deferred_error_count": len(all_errors) - blocking_count,
            "warning_count": warning_count,
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
        candidate_history: list[str] | None = None,
        candidate_labels: list[str] | None = None,
        validation_details: dict[str, Any] | None = None,
        node_edit_rounds: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.last_candidate = last_candidate
        self.diagnostics = list(diagnostics or [])
        self.attempt = attempt
        self.finish_reason = finish_reason
        self.failure_stage = failure_stage
        self.candidate_history = list(candidate_history or [])
        self.candidate_labels = list(candidate_labels or [])
        self.validation_details = dict(validation_details or {})
        self.node_edit_rounds = list(node_edit_rounds or [])


@lru_cache(maxsize=2)
def _load_phase1_tokenizer_unlocked(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


_PHASE1_TOKENIZER_LOAD_LOCK = threading.Lock()


def _load_phase1_tokenizer(path: str):
    # Hugging Face's lazy module imports are not safe when many Phase-1 worker
    # threads perform the first import concurrently.  Serialize only cache
    # misses/reads; tokenization itself remains concurrent.
    with _PHASE1_TOKENIZER_LOAD_LOCK:
        return _load_phase1_tokenizer_unlocked(path)


def phase1_request_max_tokens(messages: list[dict[str, str]]) -> int:
    """Fit every initial/retry Blueprint completion in the model context."""
    cap = int(os.environ.get("GOEDEL_PHASE1_MAX_OUTPUT_CAP", str(_max_tokens())))
    context = int(os.environ.get("GOEDEL_PHASE1_MODEL_MAX_CONTEXT", "0"))
    tokenizer_path = os.environ.get("GOEDEL_TOKENIZER_PATH", "").strip()
    if context <= 0 or not tokenizer_path:
        return cap
    margin = int(os.environ.get("GOEDEL_PHASE1_CONTEXT_SAFETY_MARGIN", "512"))
    minimum = int(os.environ.get("GOEDEL_PHASE1_MIN_OUTPUT_TOKENS", "512"))
    tokenizer = _load_phase1_tokenizer(tokenizer_path)
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    except (TypeError, ValueError):
        # Hugging Face templates and OpenAI disagree on tool-call argument
        # representation (mapping vs JSON string).  Budgeting must never make
        # a valid OpenAI tool history fail.  Raw JSON tokenization is a stable
        # conservative fallback; the configured safety margin absorbs chat
        # framing overhead.
        serialized = json.dumps(
            messages, ensure_ascii=False, sort_keys=True, default=str,
        )
        encoded = tokenizer.encode(serialized, add_special_tokens=False)
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
    payload = []
    for tc in message.tool_calls or []:
        arguments = tc.function.arguments
        if tc.function.name == "lean_compile":
            try:
                decoded = json.loads(arguments or "{}")
                lean_code = decoded.get("lean_code") if isinstance(decoded, dict) else None
            except json.JSONDecodeError:
                lean_code = None
            if isinstance(lean_code, str):
                arguments = json.dumps({
                    "lean_code_sha256": hashlib.sha256(lean_code.encode()).hexdigest(),
                    "lean_code_chars": len(lean_code),
                }, sort_keys=True)
        payload.append({
            "id": tc.id,
            "type": getattr(tc, "type", "function"),
            "function": {
                "name": tc.function.name,
                "arguments": arguments,
            },
        })
    return payload


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
    source_contexts: list[dict[str, Any]] | None = None,
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
            "source_contexts": source_contexts or [],
        },
        ok=result.success,
    ))


def _diagnostic_line(value: Any) -> int | None:
    payload = value
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            match = re.search(r"(?:line\s+|:)(\d+)(?::\d+)?", value, re.I)
            return int(match.group(1)) if match else None
    if not isinstance(payload, dict):
        return None
    for key in ("pos", "startPos", "start", "position"):
        position = payload.get(key)
        if isinstance(position, dict) and isinstance(position.get("line"), int):
            return int(position["line"])
    return int(payload["line"]) if isinstance(payload.get("line"), int) else None


def _lean_source_contexts(
    result: CompilerResult,
    blueprint: Blueprint,
    manifest,
) -> list[dict[str, Any]]:
    if manifest is None:
        return []
    contexts: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    # Warnings (especially one `declaration uses sorry` per Blueprint node)
    # are not repair targets and can multiply one source Step dozens of times.
    for diagnostic in result.errors:
        line = _diagnostic_line(diagnostic)
        if line is None:
            continue
        node = next((
            item for item in blueprint.nodes
            if item.lean_start_line <= line <= item.lean_end_line
        ), None)
        if node is None or not node.source_step_id:
            continue
        step = manifest.by_id.get(node.source_step_id.split(".", 1)[0])
        if step is None or (line, node.name) in seen:
            continue
        seen.add((line, node.name))
        contexts.append({
            "lean_line": line, "node_name": node.name, "step_id": step.step_id,
            "source_start": step.source_start, "source_end": step.source_end,
            "source_text": step.source_text, "source_sha256": step.source_sha256,
        })
    return contexts


def _format_lean_source_contexts(contexts: list[dict[str, Any]]) -> str:
    if not contexts:
        return ""
    lines = ["\nLean diagnostics mapped to source Steps (look up the Step text in the COT above):"]
    for item in contexts:
        lines.append(
            f"- line {item['lean_line']} / {item['node_name']} / {item['step_id']} "
            f"[{item['source_start']}:{item['source_end']}]"
        )
    return "\n".join(lines)


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
    if "stepMappingAbsent" in codes:
        guidance.append(
            "Create one or more substantive formal nodes for every absent source Step, "
            "using its exact `COT_STEP:SNNN` title, and then "
            "include it in a downstream `sorry_using` chain to the root; never replace "
            "the missing translation with `True`, a reflexive equality, or a placeholder."
        )
    if codes.intersection({
        "missingStepMapping", "malformedStepMapping",
    }):
        guidance.append(
            "Repair each listed node's provenance title so it names exactly the source "
            "Step that its existing formal declaration translates."
        )
    if codes.intersection({"stepNotRootReachable", "nodeNotRootReachable"}):
        guidance.append(
            "The corresponding formal nodes already exist: preserve their declarations "
            "and build mathematically faithful downstream dependencies to the root. Do not "
            "attach unrelated nodes directly to the root merely to satisfy reachability."
        )
    if any(code.startswith("vacuous") for code in codes):
        guidance.append(
            "Replace every vacuous type/body with the exact source proposition over the "
            "same shared objects, then connect that substantive node to the root."
        )
    if any(
        code.startswith("unconstrainedExists")
        or code.startswith("unboundAnswerWitness")
        for code in codes
    ):
        guidance.append(
            "Model the original constrained object and givens explicitly; the root must "
            "answer the original question, not merely choose a closed witness."
        )
    if "rootMissingClaimedAnswer" in codes:
        guidance.append(
            "Keep the original COT's claimed answer literally in the root proposition "
            "about the original modeled object."
        )
    if codes.intersection({"reflexiveStep", "reflexiveRoot"}):
        guidance.append(
            "A reflexive equality such as `c = c` does not encode the source inference. "
            "Bind the source quantity/event/function and state that modeled expression "
            "equals the claimed value; do not merely rewrite the claimed value twice."
        )
    if not guidance:
        guidance.append(
            "Fix every listed issue while preserving every substantive source Step clause."
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


_PHASE1_LEAN_COMPILE_TOOL = {
    "type": "function",
    "function": {
        "name": "lean_compile",
        "description": "Compile and structurally validate one complete Lean Blueprint file.",
        "parameters": {
            "type": "object",
            "properties": {"lean_code": {"type": "string"}},
            "required": ["lean_code"],
            "additionalProperties": False,
        },
    },
}
_PHASE1_MATHLIB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "mathlib_search",
        "description": "Search Mathlib for existing names and exact theorem/type signatures.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

_PENDING_HELPER = 'def PendingBlueprintClaim (_nodeId : String) : Prop := True'
_PENDING_HELPER_RE = re.compile(
    r"(?m)^\s*def\s+PendingBlueprintClaim\s*\(_nodeId\s*:\s*String\)\s*"
    r":\s*Prop\s*:=\s*True\s*$"
)


def _strip_lean_comments(source: str) -> str:
    """Remove nested Lean comments while preserving line boundaries."""
    result: list[str] = []
    index = 0
    block_depth = 0
    in_string = False
    while index < len(source):
        pair = source[index:index + 2]
        char = source[index]
        if block_depth:
            if pair == "/-":
                block_depth += 1
                index += 2
            elif pair == "-/":
                block_depth -= 1
                index += 2
            else:
                if char == "\n":
                    result.append(char)
                index += 1
            continue
        if not in_string and pair == "/-":
            block_depth = 1
            index += 2
            continue
        if not in_string and pair == "--":
            newline = source.find("\n", index + 2)
            if newline < 0:
                break
            result.append("\n")
            index = newline + 1
            continue
        result.append(char)
        if char == '"' and (index == 0 or source[index - 1] != "\\"):
            in_string = not in_string
        index += 1
    return "".join(result)


def _unannotated_local_declaration_errors(blueprint: Blueprint) -> list[str]:
    """Reject source commands that Phase 1B's safe-header rebuild would drop."""
    residue = blueprint.lean_file
    for node in blueprint.nodes:
        residue = residue.replace(node.lean_declaration, "", 1)
    residue = _PENDING_HELPER_RE.sub("", residue, count=1)
    residue = _strip_lean_comments(residue)
    unexpected: list[str] = []
    for raw_line in residue.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^(?:import|open(?:\s+scoped)?|set_option)\b", line):
            continue
        unexpected.append(line)
    if not unexpected:
        return []
    preview = "; ".join(unexpected[:8])
    suffix = "" if len(unexpected) <= 8 else f"; ... and {len(unexpected) - 8} more lines"
    return [
        "unannotatedLocalDeclaration: Phase 1B preserves only imports/options and "
        "@[blueprint] declarations. Convert every local declaration to a Blueprint "
        f"node or remove it. Unexpected source: {preview}{suffix}"
    ]


def _pending_node_names(blueprint: Blueprint) -> tuple[str, ...]:
    return tuple(
        node.name for node in blueprint.nodes
        if node.kind in {"lemma", "theorem"}
        and "PendingBlueprintClaim" in node.signature()
    )


def _emit_pending_summary(
    tracer,
    *,
    thm_name: str,
    phase: str,
    round_index: int,
    blueprint: Blueprint,
    initial_names: Sequence[str] = (),
    previous_names: Sequence[str] = (),
) -> None:
    if tracer is None:
        return
    current = set(_pending_node_names(blueprint))
    initial = set(initial_names)
    previous = set(previous_names)
    tracer.emit(TraceEvent(
        kind="phase1PendingSummary",
        thm_name=thm_name,
        turn=round_index,
        args={
            "phase": phase,
            "round": round_index,
            "phase1ARootPending": (
                blueprint.target_theorem in current if phase == "phase1A" else False
            ),
            "initialPendingNodeCount": len(initial) if initial_names else len(current),
            "pendingNodeCount": len(current),
            "pendingNodes": sorted(current),
            "resolvedPendingNodes": sorted(previous - current),
            "resolvedPendingNodeCount": len(previous - current),
            "finalPendingNodeCount": len(current) if phase == "phase1BFinal" else None,
        },
        ok=not current if phase == "phase1BFinal" else True,
    ))

_PHASE1A_SKELETON_SUFFIX = r"""

## Phase 1A: compilable Blueprint skeleton

Emit a complete Lean Blueprint skeleton. Immediately after the imports emit
this exact unannotated helper declaration once:

    def PendingBlueprintClaim (_nodeId : String) : Prop := True

This helper is the one deliberate exception to the earlier rule against plain
top-level helper declarations. Do not annotate it with `@[blueprint]`.

Apart from that canonical helper, emit no plain local declaration, `variable`,
`section`, `namespace`, `axiom`, or `partial def`. Every locally introduced
object required by the Blueprint must itself be an `@[blueprint]` definition.
Imported Mathlib constants may be used directly in formal types and definition
bodies. Every name inside `sorry_using [...]` must be the name of an existing
`@[blueprint]` node; never list an ordinary definition or Mathlib declaration.

Every Definition must already have a real, non-vacuous body. Any
Lemma/Theorem, including the root theorem, may either state its concrete Lean proposition immediately or,
when that proposition is not yet reliable, temporarily use exactly
`PendingBlueprintClaim "declaration_name"` as its complete conclusion. Its
proof body must still be `by sorry_using [...]` with the real DAG dependencies.
When the root is pending, its statement metadata must still describe the exact
source conclusion and claimed answer, and its dependency list must represent
the intended final inference. Phase 1B must replace every pending conclusion.

For every node, write a concise, non-empty natural-language `statement` that
identifies the source objects, relevant assumptions, and exact intended
mathematical content. For every Lemma/Theorem, also write a concise, non-empty
natural-language `proof` explaining the inference from its declared parents.
No fixed prose headings are required. Put typed variables and assumptions in
Lean binders as far as possible. A typical non-root proof node is:

    @[blueprint (title := "COT_STEP:S003")
      (statement := /-- For `x : Nat` satisfying `h : P x`, state the exact
        mathematical conclusion represented by this source step. -/)
      (proof := /-- Obtain the conclusion from `parent_node` while preserving
        the source inference direction. -/)]
    lemma node_name (x : Nat) (h : P x) :
        PendingBlueprintClaim "node_name" := by
      sorry_using [parent_node]

The prose describes the intended interface. Phase 1B edits individual nodes and
may add/delete nodes or repair source-Step titles and dependencies, but a useful
initial DAG reduces later edits. Before calling `lean_compile`, check that every node has a
meaningful statement and every proof node has a meaningful proof sketch. A
node that already has a faithful, type-correct concrete proposition does not
need to remain pending.

Prefer one shared typed model for source objects and reuse it through later
Steps and the root.  For geometry, counting, or probability problems whose
Mathlib-native representation is disproportionately expensive, define a small
typed structure or abstract relation for the original configuration, a setup
predicate for exactly the source givens, and quantity functions over that same
model.  Later lemmas and the root must bind the same model; do not reopen fresh
existential witnesses for the same points, arcs, sets, probabilities, or
counts in every Step.  A COT-derived numeric value belongs in a lemma/theorem
conclusion, not a local literal definition that makes the claim reflexive.
"""

@dataclass
class _Phase1ToolSessionResult:
    lean_code: str = ""
    successful_lean_code: str = ""
    finish_reason: str | None = None
    turns: int = 0


def _phase1_tool_output(
    result: CompilerResult,
    semantic_issues: list[SemanticIssue] | None = None,
    lean_source_contexts: list[dict[str, Any]] | None = None,
    contract_errors: list[str] | None = None,
    blocking_semantic_issues: list[SemanticIssue] | None = None,
) -> tuple[str, bool]:
    """Combine Lean and deterministic semantic validation for one tool call."""
    issues = list(semantic_issues or [])
    errors = [issue for issue in issues if issue.severity == "error"]
    blocking_errors = (
        list(blocking_semantic_issues)
        if blocking_semantic_issues is not None
        else errors
    )
    structural_errors = list(contract_errors or [])
    sections: list[str] = []
    if result.success:
        sections.append("Lean compilation and structural validation SUCCESSFUL.")
    else:
        diagnostics = "\n".join(result.diagnostics) or result.raw_output[-8000:]
        sections.append(
            "Lean compilation or structural validation FAILED.\n"
            f"{diagnostics}{_format_lean_source_contexts(lean_source_contexts or [])}"
        )
    if issues:
        semantic_status = (
            "FAILED"
            if blocking_errors
            else "PASSED WITH DEFERRED NODE REPAIRS"
            if errors
            else "PASSED WITH WARNINGS"
        )
        sections.append(
            f"Deterministic semantic-fidelity validation {semantic_status}.\n"
            f"{format_semantic_issues(issues)}"
        )
        if errors and not blocking_errors:
            sections.append(
                "These content-level semantic errors are attached to replaceable nodes "
                "and will be repaired in Phase 1B; do not regenerate the complete Blueprint "
                "solely for these diagnostics."
            )
    if structural_errors:
        sections.append(
            "Phase 1A skeleton contract FAILED.\n"
            + format_phase2_contract_errors(structural_errors, limit=100)
        )
    if blocking_errors:
        sections.append(
            "Issue-specific repair rules:\n"
            f"{_semantic_repair_guidance(blocking_errors)}\n\n"
            "Repair every blocking issue and call lean_compile with the complete file again. "
            "Use each Step ID to look up its source text in the COT already present above."
        )
    return (
        "\n\n".join(sections),
        result.success and not blocking_errors and not structural_errors,
    )


def _run_phase1_tool_session(
    client,
    model: str,
    base_messages: tuple[dict[str, Any], ...],
    *,
    compiler: KiminaLeanCompiler,
    target_name: str,
    retrieval: MathlibRetrieval,
    tracer,
    thm_name: str,
    attempt: int,
    max_tool_turns: int,
    max_tool_calls_per_turn: int,
    mathlib_search_max_calls: int,
    tool_cache: dict[str, tuple[str, bool]],
    search_state: dict[str, int],
    semantic_manifest=None,
    claimed_answer: str = "",
    semantic_fidelity_enabled: bool = False,
    semantic_require_step_ids: bool = False,
    semantic_static_gate: bool = False,
    allow_pending_claims: bool = False,
    trace_phase: str = "phase1A",
) -> _Phase1ToolSessionResult:
    """Run one bounded Phase-1 tool conversation with compact rolling history."""
    messages = [dict(message) for message in base_messages]
    result = _Phase1ToolSessionResult()
    for turn in range(1, max_tool_turns + 1):
        final_turn = turn == max_tool_turns
        tools = [_PHASE1_LEAN_COMPILE_TOOL] if final_turn else [
            _PHASE1_LEAN_COMPILE_TOOL, _PHASE1_MATHLIB_SEARCH_TOOL,
        ]
        response = chat_completion_with_retry(
            client,
            tracer=tracer,
            thm_name=thm_name,
            phase=trace_phase,
            model_id=model,
            operation="blueprint_tool_final" if final_turn else "blueprint_tool",
            trace_args={"attempt": attempt, "turn": turn, "final_turn": final_turn},
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="required",
            parallel_tool_calls=not final_turn,
            max_completion_tokens=phase1_request_max_tokens(messages),
            **_reasoning_kwargs(model),
        )
        _emit_usage(tracer, thm_name, trace_phase, model, response)
        _emit_llm_response(
            tracer, thm_name=thm_name, phase=trace_phase, model=model,
            response=response, attempt=attempt, turn=turn,
        )
        choice = response.choices[0]
        result.finish_reason = getattr(choice, "finish_reason", None)
        result.turns = turn
        message = choice.message
        selected: list[tuple[Any, str, dict[str, Any], str]] = []
        dropped: list[dict[str, Any]] = []
        seen: set[str] = set()
        compile_count = 0
        for index, call in enumerate(message.tool_calls or []):
            name = str(call.function.name)
            try:
                args = json.loads(call.function.arguments or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments must be an object")
            except (json.JSONDecodeError, ValueError) as exc:
                dropped.append({"index": index, "reason": "invalidArguments", "detail": str(exc)})
                continue
            call_hash = hashlib.sha256(json.dumps(
                {"name": name, "args": args}, sort_keys=True,
                ensure_ascii=False, separators=(",", ":"),
            ).encode()).hexdigest()
            reason = ""
            if name not in {"lean_compile", "mathlib_search"}:
                reason = "notAllowed"
            elif final_turn and name != "lean_compile":
                reason = "notAllowedOnFinalTurn"
            elif call_hash in seen:
                reason = "duplicateInTurn"
            elif len(selected) >= max_tool_calls_per_turn:
                reason = "overTurnLimit"
            elif name == "lean_compile" and compile_count >= 1:
                reason = "compileLimit"
            elif name == "lean_compile" and not isinstance(args.get("lean_code"), str):
                reason = "invalidArguments"
            elif name == "mathlib_search" and not str(args.get("query") or "").strip():
                reason = "invalidArguments"
            elif name == "mathlib_search" and search_state["count"] >= mathlib_search_max_calls:
                reason = "searchBudgetExhausted"
            if reason:
                dropped.append({"index": index, "name": name, "reason": reason, "hash": call_hash})
                continue
            seen.add(call_hash)
            compile_count += int(name == "lean_compile")
            if name == "mathlib_search":
                search_state["count"] += 1
            selected.append((call, name, args, call_hash))
        if tracer is not None and dropped:
            tracer.emit(TraceEvent(
                kind="tool_calls_dropped", thm_name=thm_name, turn=turn,
                args={"phase": trace_phase, "attempt": attempt, "calls": dropped},
            ))

        assistant_payload = {
            "role": "assistant", "content": message.content or "",
            "tool_calls": [call.model_dump() for call, _name, _args, _hash in selected],
        }
        tool_messages: list[dict[str, Any]] = []
        for call, name, args, call_hash in selected:
            span_id = uuid.uuid4().hex
            started_ns = time.monotonic_ns()
            if tracer is not None:
                trace_arguments = args
                if name == "lean_compile":
                    lean_code = str(args.get("lean_code") or "")
                    trace_arguments = {
                        "lean_code_sha256": hashlib.sha256(lean_code.encode()).hexdigest(),
                        "lean_code_chars": len(lean_code),
                    }
                tracer.emit(TraceEvent(
                    kind="tool_call", thm_name=thm_name, turn=turn,
                    call_id=call.id, tool_name=name, span_id=span_id,
                    args={"phase": trace_phase, "attempt": attempt,
                          "arguments": trace_arguments, "hash": call_hash},
                ))
            semantic_issues: list[SemanticIssue] = []
            phase1a_blocking_semantic_issues: list[SemanticIssue] | None = None
            phase1a_contract_issues: list[str] = []
            parsed_candidate: Blueprint | None = None
            if name == "lean_compile":
                lean_code = str(args["lean_code"])
                parsed_candidate = _parse_blueprint(lean_code, target_name)
                if trace_phase == "phase1A":
                    phase1a_contract_issues = _phase1a_contract_errors(parsed_candidate)
                    if not parsed_candidate.nodes:
                        phase1a_contract_issues.append(
                            "no_blueprint_nodes: no annotated declarations were parsed."
                        )
                if semantic_fidelity_enabled and parsed_candidate.nodes:
                    semantic_issues = _enabled_semantic_issues(
                        validate_blueprint_fidelity(
                            parsed_candidate,
                            semantic_manifest,
                            claimed_answer=claimed_answer,
                            require_step_bindings=semantic_require_step_ids,
                            allow_pending_claims=allow_pending_claims,
                        ),
                        require_step_ids=semantic_require_step_ids,
                        static_gate=semantic_static_gate,
                    )
                    if trace_phase == "phase1A":
                        phase1a_blocking_semantic_issues = (
                            _phase1a_blocking_semantic_issues(semantic_issues)
                        )
                    _emit_semantic_check(
                        tracer,
                        thm_name=thm_name,
                        phase=trace_phase,
                        attempt=attempt,
                        turn=turn,
                        issues=semantic_issues,
                        blocking_issues=phase1a_blocking_semantic_issues,
                    )

            cache_hit = call_hash in tool_cache
            if cache_hit:
                output, ok = tool_cache[call_hash]
            elif name == "lean_compile":
                lean_code = str(args["lean_code"])
                result.lean_code = lean_code
                compile_result = compiler.check_blueprint(lean_code, target_name)
                if compile_result.failure_kind == "infra":
                    raise KiminaInfrastructureError(
                        "\n".join(compile_result.diagnostics) or compile_result.raw_output[-2000:]
                    )
                lean_contexts = _lean_source_contexts(
                    compile_result,
                    parsed_candidate or _parse_blueprint(lean_code, target_name),
                    semantic_manifest,
                )
                output, ok = _phase1_tool_output(
                    compile_result,
                    semantic_issues,
                    lean_contexts,
                    phase1a_contract_issues,
                    phase1a_blocking_semantic_issues,
                )
                tool_cache[call_hash] = (output, ok)
            else:
                hits = retrieval.search(str(args["query"]), min(10, int(args.get("k", 5))))
                output = "\n\n".join(hit.format() for hit in hits) or "No Mathlib results."
                ok = True
                tool_cache[call_hash] = (output, ok)
            if name == "lean_compile":
                result.lean_code = str(args["lean_code"])
                if ok:
                    result.successful_lean_code = result.lean_code
            if tracer is not None:
                tracer.emit(TraceEvent(
                    kind="tool_result", thm_name=thm_name, turn=turn,
                    call_id=call.id, tool_name=name, span_id=span_id,
                    result=output, ok=ok,
                    args={"phase": trace_phase, "attempt": attempt,
                          "hash": call_hash, "cache_hit": cache_hit},
                    duration_ms=(time.monotonic_ns() - started_ns) / 1_000_000,
                ))
            tool_messages.append({
                "role": "tool", "tool_call_id": call.id, "content": output,
            })
        if result.successful_lean_code:
            return result
        # Keep only the immutable prompt and the latest complete tool exchange.
        messages = [dict(item) for item in base_messages]
        if selected:
            messages.extend([assistant_payload, *tool_messages])
    return result


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
    lean_start_line: int = 0
    lean_end_line: int = 0

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
    candidate_history: list[str] = field(default_factory=list, repr=False)
    candidate_labels: list[str] = field(default_factory=list, repr=False)
    phase1b_validation: dict = field(default_factory=dict, repr=False)
    phase1b_edit_history: list[dict] = field(default_factory=list, repr=False)

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


@dataclass(frozen=True)
class _Phase2PreflightCase:
    node_name: str
    lean_code: str
    code_hash: str
    line_ranges: tuple[tuple[int, int, str], ...]


@dataclass(frozen=True)
class Phase2StandaloneIssue:
    code: str
    node_name: str
    error_kind: str
    identifiers: tuple[str, ...]
    diagnostic: str
    preflight_hash: str
    origin_declaration: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "nodeName": self.node_name,
            "errorKind": self.error_kind,
            "identifiers": list(self.identifiers),
            "diagnostic": self.diagnostic,
            "preflightHash": self.preflight_hash,
            "originDeclaration": self.origin_declaration,
        }


@dataclass(frozen=True)
class Phase2StandaloneReport:
    issues: tuple[Phase2StandaloneIssue, ...]
    checked_node_count: int
    cached_node_count: int
    skipped_pending_node_count: int
    duration_ms: float
    not_run_reason: str = ""

    @property
    def failed_nodes(self) -> list[str]:
        return [issue.node_name for issue in self.issues]


def _phase2_preflight_case(blueprint: Blueprint, node: BlueprintNode) -> _Phase2PreflightCase:
    entries: list[tuple[str, str]] = [("<phase2Header>", blueprint.phase2_header.rstrip())]
    ancestor_deps = _transitive_node_deps(node, blueprint)
    included_proof_nodes = [
        dep_node for dep_node in blueprint.dependency_order()
        if dep_node.kind != "definition" and dep_node.name in ancestor_deps
    ]
    if any(
        "PendingBlueprintClaim" in candidate.signature()
        for candidate in [*included_proof_nodes, node]
    ):
        entries.append(("<pendingHelper>", _PENDING_HELPER))
    entries.extend(
        (definition.name, definition.full_declaration())
        for definition in blueprint.nodes
        if definition.kind == "definition" and definition.name != node.name
    )
    entries.extend(
        (dep_node.name, dep_node.full_declaration())
        for dep_node in included_proof_nodes
    )
    entries.append((node.name, node.full_declaration()))

    rendered = ""
    ranges: list[tuple[int, int, str]] = []
    for origin, raw_text in entries:
        text = raw_text.strip()
        if not text:
            continue
        if rendered:
            rendered += "\n\n"
        start_line = rendered.count("\n") + 1
        rendered += text
        end_line = rendered.count("\n") + 1
        ranges.append((start_line, end_line, origin))
    rendered += "\n"
    return _Phase2PreflightCase(
        node_name=node.name,
        lean_code=rendered,
        code_hash=hashlib.sha256(rendered.encode()).hexdigest(),
        line_ranges=tuple(ranges),
    )


def _phase2_preflight_file(blueprint: Blueprint, node: BlueprintNode) -> str:
    return _phase2_preflight_case(blueprint, node).lean_code


def _standalone_error_kind(message: str) -> str:
    if re.search(r"Unknown identifier", message, re.I):
        return "unknownIdentifier"
    if re.search(r"Unknown constant", message, re.I):
        return "unknownConstant"
    if re.search(r"(?:application )?type mismatch|Invalid field notation", message, re.I):
        return "typeMismatch"
    if re.search(r"failed to synthesize", message, re.I):
        return "synthesisFailure"
    if re.search(r"unexpected token|unexpected end", message, re.I):
        return "syntaxError"
    return "leanCompileError"


def _standalone_identifiers(message: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(re.findall(
        r"Unknown (?:identifier|constant) `([^`]+)`", message,
    )))


def _preflight_origin(case: _Phase2PreflightCase, result: CompilerResult) -> str:
    for diagnostic in result.errors:
        line = _diagnostic_line(diagnostic)
        if line is None:
            continue
        for start, end, origin in case.line_ranges:
            if start <= line <= end:
                return origin
    return ""


def _emit_standalone_report(
    tracer,
    *,
    thm_name: str,
    round_index: int,
    report: Phase2StandaloneReport,
    phase: str = "phase1B",
) -> None:
    if tracer is None:
        return
    error_counts = dict(Counter(issue.error_kind for issue in report.issues))
    tracer.emit(TraceEvent(
        kind="phase2StandaloneCheckEnd",
        thm_name=thm_name,
        turn=round_index,
        args={
            "phase": phase,
            "round": round_index,
            "checkedNodeCount": report.checked_node_count,
            "cachedNodeCount": report.cached_node_count,
            "skippedPendingNodeCount": report.skipped_pending_node_count,
            "failedNodeCount": len(report.issues),
            "errorCounts": error_counts,
            "failedNodes": report.failed_nodes,
            "notRunReason": report.not_run_reason,
        },
        ok=not report.issues and not report.not_run_reason,
        duration_ms=report.duration_ms,
    ))


def phase2_standalone_contract_report(
    blueprint: Blueprint,
    compiler: KiminaLeanCompiler,
    *,
    concurrency: int = 1,
    skip_pending: bool = False,
    cache: dict[str, CompilerResult] | None = None,
    tracer=None,
    thm_name: str = "",
    round_index: int = 0,
    trace_phase: str = "phase1B",
) -> Phase2StandaloneReport:
    """Compile proof nodes exactly as Phase 2 will assemble them."""
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    started_ns = time.monotonic_ns()
    proof_nodes = [node for node in blueprint.nodes if node.kind in {"lemma", "theorem"}]
    skipped = [
        node for node in proof_nodes
        if skip_pending and "PendingBlueprintClaim" in node.lean_declaration
    ]
    checked_nodes = [node for node in proof_nodes if node not in skipped]
    cases = [_phase2_preflight_case(blueprint, node) for node in checked_nodes]
    result_cache = cache if cache is not None else {}
    uncached = [case for case in cases if case.code_hash not in result_cache]

    if tracer is not None:
        tracer.emit(TraceEvent(
            kind="phase2StandaloneCheckStart",
            thm_name=thm_name,
            turn=round_index,
            args={
                "phase": trace_phase, "round": round_index,
                "checkedNodeCount": len(cases),
                "cachedNodeCount": len(cases) - len(uncached),
                "skippedPendingNodeCount": len(skipped),
            },
        ))

    if uncached:
        results = compiler.check_many(
            [
                CompileRequest(
                    case.lean_code,
                    allow_sorry=True,
                    request_id=f"phase2-contract-{round_index}-{index}-{case.node_name}",
                )
                for index, case in enumerate(uncached)
            ],
            batch_concurrency=concurrency,
        )
        for case, result in zip(uncached, results, strict=True):
            result_cache[case.code_hash] = result

    issues: list[Phase2StandaloneIssue] = []
    cached_count = len(cases) - len(uncached)
    for case in cases:
        result = result_cache[case.code_hash]
        if result.failure_kind == "infra":
            message = "\n".join(result.diagnostics) or result.raw_output[-2000:]
            raise KiminaInfrastructureError(message)
        if result.success:
            issue = None
        else:
            message = "\n".join(result.diagnostics) or result.raw_output[-4000:]
            issue = Phase2StandaloneIssue(
                code="phase2StandaloneFailed",
                node_name=case.node_name,
                error_kind=_standalone_error_kind(message),
                identifiers=_standalone_identifiers(message),
                diagnostic=message[-4000:],
                preflight_hash=case.code_hash,
                origin_declaration=_preflight_origin(case, result),
            )
            issues.append(issue)
        if tracer is not None:
            tracer.emit(TraceEvent(
                kind="phase2StandaloneNodeResult",
                thm_name=thm_name,
                turn=round_index,
                args={
                    "phase": trace_phase, "round": round_index,
                    "nodeName": case.node_name,
                    "preflightHash": case.code_hash,
                    "cacheHit": case.code_hash not in {item.code_hash for item in uncached},
                    "issue": issue.to_dict() if issue is not None else None,
                },
                ok=issue is None,
            ))

    report = Phase2StandaloneReport(
        issues=tuple(issues),
        checked_node_count=len(cases),
        cached_node_count=cached_count,
        skipped_pending_node_count=len(skipped),
        duration_ms=(time.monotonic_ns() - started_ns) / 1_000_000,
    )
    _emit_standalone_report(
        tracer, thm_name=thm_name, round_index=round_index, report=report,
        phase=trace_phase,
    )
    return report


def phase2_standalone_contract_errors(
    blueprint: Blueprint,
    compiler: KiminaLeanCompiler,
    *,
    limit: int = 0,
    concurrency: int = 1,
) -> list[str]:
    """Compile proof nodes as Phase 2 would see them before accepting a blueprint."""
    report = phase2_standalone_contract_report(
        blueprint, compiler, concurrency=concurrency,
    )
    errors = [
        f"phase2StandaloneFailed: node `{issue.node_name}` does not compile when "
        f"assembled as a standalone Phase 2 goal; errorKind={issue.error_kind}; "
        f"identifiers={list(issue.identifiers)}; origin={issue.origin_declaration or '<unknown>'}.\n"
        f"{issue.diagnostic}"
        for issue in report.issues
    ]
    if limit:
        errors = errors[:limit]
    return errors


def format_phase2_standalone_issues(
    issues: list[Phase2StandaloneIssue] | tuple[Phase2StandaloneIssue, ...],
) -> str:
    """Group repeated standalone failures without dropping affected nodes."""
    groups: dict[tuple[str, tuple[str, ...]], list[Phase2StandaloneIssue]] = {}
    for issue in issues:
        key = (issue.error_kind, issue.identifiers)
        groups.setdefault(key, []).append(issue)
    sections: list[str] = []
    for (error_kind, identifiers), grouped in groups.items():
        nodes = ", ".join(item.node_name for item in grouped)
        origins = ", ".join(dict.fromkeys(
            item.origin_declaration or "<unknown>" for item in grouped
        ))
        representative = grouped[0].diagnostic
        sections.append(
            f"- {error_kind}: identifiers={list(identifiers) or ['<none>']} "
            f"origins={origins}\n"
            f"  Affected nodes: {nodes}\n"
            f"  Representative diagnostic: {representative}"
        )
    return "\n".join(sections)


def phase2_contract_errors(blueprint: Blueprint) -> list[str]:
    """Return structural errors that would make Phase 2 node proving invalid."""
    errors: list[str] = []
    node_names = set(blueprint.nodes_by_name())
    for node in blueprint.nodes:
        unknown = [name for name in node.dependencies if name not in node_names]
        if unknown:
            errors.append(
                f"nonBlueprintDependency: node `{node.name}` lists names that are not "
                f"@[blueprint] nodes: {unknown}."
            )
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


def _node_hash(node: BlueprintNode) -> str:
    return hashlib.sha256(node.lean_declaration.strip().encode()).hexdigest()


def _pending_helper_errors(lean_code: str) -> list[str]:
    matches = list(_PENDING_HELPER_RE.finditer(lean_code))
    named = len(re.findall(r"\bdef\s+PendingBlueprintClaim\b", lean_code))
    errors: list[str] = []
    if len(matches) != 1 or named != 1:
        errors.append(
            "pending_helper_contract: emit exactly one unannotated canonical declaration "
            f"`{_PENDING_HELPER}`; canonical={len(matches)} named={named}."
        )
    if any("@[blueprint" in lean_code[max(0, match.start() - 200):match.start()]
           and lean_code.rfind("@[blueprint", 0, match.start())
           > lean_code.rfind("\n\n", 0, match.start()) for match in matches):
        errors.append("pending_helper_annotated: PendingBlueprintClaim must not be a Blueprint node.")
    return errors


def _phase1a_contract_errors(blueprint: Blueprint) -> list[str]:
    errors = [
        *_pending_helper_errors(blueprint.lean_file),
        *_unannotated_local_declaration_errors(blueprint),
        *phase2_contract_errors(blueprint),
    ]
    for node in blueprint.nodes:
        if node.kind == "definition" and "PendingBlueprintClaim" in node.lean_declaration:
            errors.append(
                f"pendingDefinition: definition `{node.name}` must have a concrete body; "
                "PendingBlueprintClaim is only valid as the complete conclusion of a "
                "lemma/theorem."
            )
        if not node.statement.strip():
            errors.append(
                f"missing_statement_metadata: node `{node.name}` must have a non-empty "
                "`statement := /-- ... -/` annotation."
            )
        if node.kind not in {"lemma", "theorem"}:
            continue
        if not node.proof_sketch.strip():
            errors.append(
                f"missing_proof_metadata: proof node `{node.name}` must have a non-empty "
                "`proof := /-- ... -/` annotation."
            )
    return errors


def _strip_pending_helper(lean_code: str) -> str:
    return _PENDING_HELPER_RE.sub("", lean_code, count=1).replace("\n\n\n", "\n\n").strip() + "\n"


@dataclass(frozen=True)
class _BlueprintNodeEdit:
    action: str
    node_name: str
    replacement: str = ""
    revised_node: BlueprintNode | None = None


def _validate_phase1b_proof_body(node: BlueprintNode) -> str:
    if node.kind not in {"lemma", "theorem"}:
        return ""
    declaration = strip_blueprint_attr(node.lean_declaration).strip()
    proof_matches = list(BLUEPRINT_PROOF_RE.finditer(declaration))
    if len(proof_matches) != 1 or declaration[proof_matches[0].end():].strip():
        return "proofBodyMustBeSorryUsingOnly"
    return ""


def _validate_node_edit(
    current: Blueprint,
    *,
    action: str,
    node_name: str,
    expected_hash: str,
    replacement: str,
) -> tuple[_BlueprintNodeEdit | None, str]:
    if action not in {"add", "replace", "delete"}:
        return None, "unknownAction"
    node = current.node_by_name(node_name)
    if action == "add":
        if node is not None:
            return None, "nodeAlreadyExists"
        if expected_hash:
            return None, "addExpectedHashMustBeEmpty"
    else:
        if node is None:
            return None, "unknownNode"
        if expected_hash != _node_hash(node):
            return None, "staleNodeHash"
        if node.name == current.target_theorem and action == "delete":
            return None, "rootMutationNotAllowed"

    if action == "delete":
        if replacement.strip():
            return None, "deleteReplacementMustBeEmpty"
        return _BlueprintNodeEdit(action, node_name), ""

    replacement = _extract_lean_code(replacement).strip()
    if not replacement or re.search(r"(?m)^\s*(?:import|namespace|end)\b", replacement):
        return None, "replacementMustBeOneDeclaration"
    if action == "replace" and node is not None and replacement == node.lean_declaration.strip():
        return None, "identicalReplacement"
    parsed = _parse_blueprint(replacement, current.target_theorem)
    if len(parsed.nodes) != 1:
        return None, "replacementMustBeOneBlueprintNode"
    revised = parsed.nodes[0]
    if revised.name != node_name:
        return None, "nodeNameMismatch"
    if revised.name == current.target_theorem:
        if action != "replace" or node is None:
            return None, "rootMutationNotAllowed"
        if revised.kind != "theorem" or revised.title != node.title:
            return None, "rootMutationNotAllowed"
    if not revised.statement.strip():
        return None, "missingStatementMetadata"
    if revised.kind in {"lemma", "theorem"} and not revised.proof_sketch.strip():
        return None, "missingProofMetadata"
    proof_error = _validate_phase1b_proof_body(revised)
    if proof_error:
        return None, proof_error
    return _BlueprintNodeEdit(action, node_name, replacement, revised), ""


def _render_edited_blueprint(current: Blueprint, nodes: list[BlueprintNode]) -> Blueprint:
    names = [node.name for node in nodes]
    if len(names) != len(set(names)):
        raise ValueError("duplicateNodeName")
    node_map = {node.name: node for node in nodes}
    root = node_map.get(current.target_theorem)
    if root is None or root.kind != "theorem":
        raise ValueError("missingOrInvalidRoot")
    for node in nodes:
        unknown = [dependency for dependency in node.dependencies if dependency not in node_map]
        if unknown:
            raise ValueError(
                f"unknownDependencies:{node.name}:{','.join(unknown)}"
            )

    draft = Blueprint(
        nodes=nodes,
        lean_file="",
        target_theorem=current.target_theorem,
        phase2_header=current.phase2_header,
    )
    ordered = draft.dependency_order()
    definitions = [node for node in nodes if node.kind == "definition"]
    proofs = [
        node for node in ordered
        if node.kind != "definition" and node.name != current.target_theorem
    ]
    ordered_nodes = definitions + proofs + [root]
    parts = [current.phase2_header.rstrip()]
    if _PENDING_HELPER_RE.search(current.lean_file):
        parts.append(_PENDING_HELPER)
    parts.extend(node.lean_declaration.strip() for node in ordered_nodes)
    lean_code = "\n\n".join(part for part in parts if part.strip()).strip() + "\n"
    revised = _parse_blueprint(lean_code, current.target_theorem)
    if [node.name for node in revised.nodes] != [node.name for node in ordered_nodes]:
        raise ValueError("editedBlueprintParseMismatch")
    revised.dependency_order()
    return revised


def _apply_node_edits(
    current: Blueprint,
    edits: list[_BlueprintNodeEdit],
) -> Blueprint:
    by_name = {node.name: node for node in current.nodes}
    additions: list[BlueprintNode] = []
    for edit in edits:
        if edit.action == "delete":
            by_name.pop(edit.node_name, None)
        elif edit.action == "replace" and edit.revised_node is not None:
            by_name[edit.node_name] = edit.revised_node
        elif edit.action == "add" and edit.revised_node is not None:
            by_name[edit.node_name] = edit.revised_node
            additions.append(edit.revised_node)
    existing_order = [
        by_name[node.name]
        for node in current.nodes
        if node.name in by_name
    ]
    existing_names = {node.name for node in existing_order}
    final_nodes = existing_order + [node for node in additions if node.name not in existing_names]
    return _render_edited_blueprint(current, final_nodes)


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
    semantic_source_mode: str = "step_grounded",
    phase1_max_tool_turns: int = 3,
    phase1_max_tool_calls_per_turn: int = 3,
    phase1_mathlib_search_max_calls: int = 3,
    phase1b_semantic_audit_enabled: bool = False,
    phase1b_formal_decompiler_max_tokens: int = 4096,
    phase1b_strict_comparator_max_tokens: int = 4096,
    phase1b_semantic_format_max_attempts: int = 2,
    phase1b_seed_lean_code: str = "",
    phase1b_repair_strategy: str = "directEdit",
    phase1b_editor_attempts_per_turn: int = 3,
    phase1b_plan_max_tokens: int = 768,
    phase1b_plan_format_attempts: int = 2,
    phase1b_plan_max_chars: int = 600,
    phase1b_subgraph_max_edits: int = 8,
    phase1b_closure_rounds: int = 0,
    phase1b_mathlib_search_policy: str = "leanErrorsOnly",
    phase1b_mathlib_search_max_queries_per_turn: int = 0,
    phase1b_mathlib_search_max_results_per_query: int = 5,
) -> Blueprint:
    """Generate a Phase-1A draft, then edit its nodes and DAG in Phase 1B."""
    if phase1_max_tool_turns <= 0 or phase1_max_tool_calls_per_turn <= 0:
        raise ValueError("Phase-1 tool turn/call limits must be positive")
    if phase1_mathlib_search_max_calls < 0:
        raise ValueError("Phase-1 search limit must be non-negative")
    if phase1b_formal_decompiler_max_tokens <= 0:
        raise ValueError("Phase-1B Formal Decompiler max tokens must be positive")
    if phase1b_strict_comparator_max_tokens <= 0:
        raise ValueError("Phase-1B Strict Comparator max tokens must be positive")
    if phase1b_semantic_format_max_attempts <= 0:
        raise ValueError("Phase-1B semantic format attempts must be positive")
    if phase1b_plan_max_tokens <= 0:
        raise ValueError("Phase-1B Planner max tokens must be positive")
    if phase1b_plan_format_attempts <= 0:
        raise ValueError("Phase-1B Planner format attempts must be positive")
    if phase1b_plan_max_chars <= 0:
        raise ValueError("Phase-1B Plan max characters must be positive")
    if phase1b_subgraph_max_edits <= 0:
        raise ValueError("Phase-1B subgraph edit limit must be positive")
    if phase1b_editor_attempts_per_turn <= 0:
        raise ValueError("Phase-1B Editor attempts per turn must be positive")
    if phase1b_closure_rounds < 0 or phase1b_closure_rounds > phase1_max_tool_turns:
        raise ValueError(
            "Phase-1B closure rounds must be between 0 and phase1_max_tool_turns"
        )
    if phase1b_mathlib_search_max_queries_per_turn < 0:
        raise ValueError("Phase-1B Mathlib search query limit must be non-negative")
    if phase1b_mathlib_search_policy != "leanErrorsOnly":
        raise ValueError("phase1b_mathlib_search_policy must be leanErrorsOnly")
    if phase1b_mathlib_search_max_results_per_query <= 0:
        raise ValueError("Phase-1B Mathlib search result limit must be positive")
    from phase1b import REPAIR_STRATEGIES
    if phase1b_repair_strategy not in REPAIR_STRATEGIES:
        raise ValueError(
            "phase1b_repair_strategy must be one of: "
            + ", ".join(sorted(REPAIR_STRATEGIES))
        )
    if semantic_source_mode not in SEMANTIC_SOURCE_MODES:
        raise ValueError(
            "semantic_source_mode must be one of: step_grounded, whole_cot"
        )
    if (
        semantic_require_step_ids
        or semantic_static_gate
        or semantic_minimal_ir
        or phase1b_semantic_audit_enabled
    ) and not semantic_fidelity_enabled:
        raise ValueError("semantic subfeatures require semantic_fidelity_enabled=true")
    if semantic_source_mode == "whole_cot" and not semantic_fidelity_enabled:
        raise ValueError("whole_cot semantic source mode requires semantic fidelity")
    if semantic_source_mode == "whole_cot" and semantic_require_step_ids:
        raise ValueError("whole_cot semantic source mode cannot require Step IDs")
    manifest_rows = _decode_manifest_rows(cot_manifest_json) if semantic_fidelity_enabled else []
    semantic_manifest = (
        parse_cot_manifest(manifest_rows) if semantic_fidelity_enabled else None
    )
    if semantic_fidelity_enabled and not manifest_rows:
        raise ValueError("semantic_fidelity_enabled requires a non-empty cot_manifest_json")
    prompt_proof = informal_proof or ""
    if semantic_fidelity_enabled and semantic_source_mode == "step_grounded":
        prompt_proof = _render_step_grounded_proof(
            cot_manifest_json,
            include_ir=semantic_minimal_ir,
        )

    semantic_system_suffix = ""
    if semantic_fidelity_enabled:
        semantic_system_suffix = (
            _SEMANTIC_BLUEPRINT_SYSTEM_SUFFIX
            if semantic_source_mode == "step_grounded"
            else _WHOLE_COT_SEMANTIC_BLUEPRINT_SYSTEM_SUFFIX
        )

    client = make_client(model)
    messages = [
        {
            "role": "system",
            "content": ROBUSTPA_BLUEPRINT_SYSTEM_PROMPT
            + semantic_system_suffix
            + _PHASE1A_SKELETON_SUFFIX,
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
                f"problem object and final formalized COT step is: `{claimed_answer}`."
                if semantic_fidelity_enabled else ""
            ),
        },
    ]
    base_messages = tuple(dict(message) for message in messages)

    last_error_feedback = ""
    last_candidate = ""
    last_diagnostics: list[str] = []
    last_finish_reason: str | None = None
    last_failure_stage = "phase1A_model_output"
    candidate_history: list[str] = []
    candidate_labels: list[str] = []
    tool_cache: dict[str, tuple[str, bool]] = {}
    search_state = {"count": 0}
    retrieval = MathlibRetrieval()
    phase1a_blueprint: Blueprint | None = None
    if phase1b_seed_lean_code.strip():
        seed_target = _extract_target_name(phase1b_seed_lean_code, "")
        if seed_target != target_name:
            raise BlueprintGenerationError(
                f"Seed Blueprint target `{seed_target or '<missing>'}` does not match "
                f"expected `{target_name}`.",
                last_candidate=phase1b_seed_lean_code,
                diagnostics=["seed theorem name mismatch"],
                failure_stage="seedBlueprintInvalid",
            )
        phase1a_blueprint = _parse_blueprint(phase1b_seed_lean_code, target_name)
        if not phase1a_blueprint.nodes:
            raise BlueprintGenerationError(
                "Seed Blueprint contains no parsed Blueprint nodes.",
                last_candidate=phase1b_seed_lean_code,
                diagnostics=["seed has no Blueprint nodes"],
                failure_stage="seedBlueprintInvalid",
            )
        candidate_history.append(phase1a_blueprint.lean_file)
        candidate_labels.append("phase1b_seed")

    for attempt in range(max_retries if phase1a_blueprint is None else 0):
        session = _run_phase1_tool_session(
            client, model, tuple(dict(message) for message in messages),
            compiler=compiler, target_name=target_name, retrieval=retrieval,
            tracer=tracer, thm_name=thm_name,
            attempt=attempt + 1,
            max_tool_turns=phase1_max_tool_turns,
            max_tool_calls_per_turn=phase1_max_tool_calls_per_turn,
            mathlib_search_max_calls=phase1_mathlib_search_max_calls,
            tool_cache=tool_cache, search_state=search_state,
            semantic_manifest=semantic_manifest,
            claimed_answer=claimed_answer,
            semantic_fidelity_enabled=semantic_fidelity_enabled,
            semantic_require_step_ids=semantic_require_step_ids,
            semantic_static_gate=semantic_static_gate,
            allow_pending_claims=True,
            trace_phase="phase1A",
        )
        lean_code = session.successful_lean_code or session.lean_code
        last_candidate = lean_code
        candidate_history.append(lean_code)
        candidate_labels.append(f"phase1a_attempt_{attempt + 1}")
        last_finish_reason = session.finish_reason
        emitted_target = _extract_target_name(lean_code, "")
        feedback_parts: list[str] = []
        if emitted_target != target_name:
            feedback_parts.append(
                f"The main theorem must be named `{target_name}`, but the "
                f"latest output's final theorem is `{emitted_target or '<missing>'}`."
            )
        candidate = _parse_blueprint(lean_code, target_name)
        _emit_pending_summary(
            tracer, thm_name=thm_name, phase="phase1A",
            round_index=attempt + 1, blueprint=candidate,
        )
        semantic_check_issues = _enabled_semantic_issues(
            validate_blueprint_fidelity(
                candidate,
                semantic_manifest,
                claimed_answer=claimed_answer,
                require_step_bindings=semantic_require_step_ids,
                allow_pending_claims=True,
            ) if semantic_fidelity_enabled and candidate.nodes else [],
            require_step_ids=semantic_require_step_ids,
            static_gate=semantic_static_gate,
        )
        semantic_check_errors = [
            issue for issue in semantic_check_issues if issue.severity == "error"
        ]
        phase1a_blocking_semantic_issues = _phase1a_blocking_semantic_issues(
            semantic_check_issues
        )
        _emit_semantic_check(
            tracer, thm_name=thm_name, phase="phase1A",
            attempt=attempt + 1, issues=semantic_check_issues,
            blocking_issues=phase1a_blocking_semantic_issues,
        )
        if phase1a_blocking_semantic_issues:
            feedback_parts.append(
                "The deterministic semantic gate rejected the skeleton:\n"
                + format_semantic_issues(phase1a_blocking_semantic_issues)
            )
        contract_errors: list[str] = []
        if not candidate.nodes:
            contract_errors.append("no_blueprint_nodes: no annotated declarations were parsed.")
        else:
            contract_errors.extend(_phase1a_contract_errors(candidate))
        if contract_errors:
            feedback_parts.append(
                "Phase 1A skeleton contract errors:\n"
                + format_phase2_contract_errors(contract_errors, limit=100)
            )

        result = (
            compiler.check_blueprint(lean_code, target_name)
            if emitted_target == target_name
            else CompilerResult(
                False,
                errors=[feedback_parts[0]],
                failure_kind="lean",
            )
        )
        lean_source_contexts = _lean_source_contexts(
            result, candidate, semantic_manifest,
        )
        _emit_lean_check_result(
            tracer,
            thm_name=thm_name,
            phase="phase1A",
            attempt=attempt + 1,
            target=target_name,
            result=result,
            source_contexts=lean_source_contexts,
        )
        if result.failure_kind == "infra":
            raise KiminaInfrastructureError(
                "\n".join(result.diagnostics) or result.raw_output[-2000:]
            )
        if not result.success:
            feedback_parts.append(
                "Lean compilation failed:\n"
                +
                ("\n".join(result.diagnostics) or result.raw_output[-2000:])
                + _format_lean_source_contexts(lean_source_contexts)
            )
        canonical_candidate: Blueprint | None = None
        if not feedback_parts and result.success:
            try:
                canonical_candidate = _render_edited_blueprint(candidate, list(candidate.nodes))
            except ValueError as exc:
                feedback_parts.append(
                    "Phase 1A canonical rebuild failed:\n"
                    f"- {exc}\n"
                    "Every sorry_using dependency must be an existing @[blueprint] node."
                )
            else:
                canonical_result = compiler.check_blueprint(
                    canonical_candidate.lean_file, target_name,
                )
                if canonical_result.failure_kind == "infra":
                    raise KiminaInfrastructureError(
                        "\n".join(canonical_result.diagnostics)
                        or canonical_result.raw_output[-2000:]
                    )
                if tracer is not None:
                    tracer.emit(TraceEvent(
                        kind="phase1ACanonicalCheck",
                        thm_name=thm_name,
                        turn=attempt + 1,
                        args={
                            "phase": "phase1A",
                            "attempt": attempt + 1,
                            "sourceHash": hashlib.sha256(lean_code.encode()).hexdigest(),
                            "canonicalHash": hashlib.sha256(
                                canonical_candidate.lean_file.encode()
                            ).hexdigest(),
                        },
                        ok=canonical_result.success,
                    ))
                if not canonical_result.success:
                    feedback_parts.append(
                        "Phase 1A canonical rebuild does not compile:\n"
                        + ("\n".join(canonical_result.diagnostics)
                           or canonical_result.raw_output[-4000:])
                    )
                else:
                    standalone_report = phase2_standalone_contract_report(
                        canonical_candidate,
                        compiler,
                        concurrency=phase2_contract_check_concurrency,
                        skip_pending=True,
                        tracer=tracer,
                        thm_name=thm_name,
                        round_index=attempt + 1,
                        trace_phase="phase1A",
                    )
                    if standalone_report.issues:
                        feedback_parts.append(
                            "Phase 1A concrete-node standalone checks failed:\n"
                            + format_phase2_standalone_issues(standalone_report.issues)
                        )
                candidate_history.append(canonical_candidate.lean_file)
                candidate_labels.append(f"phase1a_canonical_attempt_{attempt + 1}")
        if not feedback_parts and result.success and canonical_candidate is not None:
            phase1a_blueprint = canonical_candidate
            phase1a_blueprint.semantic_gate_results.append({
                "stage": "phase1A",
                "passed": True,
                "issues": [issue.to_dict() for issue in semantic_check_issues],
                "warning_count": sum(issue.severity == "warning" for issue in semantic_check_issues),
                "deferred_error_count": len(semantic_check_errors),
                "phase1ARootPending": target_name in _pending_node_names(canonical_candidate),
                "initialPendingNodeCount": len(_pending_node_names(canonical_candidate)),
            })
            break
        last_error_feedback = "\n\n".join(feedback_parts)
        last_diagnostics = list(result.diagnostics) or [last_error_feedback]
        last_failure_stage = "phase1AValidation"
        _set_latest_blueprint_retry(
            messages, base_messages, lean_code,
            last_error_feedback
            + "\n\nRegenerate the complete Phase 1A skeleton and fix every listed issue.",
            finish_reason=last_finish_reason,
        )
    if phase1a_blueprint is None:
        raise BlueprintGenerationError(
            f"Phase 1A skeleton generation failed after {max_retries} attempts. "
            f"Last error:\n{last_error_feedback[-2000:]}",
            last_candidate=last_candidate,
            diagnostics=last_diagnostics,
            attempt=max_retries,
            finish_reason=last_finish_reason,
            failure_stage=last_failure_stage,
            candidate_history=candidate_history,
            candidate_labels=candidate_labels,
        )

    from phase1b import run_phase1b_patch_session

    final = run_phase1b_patch_session(
        client, model, phase1a_blueprint,
        compiler=compiler,
        informal_statement=informal_statement,
        prompt_proof=prompt_proof,
        claimed_answer=claimed_answer,
        semantic_manifest=semantic_manifest,
        semantic_fidelity_enabled=semantic_fidelity_enabled,
        semantic_require_step_ids=semantic_require_step_ids,
        semantic_static_gate=semantic_static_gate,
        max_rounds=phase1_max_tool_turns,
        semantic_audit_enabled=phase1b_semantic_audit_enabled,
        formal_decompiler_max_tokens=phase1b_formal_decompiler_max_tokens,
        strict_comparator_max_tokens=phase1b_strict_comparator_max_tokens,
        semantic_format_max_attempts=phase1b_semantic_format_max_attempts,
        repair_strategy=phase1b_repair_strategy,
        editor_attempts_per_turn=phase1b_editor_attempts_per_turn,
        plan_max_tokens=phase1b_plan_max_tokens,
        plan_format_attempts=phase1b_plan_format_attempts,
        plan_max_chars=phase1b_plan_max_chars,
        subgraph_max_edits=phase1b_subgraph_max_edits,
        closure_rounds=phase1b_closure_rounds,
        mathlib_search_policy=phase1b_mathlib_search_policy,
        mathlib_search_max_queries_per_turn=(
            phase1b_mathlib_search_max_queries_per_turn
        ),
        mathlib_search_max_results_per_query=(
            phase1b_mathlib_search_max_results_per_query
        ),
        phase2_contract_check_concurrency=phase2_contract_check_concurrency,
        tracer=tracer,
        thm_name=thm_name,
        candidate_history=candidate_history,
        candidate_labels=candidate_labels,
    )
    contract_errors = phase2_contract_errors(final)
    if not contract_errors:
        contract_errors = phase2_standalone_contract_errors(
            final, compiler, concurrency=phase2_contract_check_concurrency,
        )
    if contract_errors:
        raise BlueprintGenerationError(
            "Phase 1B produced a Blueprint that is not usable by Phase 2:\n"
            + format_phase2_contract_errors(contract_errors),
            last_candidate=final.lean_file,
            diagnostics=contract_errors,
            failure_stage="blueprint_contract",
            candidate_history=candidate_history,
            candidate_labels=candidate_labels,
        )
    if final.phase1b_validation.get("semanticAudit"):
        final.semantic_audit_result = dict(final.phase1b_validation["semanticAudit"])
    final.candidate_history = list(candidate_history)
    final.candidate_labels = list(candidate_labels)
    return final


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
        source_match = re.fullmatch(r"COT_STEP:(S\d{3,})", title.strip())
        source_step_id = source_match.group(1) if source_match else ""

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
            lean_start_line=lean_code.count("\n", 0, m.start()) + 1,
            lean_end_line=lean_code.count("\n", 0, m.end()) + 1,
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

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
import threading
import time
import uuid
from pathlib import Path
from typing import Any

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

The root theorem itself must perform the COT's final inference. Do not create
a non-root lemma/theorem with the same complete proposition as the root merely
because the COT repeats or emphasizes its final answer. A final verbal answer
restatement contributes no extra Blueprint node; depend directly on the last
substantive premises and give their conclusion to the root.
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
The root must state the claimed answer about that modeled object. It must
perform the COT's final inference directly; do not create a non-root node with
the same complete proposition merely because the COT repeats its answer.
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


def _emit_semantic_check(
    tracer,
    *,
    thm_name: str,
    phase: str,
    attempt: int,
    issues: list[SemanticIssue],
    turn: int | None = None,
) -> None:
    if tracer is None:
        return
    blocking_count = sum(issue.severity == "error" for issue in issues)
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
            "blocking_count": blocking_count,
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
    ) -> None:
        super().__init__(message)
        self.last_candidate = last_candidate
        self.diagnostics = list(diagnostics or [])
        self.attempt = attempt
        self.finish_reason = finish_reason
        self.failure_stage = failure_stage
        self.candidate_history = list(candidate_history or [])


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
    if any(code.startswith("unconstrainedExists") for code in codes):
        guidance.append(
            "Model the original constrained object and givens explicitly; the root must "
            "answer the original question, not merely choose a closed witness."
        )
    if "rootMissingClaimedAnswer" in codes:
        guidance.append(
            "Keep the original COT's claimed answer literally in the root proposition "
            "about the original modeled object."
        )
    if "duplicateRootConclusion" in codes:
        guidance.append(
            "Remove the duplicated non-root answer proposition. Let the root perform the "
            "final inference directly from the last substantive premises."
        )
    if codes.intersection({"claimedAnswerInDefinition", "claimedAnswerInPropDefinition"}):
        guidance.append(
            "Do not hard-code the claimed answer in a non-root definition. Model the "
            "source object and state the COT's asserted equality as a proof node."
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
) -> tuple[str, bool]:
    """Combine Lean and deterministic semantic validation for one tool call."""
    issues = list(semantic_issues or [])
    errors = [issue for issue in issues if issue.severity == "error"]
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
        semantic_status = "FAILED" if errors else "PASSED WITH WARNINGS"
        sections.append(
            f"Deterministic semantic-fidelity validation {semantic_status}.\n"
            f"{format_semantic_issues(issues)}"
        )
    if errors:
        sections.append(
            "Issue-specific repair rules:\n"
            f"{_semantic_repair_guidance(errors)}\n\n"
            "Repair every blocking issue and call lean_compile with the complete file again. "
            "Use each Step ID to look up its source text in the COT already present above."
        )
    return "\n\n".join(sections), result.success and not errors


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
            phase="phase1",
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
        _emit_usage(tracer, thm_name, "phase1", model, response)
        _emit_llm_response(
            tracer, thm_name=thm_name, phase="phase1", model=model,
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
                args={"phase": "phase1", "attempt": attempt, "calls": dropped},
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
                    args={"phase": "phase1", "attempt": attempt,
                          "arguments": trace_arguments, "hash": call_hash},
                ))
            semantic_issues: list[SemanticIssue] = []
            parsed_candidate: Blueprint | None = None
            if name == "lean_compile" and semantic_fidelity_enabled:
                lean_code = str(args["lean_code"])
                parsed_candidate = _parse_blueprint(lean_code, target_name)
                if parsed_candidate.nodes:
                    semantic_issues = _enabled_semantic_issues(
                        validate_blueprint_fidelity(
                            parsed_candidate,
                            semantic_manifest,
                            claimed_answer=claimed_answer,
                            require_step_bindings=semantic_require_step_ids,
                        ),
                        require_step_ids=semantic_require_step_ids,
                        static_gate=semantic_static_gate,
                    )
                    _emit_semantic_check(
                        tracer,
                        thm_name=thm_name,
                        phase="phase1Tool",
                        attempt=attempt,
                        turn=turn,
                        issues=semantic_issues,
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
                    args={"phase": "phase1", "attempt": attempt,
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
    semantic_source_mode: str = "step_grounded",
    phase1_max_tool_turns: int = 3,
    phase1_max_tool_calls_per_turn: int = 3,
    phase1_mathlib_search_max_calls: int = 3,
) -> Blueprint:
    """Generate and strictly validate a blueprint from informal text only.

    Unlike generate_blueprint(), this entry point has no formal Lean theorem
    signature to preserve. The model must formalize the main theorem itself,
    but the theorem identifier is fixed by target_name so downstream
    checkpointing, validation, and scoring remain stable.
    """
    if phase1_max_tool_turns <= 0 or phase1_max_tool_calls_per_turn <= 0:
        raise ValueError("Phase-1 tool turn/call limits must be positive")
    if phase1_mathlib_search_max_calls < 0:
        raise ValueError("Phase-1 search limit must be non-negative")
    if semantic_audit_mode not in {"none", "risk", "full"}:
        raise ValueError("semantic_audit_mode must be one of: none, risk, full")
    if semantic_source_mode not in SEMANTIC_SOURCE_MODES:
        raise ValueError(
            "semantic_source_mode must be one of: step_grounded, whole_cot"
        )
    if (
        semantic_require_step_ids
        or semantic_static_gate
        or semantic_minimal_ir
        or semantic_audit_mode != "none"
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
            + semantic_system_suffix,
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
    observed_semantic_issues: list[str] = []
    candidate_history: list[str] = []
    tool_cache: dict[str, tuple[str, bool]] = {}
    search_state = {"count": 0}
    retrieval = MathlibRetrieval()
    for attempt in range(max_retries):
        semantic_check_issues: list[SemanticIssue] = []
        semantic_check_errors: list[SemanticIssue] = []
        candidate: Blueprint | None = None
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
        )
        lean_code = session.successful_lean_code or session.lean_code
        last_candidate = lean_code
        candidate_history.append(lean_code)
        last_finish_reason = session.finish_reason
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
                semantic_check_errors = [
                    issue for issue in semantic_check_issues
                    if issue.severity == "error"
                ]
                _emit_semantic_check(
                    tracer,
                    thm_name=thm_name,
                    phase="phase1",
                    attempt=attempt + 1,
                    issues=semantic_check_issues,
                )
                if semantic_check_errors:
                    for issue in semantic_check_errors:
                        issue_key = ":".join(
                            part for part in (issue.code, issue.step_id, issue.node_name) if part
                        )
                        if issue_key not in observed_semantic_issues:
                            observed_semantic_issues.append(issue_key)

        result = compiler.check_blueprint(lean_code, target_name)
        if candidate is None:
            candidate = _parse_blueprint(lean_code, target_name)
        lean_source_contexts = _lean_source_contexts(
            result, candidate, semantic_manifest,
        )
        _emit_lean_check_result(
            tracer,
            thm_name=thm_name,
            phase="phase1",
            attempt=attempt + 1,
            target=target_name,
            result=result,
            source_contexts=lean_source_contexts,
        )
        if result.failure_kind == "infra":
            raise KiminaInfrastructureError(
                "\n".join(result.diagnostics) or result.raw_output[-2000:]
            )
        if semantic_check_errors:
            # The repair budget is deliberately one turn.  Compile the same
            # rejected candidate as well so that this single turn receives
            # both semantic and Lean diagnostics instead of discovering Lean
            # errors only after it has spent its sole repair on graph shape.
            semantic_feedback = format_semantic_issues(semantic_check_errors)
            lean_feedback = ""
            if not result.success:
                lean_feedback = (
                    "\n\nThe same candidate also failed Lean compilation:\n\n"
                    + ("\n".join(result.diagnostics) or result.raw_output[-2000:])
                    + _format_lean_source_contexts(lean_source_contexts)
                )
            last_error_feedback = (
                "The local semantic-fidelity gate rejected this candidate:\n\n"
                f"{semantic_feedback}{lean_feedback}"
            )
            last_diagnostics = [last_error_feedback]
            last_failure_stage = "semantic_gate"
            feedback = (
                f"{last_error_feedback}\n\nCorrect all listed translation and Lean "
                "errors in one pass and re-emit the entire file. Preserve the source COT "
                "exactly; a false step must remain a proposition and explicit proof gap.\n\n"
                f"Issue-specific repair rules:\n{_semantic_repair_guidance(semantic_check_errors)}\n\n"
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
                        "issues": [issue.to_dict() for issue in semantic_check_issues],
                        "warning_count": sum(
                            issue.severity == "warning" for issue in semantic_check_issues
                        ),
                        "require_step_ids": semantic_require_step_ids,
                        "static_gate": semantic_static_gate,
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
                            parsed.candidate_history = list(candidate_history)
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
                        last_error_feedback = (
                            "The semantic-fidelity audit rejected this candidate:\n\n"
                            f"{audit_feedback}"
                        )
                        last_diagnostics = [last_error_feedback]
                        last_failure_stage = "semantic_audit"
                        feedback = (
                            f"{last_error_feedback}\n\nRepair only the Lean translation. "
                            "Do not repair, weaken, or omit any original COT Step clause. "
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
                    parsed.candidate_history = list(candidate_history)
                    return parsed
                last_error_feedback = (
                    "The file compiled, but the blueprint is not usable by Phase 2:\n\n"
                    f"{format_phase2_contract_errors(contract_errors)}"
                )
                last_failure_stage = "blueprint_contract"
        else:
            last_error_feedback = (
                ("\n".join(result.diagnostics) or result.raw_output[-2000:])
                + _format_lean_source_contexts(lean_source_contexts)
            )
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
        candidate_history=candidate_history,
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

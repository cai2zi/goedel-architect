"""The single COT-to-Blueprint generation pipeline.

Each round emits a complete Blueprint.  Deterministic Lean checks and the
strict semantic audit are fed back into the next full-file regeneration.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import time
import uuid
from typing import Any, Sequence

from blueprint import (
    Blueprint,
    BlueprintGenerationError,
    Phase2StandaloneReport,
    ROBUSTPA_BLUEPRINT_MINIMAL_SYSTEM_PROMPT,
    ROBUSTPA_BLUEPRINT_SYSTEM_PROMPT,
    ROBUSTPA_BLUEPRINT_USER_TEMPLATE,
    LEAN_COMPILE_TOOL,
    _emit_llm_response,
    _emit_usage,
    _extract_target_name,
    _load_phase1_tokenizer,
    _parse_blueprint,
    canonicalize_blueprint,
    _unannotated_local_declaration_errors,
    phase2_contract_errors,
    phase2_standalone_contract_report,
)
from goedel_prompts import render
from kimina_lean_compiler import (
    CompilerResult,
    KiminaInfrastructureError,
    KiminaLeanCompiler,
)
from llm_client import chat_completion_with_retry, make_client
from semantic_audit import (
    FormalDecompilerResult,
    JointWholeCotAuditResult,
    WholeCotComparatorResult,
    JOINT_WHOLE_COT_PROMPT_VERSION,
    WHOLE_COT_PROMPT_VERSION,
    build_formal_view,
    joint_whole_cot_audit_messages,
    run_formal_decompiler,
    run_joint_whole_cot_audit,
    run_whole_cot_comparator,
    semantic_audit_cache_key,
    whole_cot_comparator_defects,
    whole_cot_comparator_messages,
)
from semantic_fidelity import SemanticIssue, validate_blueprint_fidelity
from tracer import TraceEvent


WHOLE_COT_GENERATION_SYSTEM_SUFFIX = r"""

## Complete Whole-COT Blueprint generation contract

Return a complete replacement Blueprint on every round. Do not emit or refer
to `PendingBlueprintClaim`: every definition body and every lemma/theorem
proposition must be concrete Lean. Proof bodies remain exactly
`:= by sorry_using [...]`. Faithfully formalize the complete supplied COT even
when it is mathematically wrong. Preserve shared objects, relation directions,
quantifiers, dependencies, and the final target. Do not use `COT_STEP` titles;
Blueprint `title` metadata is optional and carries no semantic credit.

`title`, `statement`, and `proof` metadata are optional. Call `lean_compile`
exactly once with the entire replacement file. Do not return prose or a partial
declaration.
"""


@dataclass(frozen=True)
class GenerationRound:
    round_index: int
    candidate_hash: str
    input_tokens: int
    max_completion_tokens: int
    deterministic_errors: tuple[dict[str, Any], ...]
    semantic_errors: tuple[dict[str, Any], ...]
    warnings: tuple[dict[str, Any], ...]
    validation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_index,
            "candidateHash": self.candidate_hash,
            "inputTokens": self.input_tokens,
            "maxCompletionTokens": self.max_completion_tokens,
            "deterministicErrors": list(self.deterministic_errors),
            "semanticErrors": list(self.semantic_errors),
            "warnings": list(self.warnings),
            "validation": self.validation,
        }


@dataclass
class BlueprintValidation:
    lean_result: CompilerResult
    semantic_issues: list[SemanticIssue]
    structural_errors: list[str]
    standalone_report: Phase2StandaloneReport
    canonical_lean_result: CompilerResult | None = None
    mechanical_stage_reached: str = "parse_basic"
    mechanical_failure_stage: str | None = None
    semantic_audit_invoked: bool = False
    semantic_audit_mode: str = "separate"
    semantic_request_count: int = 0
    semantic_cache_hits: dict[str, bool] | None = None
    semantic_output_budget: int | None = None
    formal_decompiler_result: FormalDecompilerResult | None = None
    strict_comparator_result: WholeCotComparatorResult | None = None
    joint_audit_result: JointWholeCotAuditResult | None = None
    semantic_audit_protocol: str = WHOLE_COT_PROMPT_VERSION

    @property
    def passed(self) -> bool:
        return (
            self.lean_result.success
            and self.canonical_lean_result is not None
            and self.canonical_lean_result.success
            and not self.structural_errors
            and not self.standalone_report.issues
            and self.strict_comparator_result is not None
            and self.strict_comparator_result.passed
        )


class SemanticAuditExecutionError(RuntimeError):
    """Terminal Decompiler/Comparator failure after its schema retry budget."""

    def __init__(self, message: str, validation: BlueprintValidation):
        super().__init__(message)
        self.validation = validation


def validation_details(validation: BlueprintValidation) -> dict[str, Any]:
    standalone = validation.standalone_report
    audit = None
    if validation.formal_decompiler_result and validation.strict_comparator_result:
        audit = {
            "formalDecompiler": validation.formal_decompiler_result.to_dict(),
            "protocol": validation.semantic_audit_protocol,
            "mode": validation.semantic_audit_mode,
            "actualRequestCount": validation.semantic_request_count,
            "cacheHits": dict(validation.semantic_cache_hits or {}),
            "outputBudget": validation.semantic_output_budget,
            "classification": "strictAccepted" if validation.passed else "semanticRejected",
        }
        audit["wholeCotComparator"] = validation.strict_comparator_result.to_dict()
        if validation.joint_audit_result is not None:
            audit["jointRequest"] = validation.joint_audit_result.to_dict()
    static_errors = [
        issue.to_dict() for issue in validation.semantic_issues
        if issue.severity == "error"
    ]
    static_warnings = [
        issue.to_dict() for issue in validation.semantic_issues
        if issue.severity == "warning"
    ]
    return {
        "passed": validation.passed,
        "mechanicalStageReached": validation.mechanical_stage_reached,
        "mechanicalFailureStage": validation.mechanical_failure_stage,
        "mechanicalPassed": validation.mechanical_failure_stage is None,
        "wholeFileLeanSuccess": validation.lean_result.success,
        "leanErrors": list(validation.lean_result.diagnostics),
        "canonicalLeanSuccess": (
            validation.canonical_lean_result.success
            if validation.canonical_lean_result is not None else None
        ),
        "canonicalLeanErrors": (
            list(validation.canonical_lean_result.diagnostics)
            if validation.canonical_lean_result is not None else []
        ),
        "staticShadowErrors": static_errors,
        "staticShadowWarnings": static_warnings,
        "staticShadowWouldReject": bool(static_errors),
        "phase2StructuralErrors": list(validation.structural_errors),
        "phase2StandaloneErrors": [
            issue.to_dict() for issue in standalone.issues
        ],
        "phase2StandaloneSummary": {
            "checkedNodeCount": standalone.checked_node_count,
            "cachedNodeCount": standalone.cached_node_count,
            "failedNodeCount": len(standalone.issues),
            "notRunReason": standalone.not_run_reason,
            "durationMs": standalone.duration_ms,
        },
        "semanticAuditInvoked": validation.semantic_audit_invoked,
        "semanticAuditMode": validation.semantic_audit_mode,
        "semanticActualRequestCount": validation.semantic_request_count,
        "semanticCacheHits": dict(validation.semantic_cache_hits or {}),
        "semanticOutputBudget": validation.semantic_output_budget,
        "semanticAudit": audit,
    }


def _with_semantic_audit(
    validation: BlueprintValidation,
    blueprint: Blueprint,
    *,
    client: Any,
    model: str,
    informal_statement: str,
    informal_proof: str,
    claimed_answer: str,
    formal_decompiler_max_tokens: int,
    strict_comparator_max_tokens: int,
    format_max_attempts: int,
    semantic_audit_enable_thinking: bool,
    semantic_audit_temperature: float,
    semantic_audit_top_p: float,
    semantic_audit_top_k: int,
    semantic_audit_min_p: float,
    semantic_audit_presence_penalty: float,
    semantic_audit_repetition_penalty: float,
    semantic_audit_mode: str,
    joint_semantic_audit_max_tokens: int,
    tokenizer_path: str,
    model_max_context: int,
    context_safety_margin: int,
    decompiler_cache: dict[str, FormalDecompilerResult],
    comparator_cache: dict[str, Any],
    joint_cache: dict[str, JointWholeCotAuditResult],
    tracer: Any,
    thm_name: str,
    round_index: int,
) -> BlueprintValidation:
    view = build_formal_view(blueprint)
    if semantic_audit_mode == "joint":
        messages = joint_whole_cot_audit_messages(
            informal_statement, informal_proof, claimed_answer, view,
        )
        joint_input_tokens, available = generation_request_budget(
            messages, tokenizer_path=tokenizer_path,
            model_max_context=model_max_context,
            safety_margin=context_safety_margin,
        )
        output_budget = min(joint_semantic_audit_max_tokens, available)
        if output_budget < 512:
            raise RuntimeError(
                "joint semantic audit has insufficient context: "
                f"inputTokens={joint_input_tokens} availableOutputTokens={available} "
                f"minimumOutputTokens=512 modelContext={model_max_context}"
            )
        request_params = {
            "enable_thinking": semantic_audit_enable_thinking,
            "temperature": semantic_audit_temperature,
            "top_p": semantic_audit_top_p,
            "top_k": semantic_audit_top_k,
            "min_p": semantic_audit_min_p,
            "presence_penalty": semantic_audit_presence_penalty,
            "repetition_penalty": semantic_audit_repetition_penalty,
            "max_tokens": output_budget,
        }
        cache_key = semantic_audit_cache_key(
            model, messages, version=JOINT_WHOLE_COT_PROMPT_VERSION,
            request_params=request_params,
        )
        joint = joint_cache.get(cache_key)
        cache_hit = joint is not None
        validation.semantic_audit_invoked = True
        validation.semantic_audit_mode = "joint"
        validation.semantic_cache_hits = {"joint": cache_hit}
        validation.semantic_output_budget = output_budget
        validation.semantic_audit_protocol = JOINT_WHOLE_COT_PROMPT_VERSION
        if joint is None:
            try:
                joint = run_joint_whole_cot_audit(
                    client, model, informal_statement=informal_statement,
                    informal_proof=informal_proof, claimed_answer=claimed_answer,
                    view=view, max_tokens=output_budget,
                    max_attempts=format_max_attempts, tracer=tracer,
                    enable_thinking=semantic_audit_enable_thinking,
                    temperature=semantic_audit_temperature,
                    top_p=semantic_audit_top_p, top_k=semantic_audit_top_k,
                    min_p=semantic_audit_min_p,
                    presence_penalty=semantic_audit_presence_penalty,
                    repetition_penalty=semantic_audit_repetition_penalty,
                    thm_name=thm_name, round_index=round_index,
                )
            except Exception as exc:
                validation.semantic_request_count = max(
                    1, len(getattr(exc, "attempts", ()) or ()),
                )
                raise
            joint_cache[cache_key] = joint
        elif tracer:
            tracer.emit(TraceEvent(
                kind="jointSemanticAuditCacheHit", thm_name=thm_name,
                turn=round_index,
                args={
                    "round": round_index, "cacheKey": cache_key,
                    "formalViewHash": view.sha256,
                    "actualRequestCount": 0,
                },
                ok=True,
            ))
        return BlueprintValidation(
            lean_result=validation.lean_result,
            semantic_issues=validation.semantic_issues,
            structural_errors=validation.structural_errors,
            standalone_report=validation.standalone_report,
            canonical_lean_result=validation.canonical_lean_result,
            mechanical_stage_reached=validation.mechanical_stage_reached,
            mechanical_failure_stage=validation.mechanical_failure_stage,
            semantic_audit_invoked=True,
            semantic_audit_mode="joint",
            semantic_request_count=0 if cache_hit else len(joint.attempts),
            semantic_cache_hits={"joint": cache_hit},
            semantic_output_budget=output_budget,
            formal_decompiler_result=joint.decompiler,
            strict_comparator_result=joint.comparator,
            joint_audit_result=joint,
            semantic_audit_protocol=JOINT_WHOLE_COT_PROMPT_VERSION,
        )

    validation.semantic_audit_invoked = True
    validation.semantic_audit_mode = "separate"
    validation.semantic_audit_protocol = WHOLE_COT_PROMPT_VERSION
    decompiler = decompiler_cache.get(view.sha256)
    decompiler_hit = decompiler is not None
    validation.semantic_cache_hits = {"formalDecompiler": decompiler_hit}
    if decompiler is None:
        try:
            decompiler = run_formal_decompiler(
                client,
                model,
                view=view,
                max_tokens=formal_decompiler_max_tokens,
                max_attempts=format_max_attempts,
                enable_thinking=semantic_audit_enable_thinking,
                temperature=semantic_audit_temperature,
                top_p=semantic_audit_top_p,
                top_k=semantic_audit_top_k,
                min_p=semantic_audit_min_p,
                presence_penalty=semantic_audit_presence_penalty,
                repetition_penalty=semantic_audit_repetition_penalty,
                tracer=tracer,
                thm_name=thm_name,
                round_index=round_index,
            )
        except Exception as exc:
            validation.semantic_request_count = max(
                1, len(getattr(exc, "attempts", ()) or ()),
            )
            raise
        decompiler_cache[view.sha256] = decompiler
        validation.semantic_request_count = len(decompiler.attempts)
    messages = whole_cot_comparator_messages(
        informal_statement, informal_proof, claimed_answer, view, decompiler,
    )
    cache_key = semantic_audit_cache_key(
        model, messages, version=WHOLE_COT_PROMPT_VERSION,
    )
    comparator = comparator_cache.get(cache_key)
    comparator_hit = comparator is not None
    validation.semantic_cache_hits["wholeCotComparator"] = comparator_hit
    if comparator is None:
        try:
            comparator = run_whole_cot_comparator(
                client, model, informal_statement=informal_statement,
                informal_proof=informal_proof, claimed_answer=claimed_answer,
                view=view, decompiler=decompiler,
                max_tokens=strict_comparator_max_tokens,
                max_attempts=format_max_attempts, tracer=tracer,
                enable_thinking=semantic_audit_enable_thinking,
                temperature=semantic_audit_temperature,
                top_p=semantic_audit_top_p,
                top_k=semantic_audit_top_k,
                min_p=semantic_audit_min_p,
                presence_penalty=semantic_audit_presence_penalty,
                repetition_penalty=semantic_audit_repetition_penalty,
                thm_name=thm_name, round_index=round_index,
            )
        except Exception as exc:
            validation.semantic_request_count += max(
                1, len(getattr(exc, "attempts", ()) or ()),
            )
            raise
        comparator_cache[cache_key] = comparator
    return BlueprintValidation(
        lean_result=validation.lean_result,
        semantic_issues=validation.semantic_issues,
        structural_errors=validation.structural_errors,
        standalone_report=validation.standalone_report,
        canonical_lean_result=validation.canonical_lean_result,
        mechanical_stage_reached=validation.mechanical_stage_reached,
        mechanical_failure_stage=validation.mechanical_failure_stage,
        semantic_audit_invoked=True,
        semantic_audit_mode="separate",
        semantic_request_count=(
            (0 if decompiler_hit else len(decompiler.attempts))
            + (0 if comparator_hit else len(comparator.attempts))
        ),
        semantic_cache_hits={
            "formalDecompiler": decompiler_hit,
            "wholeCotComparator": comparator_hit,
        },
        semantic_output_budget=None,
        formal_decompiler_result=decompiler,
        strict_comparator_result=comparator,
        semantic_audit_protocol=WHOLE_COT_PROMPT_VERSION,
    )


def _accepted_validation_details(
    details: dict[str, Any],
    rounds: Sequence[GenerationRound],
    *,
    classification: str,
    deterministic_errors: Sequence[dict[str, Any]],
    semantic_errors: Sequence[dict[str, Any]],
    warnings: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Build terminal details without linking a round back to itself."""
    final_details = dict(details)
    audit = dict(final_details.get("semanticAudit") or {})
    audit["classification"] = classification
    final_details["semanticAudit"] = audit
    final_details.update({
        "classification": classification,
        "generationRounds": [item.to_dict() for item in rounds],
        "finalDeterministicErrors": list(deterministic_errors),
        "finalSemanticErrors": list(semantic_errors),
        "finalWarnings": list(warnings),
        "warningCount": len(warnings),
    })
    return final_details


def generation_round_classification(
    *,
    round_index: int,
    max_turns: int,
    deterministic_error_count: int,
    semantic_error_count: int,
    warning_count: int,
) -> str | None:
    """Return a terminal generation classification, or None when another round is due."""
    if deterministic_error_count:
        return "structuralRejected" if round_index >= max_turns else None
    if semantic_error_count:
        return "semanticRejected" if round_index >= max_turns else None
    del warning_count
    return "strictAccepted"


def _normalized_message(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text


def _issue(
    code: str,
    message: Any,
    *,
    stage: str,
    node_name: str = "",
) -> dict[str, Any]:
    normalized = _normalized_message(message)
    result = {
        "stage": stage,
        "code": code,
        "nodeName": node_name,
        "message": normalized,
    }
    result["diagnosticFingerprint"] = hashlib.sha256(
        "\x00".join((stage, code, node_name, normalized)).encode()
    ).hexdigest()
    return result


def _deduplicate(items: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        fingerprint = "|".join((
            str(item.get("stage") or ""),
            str(item.get("code") or ""),
            str(item.get("nodeName") or ""),
            str(item.get("message") or ""),
        ))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        result.append(dict(item))
    return tuple(result)


def generation_request_budget(
    messages: list[dict[str, Any]],
    *,
    tokenizer_path: str,
    model_max_context: int,
    safety_margin: int,
    tools: list[dict[str, Any]] | None = None,
) -> tuple[int, int]:
    """Return serialized input tokens and the exact remaining completion budget."""
    tokenizer = _load_phase1_tokenizer(tokenizer_path)
    try:
        template_kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
        }
        if tools is not None:
            template_kwargs["tools"] = tools
        encoded = tokenizer.apply_chat_template(messages, **template_kwargs)
        input_ids = encoded.get("input_ids") if hasattr(encoded, "get") else encoded
        if (
            isinstance(input_ids, Sequence)
            and input_ids
            and isinstance(input_ids[0], Sequence)
            and not isinstance(input_ids[0], (str, bytes))
        ):
            input_ids = input_ids[0]
        input_tokens = len(input_ids)
    except (TypeError, ValueError):
        payload = json.dumps(
            {"messages": messages, **({"tools": tools} if tools is not None else {})},
            ensure_ascii=False,
            sort_keys=True,
        )
        input_tokens = len(tokenizer.encode(payload, add_special_tokens=True))
    return input_tokens, model_max_context - input_tokens - safety_margin


def _feedback(
    deterministic: Sequence[dict[str, Any]],
    semantic: Sequence[dict[str, Any]],
    warnings: Sequence[dict[str, Any]],
) -> str:
    del warnings
    sections: list[str] = []
    for title, values in (
        ("DETERMINISTIC_ERRORS", deterministic),
        ("SEMANTIC_ERRORS", semantic),
    ):
        lines = [title]
        if not values:
            lines.append("- none")
        else:
            for item in values:
                lines.append(
                    f"- [{item.get('stage')}] {item.get('code')} "
                    f"node={item.get('nodeName') or '<none>'}: {item.get('message')}"
                )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _messages(
    *,
    target_name: str,
    informal_statement: str,
    informal_proof: str,
    claimed_answer: str,
    previous_blueprint: str,
    previous_feedback: str,
    prompt_profile: str = "whole_cot_minimal",
) -> list[dict[str, str]]:
    user = render(
        ROBUSTPA_BLUEPRINT_USER_TEMPLATE,
        target_name=target_name,
        informal_statement=informal_statement,
        informal_proof=informal_proof,
    )
    user += (
        "\n\n## Claimed answer\n"
        f"`{claimed_answer}`\n"
        "Preserve this answer exactly while binding it to the original problem object."
    )
    if previous_blueprint:
        user += (
            "\n\n## Previous complete Blueprint\n```lean\n"
            + previous_blueprint.rstrip()
            + "\n```\n\n## Complete previous diagnostics\n"
            + previous_feedback
        )
    return [
        {"role": "system", "content": (
            ROBUSTPA_BLUEPRINT_MINIMAL_SYSTEM_PROMPT
            if prompt_profile == "whole_cot_minimal"
            else ROBUSTPA_BLUEPRINT_SYSTEM_PROMPT + WHOLE_COT_GENERATION_SYSTEM_SUFFIX
        )},
        {"role": "user", "content": user},
    ]


def _submitted_code(response: Any) -> tuple[str, list[dict[str, Any]]]:
    choice = response.choices[0]
    calls = list(getattr(choice.message, "tool_calls", None) or ())
    problems: list[dict[str, Any]] = []
    if len(calls) != 1:
        problems.append(_issue(
            "phase1ToolCallCount",
            f"expected exactly one lean_compile call, received {len(calls)}",
            stage="parse_basic",
        ))
        return "", problems
    call = calls[0]
    if str(call.function.name) != "lean_compile":
        problems.append(_issue(
            "phase1WrongTool", f"expected lean_compile, received {call.function.name}",
            stage="parse_basic",
        ))
        return "", problems
    try:
        arguments = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError as exc:
        problems.append(_issue("phase1ToolArguments", str(exc), stage="parse_basic"))
        return "", problems
    code = arguments.get("lean_code") if isinstance(arguments, dict) else None
    if not isinstance(code, str) or not code.strip():
        problems.append(_issue(
            "phase1ToolArguments", "lean_code must be a non-empty string",
            stage="parse_basic",
        ))
        return "", problems
    return code.strip() + "\n", problems


def _contract_errors(blueprint: Blueprint, target_name: str) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if not blueprint.nodes:
        errors.append(_issue(
            "noBlueprintNodes", "no annotated declarations were parsed", stage="parse_basic",
        ))
        return errors
    if _extract_target_name(blueprint.lean_file, "") != target_name:
        errors.append(_issue(
            "missingOrWrongRoot", f"root theorem must be named {target_name}",
            stage="parse_basic",
        ))
    if "PendingBlueprintClaim" in blueprint.lean_file:
        errors.append(_issue(
            "forbiddenPendingClaim", "Phase 1 forbids PendingBlueprintClaim everywhere",
            stage="parse_basic",
        ))
    for raw in _unannotated_local_declaration_errors(blueprint):
        errors.append(_issue(raw.split(":", 1)[0], raw, stage="parse_basic"))
    return errors


def _non_sorry_warnings(
    values: Sequence[str], code: str, *, stage: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for value in values:
        lowered = value.lower()
        if "declaration uses" in lowered and ("sorry" in lowered or "admit" in lowered):
            continue
        result.append(_issue(code, value, stage=stage))
    return result


def _not_run_report(reason: str) -> Phase2StandaloneReport:
    return Phase2StandaloneReport((), 0, 0, 0.0, reason)


def _failed_compile(message: str = "not run") -> CompilerResult:
    return CompilerResult(False, errors=[message], failure_kind="assembly")


def _mechanical_validation(
    *,
    whole_result: CompilerResult,
    stage_reached: str,
    failure_stage: str | None,
    canonical_result: CompilerResult | None = None,
    structural: Sequence[str] = (),
    standalone: Phase2StandaloneReport | None = None,
    static_issues: Sequence[SemanticIssue] = (),
    semantic_audit_mode: str = "separate",
) -> BlueprintValidation:
    return BlueprintValidation(
        lean_result=whole_result,
        semantic_issues=list(static_issues),
        structural_errors=list(structural),
        standalone_report=standalone or _not_run_report(failure_stage or "notRun"),
        canonical_lean_result=canonical_result,
        mechanical_stage_reached=stage_reached,
        mechanical_failure_stage=failure_stage,
        semantic_audit_mode=semantic_audit_mode,
    )


def _validate_round(
    code: str,
    *,
    target_name: str,
    compiler: KiminaLeanCompiler,
    informal_statement: str,
    informal_proof: str,
    claimed_answer: str,
    standalone_concurrency: int,
    client: Any,
    model: str,
    decompiler_max_tokens: int,
    comparator_max_tokens: int,
    semantic_format_attempts: int,
    semantic_audit_enable_thinking: bool,
    semantic_audit_temperature: float,
    semantic_audit_top_p: float,
    semantic_audit_top_k: int,
    semantic_audit_min_p: float,
    semantic_audit_presence_penalty: float,
    semantic_audit_repetition_penalty: float,
    semantic_audit_mode: str,
    joint_semantic_audit_max_tokens: int,
    tokenizer_path: str,
    model_max_context: int,
    context_safety_margin: int,
    tracer,
    thm_name: str,
    round_index: int,
    standalone_cache: dict[str, Any],
    decompiler_cache: dict[str, Any],
    comparator_cache: dict[str, Any],
    joint_cache: dict[str, JointWholeCotAuditResult],
) -> tuple[Blueprint, BlueprintValidation, tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Validate one candidate with strict mechanical short-circuiting."""
    candidate = _parse_blueprint(code, target_name)
    deterministic: list[dict[str, Any]] = _contract_errors(candidate, target_name)
    warnings: list[dict[str, Any]] = []
    if deterministic:
        validation = _mechanical_validation(
            whole_result=_failed_compile("whole_file_lean not run"),
            stage_reached="parse_basic", failure_stage="parse_basic",
            semantic_audit_mode=semantic_audit_mode,
        )
        return candidate, validation, _deduplicate(deterministic), (), ()

    lean_result = compiler.check_blueprint(code, target_name)
    if lean_result.failure_kind == "infra":
        raise KiminaInfrastructureError(
            "\n".join(lean_result.diagnostics) or lean_result.raw_output[-2000:]
        )
    warnings.extend(_non_sorry_warnings(
        lean_result.warnings, "leanWarning", stage="whole_file_lean",
    ))
    if not lean_result.success:
        deterministic.extend(
            _issue("wholeFileLean", value, stage="whole_file_lean")
            for value in lean_result.diagnostics
        )
        validation = _mechanical_validation(
            whole_result=lean_result, stage_reached="whole_file_lean",
            failure_stage="whole_file_lean",
            semantic_audit_mode=semantic_audit_mode,
        )
        return candidate, validation, _deduplicate(deterministic), (), _deduplicate(warnings)

    try:
        formal_blueprint = canonicalize_blueprint(candidate, list(candidate.nodes))
    except ValueError as exc:
        deterministic.append(_issue("canonicalRebuild", str(exc), stage="canonical_rebuild"))
        validation = _mechanical_validation(
            whole_result=lean_result, stage_reached="canonical_rebuild",
            failure_stage="canonical_rebuild",
            semantic_audit_mode=semantic_audit_mode,
        )
        return candidate, validation, _deduplicate(deterministic), (), _deduplicate(warnings)

    structural = phase2_contract_errors(formal_blueprint)
    for raw in structural:
        deterministic.append(_issue(
            raw.split(":", 1)[0], raw, stage="phase2_contract",
        ))
    if structural:
        validation = _mechanical_validation(
            whole_result=lean_result, stage_reached="phase2_contract",
            failure_stage="phase2_contract", structural=structural,
            semantic_audit_mode=semantic_audit_mode,
        )
        return formal_blueprint, validation, _deduplicate(deterministic), (), _deduplicate(warnings)

    canonical_result = compiler.check_blueprint(formal_blueprint.lean_file, target_name)
    if canonical_result.failure_kind == "infra":
        raise KiminaInfrastructureError(
            "\n".join(canonical_result.diagnostics) or canonical_result.raw_output[-2000:]
        )
    warnings.extend(_non_sorry_warnings(
        canonical_result.warnings, "canonicalLeanWarning", stage="canonical_lean",
    ))
    if not canonical_result.success:
        deterministic.extend(
            _issue("canonicalLean", value, stage="canonical_lean")
            for value in canonical_result.diagnostics
        )
        validation = _mechanical_validation(
            whole_result=lean_result, canonical_result=canonical_result,
            stage_reached="canonical_lean", failure_stage="canonical_lean",
            semantic_audit_mode=semantic_audit_mode,
        )
        return formal_blueprint, validation, _deduplicate(deterministic), (), _deduplicate(warnings)

    standalone = phase2_standalone_contract_report(
        formal_blueprint, compiler, concurrency=standalone_concurrency,
        cache=standalone_cache, tracer=tracer, thm_name=thm_name,
        round_index=round_index,
    )
    for item in standalone.issues:
        deterministic.append(_issue(
            item.code or "phase2Standalone", item.diagnostic,
            stage="phase2_standalone", node_name=item.node_name,
        ))
    if standalone.issues:
        validation = _mechanical_validation(
            whole_result=lean_result, canonical_result=canonical_result,
            stage_reached="phase2_standalone", failure_stage="phase2_standalone",
            structural=structural, standalone=standalone,
            semantic_audit_mode=semantic_audit_mode,
        )
        return formal_blueprint, validation, _deduplicate(deterministic), (), _deduplicate(warnings)

    semantic_issues = validate_blueprint_fidelity(
        formal_blueprint, claimed_answer=claimed_answer,
    )
    validation = _mechanical_validation(
        whole_result=lean_result, canonical_result=canonical_result,
        stage_reached="static_shadow", failure_stage=None,
        structural=structural, standalone=standalone, static_issues=semantic_issues,
        semantic_audit_mode=semantic_audit_mode,
    )
    semantic: list[dict[str, Any]] = []
    try:
        validation = _with_semantic_audit(
            validation,
            formal_blueprint,
            client=client,
            model=model,
            informal_statement=informal_statement,
            informal_proof=informal_proof,
            claimed_answer=claimed_answer,
            formal_decompiler_max_tokens=decompiler_max_tokens,
            strict_comparator_max_tokens=comparator_max_tokens,
            format_max_attempts=semantic_format_attempts,
            semantic_audit_enable_thinking=semantic_audit_enable_thinking,
            semantic_audit_temperature=semantic_audit_temperature,
            semantic_audit_top_p=semantic_audit_top_p,
            semantic_audit_top_k=semantic_audit_top_k,
            semantic_audit_min_p=semantic_audit_min_p,
            semantic_audit_presence_penalty=semantic_audit_presence_penalty,
            semantic_audit_repetition_penalty=semantic_audit_repetition_penalty,
            semantic_audit_mode=semantic_audit_mode,
            joint_semantic_audit_max_tokens=joint_semantic_audit_max_tokens,
            tokenizer_path=tokenizer_path,
            model_max_context=model_max_context,
            context_safety_margin=context_safety_margin,
            decompiler_cache=decompiler_cache,
            comparator_cache=comparator_cache,
            joint_cache=joint_cache,
            tracer=tracer,
            thm_name=thm_name,
            round_index=round_index,
        )
    except Exception as exc:
        validation.semantic_audit_invoked = True
        validation.semantic_audit_mode = semantic_audit_mode
        validation.mechanical_stage_reached = (
            "joint_semantic_audit"
            if semantic_audit_mode == "joint" else "formal_decompiler_or_comparator"
        )
        raise SemanticAuditExecutionError(str(exc), validation) from exc

    decompiler = validation.formal_decompiler_result
    comparator = validation.strict_comparator_result
    if decompiler is not None:
        for node_name in decompiler.vacuous_nodes:
            semantic.append(_issue(
                "vacuousFormalNode",
                "Formal Decompiler classified this node as vacuous",
                stage="formal_decompiler", node_name=node_name,
            ))
    if comparator is not None:
        for defect in whole_cot_comparator_defects(comparator):
            names = list(defect.get("node_names") or ())
            semantic.append(_issue(
                str(defect.get("category") or "semanticDefect"),
                f"{defect.get('requirement')} Reason: {defect.get('reason')}",
                stage="whole_cot_comparator", node_name=",".join(names),
            ))
        if not comparator.passed and not semantic:
            semantic.append(_issue(
                "semanticComparatorRejected",
                "The Whole-COT semantic comparator rejected the candidate.",
                stage="whole_cot_comparator",
            ))
        for item in comparator.unreachable_nodes:
            if item.get("justified_side_branch"):
                warnings.append(_issue(
                    "justifiedSideBranch", item.get("reason"),
                    stage="whole_cot_comparator",
                    node_name=str(item.get("node_name") or ""),
                ))

    return (
        formal_blueprint,
        validation,
        _deduplicate(deterministic),
        _deduplicate(semantic),
        _deduplicate(warnings),
    )


def generate_blueprint(
    *,
    informal_statement: str,
    informal_proof: str,
    claimed_answer: str,
    target_name: str,
    model: str,
    compiler: KiminaLeanCompiler,
    tracer,
    thm_name: str,
    max_turns: int,
    tokenizer_path: str,
    model_max_context: int,
    context_safety_margin: int,
    enable_thinking: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    presence_penalty: float,
    repetition_penalty: float,
    standalone_concurrency: int,
    decompiler_max_tokens: int,
    comparator_max_tokens: int,
    semantic_format_attempts: int,
    semantic_audit_enable_thinking: bool = False,
    semantic_audit_temperature: float = 0.0,
    semantic_audit_top_p: float = 1.0,
    semantic_audit_top_k: int = -1,
    semantic_audit_min_p: float = 0.0,
    semantic_audit_presence_penalty: float = 0.0,
    semantic_audit_repetition_penalty: float = 1.0,
    semantic_audit_mode: str = "separate",
    joint_semantic_audit_max_tokens: int = 32768,
    prompt_profile: str = "whole_cot_minimal",
) -> Blueprint:
    if max_turns <= 0:
        raise ValueError("generation max_turns must be positive")
    if prompt_profile not in {"whole_cot_minimal", "standard"}:
        raise ValueError("prompt_profile must be whole_cot_minimal or standard")
    if semantic_audit_mode not in {"separate", "joint"}:
        raise ValueError("semantic_audit_mode must be separate or joint")
    if joint_semantic_audit_max_tokens <= 0:
        raise ValueError("joint_semantic_audit_max_tokens must be positive")
    client = make_client(model)
    previous_code = ""
    previous_feedback = ""
    candidates: list[str] = []
    labels: list[str] = []
    rounds: list[GenerationRound] = []
    standalone_cache: dict[str, Any] = {}
    decompiler_cache: dict[str, Any] = {}
    comparator_cache: dict[str, Any] = {}
    joint_cache: dict[str, JointWholeCotAuditResult] = {}
    latest_blueprint: Blueprint | None = None
    latest_validation: BlueprintValidation | None = None
    latest_deterministic: tuple[dict[str, Any], ...] = ()
    latest_semantic: tuple[dict[str, Any], ...] = ()
    latest_warnings: tuple[dict[str, Any], ...] = ()

    for round_index in range(1, max_turns + 1):
        messages = _messages(
            target_name=target_name,
            informal_statement=informal_statement,
            informal_proof=informal_proof,
            claimed_answer=claimed_answer,
            previous_blueprint=previous_code,
            previous_feedback=previous_feedback,
            prompt_profile=prompt_profile,
        )
        input_tokens, completion_budget = generation_request_budget(
            messages,
            tokenizer_path=tokenizer_path,
            model_max_context=model_max_context,
            safety_margin=context_safety_margin,
            tools=[LEAN_COMPILE_TOOL],
        )
        if completion_budget <= 0:
            raise BlueprintGenerationError(
                "Phase 1 input exhausts the model context window.",
                last_candidate=previous_code,
                diagnostics=[
                    f"inputTokens={input_tokens} modelContext={model_max_context} "
                    f"safetyMargin={context_safety_margin}"
                ],
                attempt=round_index,
                failure_stage="phase1ContextBudgetExceeded",
                candidate_history=candidates,
                candidate_labels=labels,
                validation_details={"generationRounds": [item.to_dict() for item in rounds]},
            )
        span_id = uuid.uuid4().hex
        started_ns = time.monotonic_ns()
        if tracer:
            tracer.emit(TraceEvent(
                kind="phase1GenerationStart",
                thm_name=thm_name,
                turn=round_index,
                span_id=span_id,
                args={
                    "round": round_index,
                    "promptProfile": prompt_profile,
                    "inputTokens": input_tokens,
                    "maxCompletionTokens": completion_budget,
                    "enableThinking": enable_thinking,
                    "temperature": temperature,
                    "topP": top_p,
                    "topK": top_k,
                    "minP": min_p,
                    "presencePenalty": presence_penalty,
                    "repetitionPenalty": repetition_penalty,
                    "semanticAuditMode": semantic_audit_mode,
                    "semanticAuditSampling": {
                        "enableThinking": semantic_audit_enable_thinking,
                        "temperature": semantic_audit_temperature,
                        "topP": semantic_audit_top_p,
                        "topK": semantic_audit_top_k,
                        "minP": semantic_audit_min_p,
                        "presencePenalty": semantic_audit_presence_penalty,
                        "repetitionPenalty": semantic_audit_repetition_penalty,
                    },
                },
            ))
        seed = int.from_bytes(hashlib.sha256(
            f"blueprint-generation|{thm_name}|{round_index}".encode()
        ).digest()[:4], "big")
        response = chat_completion_with_retry(
            client,
            tracer=tracer,
            thm_name=thm_name,
            phase="phase1",
            model_id=model,
            operation="blueprint_generation",
            trace_args={"round": round_index, "max_completion_tokens": completion_budget},
            model=model,
            messages=messages,
            tools=[LEAN_COMPILE_TOOL],
            tool_choice="required",
            parallel_tool_calls=False,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            seed=seed,
            max_completion_tokens=completion_budget,
            extra_body={
                "top_k": top_k,
                "min_p": min_p,
                "repetition_penalty": repetition_penalty,
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            },
        )
        _emit_usage(tracer, thm_name, "phase1", model, response)
        _emit_llm_response(
            tracer,
            thm_name=thm_name,
            phase="phase1",
            model=model,
            response=response,
            attempt=1,
            turn=round_index,
        )
        code, submission_errors = _submitted_code(response)
        if submission_errors:
            latest_deterministic = _deduplicate(submission_errors)
            latest_semantic = ()
            latest_warnings = ()
            previous_feedback = _feedback(
                latest_deterministic, latest_semantic, latest_warnings,
            )
            candidate_hash = ""
            details: dict[str, Any] = {}
        else:
            previous_code = code
            candidates.append(code)
            labels.append(f"generation_round_{round_index}")
            try:
                (
                    latest_blueprint,
                    latest_validation,
                    latest_deterministic,
                    latest_semantic,
                    latest_warnings,
                ) = _validate_round(
                    code, target_name=target_name, compiler=compiler,
                    informal_statement=informal_statement,
                    informal_proof=informal_proof, claimed_answer=claimed_answer,
                    standalone_concurrency=standalone_concurrency, client=client,
                    model=model, decompiler_max_tokens=decompiler_max_tokens,
                    comparator_max_tokens=comparator_max_tokens,
                    semantic_format_attempts=semantic_format_attempts,
                    semantic_audit_enable_thinking=semantic_audit_enable_thinking,
                    semantic_audit_temperature=semantic_audit_temperature,
                    semantic_audit_top_p=semantic_audit_top_p,
                    semantic_audit_top_k=semantic_audit_top_k,
                    semantic_audit_min_p=semantic_audit_min_p,
                    semantic_audit_presence_penalty=semantic_audit_presence_penalty,
                    semantic_audit_repetition_penalty=semantic_audit_repetition_penalty,
                    semantic_audit_mode=semantic_audit_mode,
                    joint_semantic_audit_max_tokens=joint_semantic_audit_max_tokens,
                    tokenizer_path=tokenizer_path,
                    model_max_context=model_max_context,
                    context_safety_margin=context_safety_margin,
                    tracer=tracer, thm_name=thm_name, round_index=round_index,
                    standalone_cache=standalone_cache,
                    decompiler_cache=decompiler_cache,
                    comparator_cache=comparator_cache,
                    joint_cache=joint_cache,
                )
            except SemanticAuditExecutionError as exc:
                latest_blueprint = _parse_blueprint(code, target_name)
                latest_validation = exc.validation
                details = validation_details(latest_validation)
                audit_error = _issue(
                    "semanticAuditError", exc,
                    stage=(
                        "joint_semantic_audit"
                        if semantic_audit_mode == "joint"
                        else "formal_decompiler_or_comparator"
                    ),
                )
                failed_round = GenerationRound(
                    round_index=round_index,
                    candidate_hash=hashlib.sha256(code.encode()).hexdigest(),
                    input_tokens=input_tokens,
                    max_completion_tokens=completion_budget,
                    deterministic_errors=(),
                    semantic_errors=(),
                    warnings=(),
                    validation={**details, "semanticAuditError": audit_error},
                )
                rounds.append(failed_round)
                details.update({
                    "classification": "semanticAuditError",
                    "terminalCategory": "semanticAuditError",
                    "semanticAuditError": audit_error,
                    "generationRounds": [item.to_dict() for item in rounds],
                })
                if tracer:
                    tracer.emit(TraceEvent(
                        kind="phase1FinalClassification",
                        thm_name=thm_name,
                        turn=round_index,
                        args={
                            "classification": "semanticAuditError",
                            "error": audit_error,
                        },
                        ok=False,
                    ))
                raise BlueprintGenerationError(
                    "Phase 1 semantic audit exhausted its request/schema retries.",
                    last_candidate=code, diagnostics=[str(exc)], attempt=round_index,
                    failure_stage="phase1SemanticAuditError",
                    candidate_history=candidates, candidate_labels=labels,
                    validation_details=details,
                    generation_history=[item.to_dict() for item in rounds],
                ) from exc
            previous_feedback = _feedback(
                latest_deterministic, latest_semantic, latest_warnings,
            )
            candidate_hash = hashlib.sha256(code.encode()).hexdigest()
            details = validation_details(latest_validation)

        round_row = GenerationRound(
            round_index,
            candidate_hash,
            input_tokens,
            completion_budget,
            latest_deterministic,
            latest_semantic,
            latest_warnings,
            details,
        )
        rounds.append(round_row)
        semantic_invoked = bool(
            latest_validation and latest_validation.semantic_audit_invoked
        )
        for kind, inventory, ok in (
            ("phase1DeterministicValidation", latest_deterministic, not latest_deterministic),
            (
                "phase1SemanticAudit" if semantic_invoked else "phase1SemanticAuditSkipped",
                latest_semantic,
                not latest_semantic if semantic_invoked else True,
            ),
            ("phase1WarningInventory", latest_warnings, not latest_warnings),
        ):
            if tracer:
                tracer.emit(TraceEvent(
                    kind=kind,
                    thm_name=thm_name,
                    turn=round_index,
                    args={
                        "round": round_index, "invoked": semantic_invoked,
                        "count": len(inventory), "issues": list(inventory),
                        "semanticAuditMode": semantic_audit_mode,
                        "actualRequestCount": (
                            latest_validation.semantic_request_count
                            if latest_validation else 0
                        ),
                        "cacheHits": dict(
                            latest_validation.semantic_cache_hits or {}
                        ) if latest_validation else {},
                        "outputBudget": (
                            latest_validation.semantic_output_budget
                            if latest_validation else None
                        ),
                    },
                    ok=ok,
                ))
        if tracer:
            tracer.emit(TraceEvent(
                kind="phase1RoundAssessment",
                thm_name=thm_name,
                turn=round_index,
                args={
                    "round": round_index,
                    "deterministicErrorCount": len(latest_deterministic),
                    "semanticErrorCount": len(latest_semantic),
                    "warningCount": len(latest_warnings),
                    "mechanicalStageReached": (
                        latest_validation.mechanical_stage_reached
                        if latest_validation else "submission"
                    ),
                    "mechanicalFailureStage": (
                        latest_validation.mechanical_failure_stage
                        if latest_validation else "submission"
                    ),
                    "semanticEligible": bool(
                        latest_validation
                        and latest_validation.mechanical_failure_stage is None
                    ),
                    "semanticAuditMode": semantic_audit_mode,
                    "semanticAuditInvoked": semantic_invoked,
                    "semanticRequestCount": (
                        latest_validation.semantic_request_count
                        if latest_validation else 0
                    ),
                },
                ok=not latest_deterministic and not latest_semantic and not latest_warnings,
            ))
            tracer.emit(TraceEvent(
                kind="phase1GenerationEnd",
                thm_name=thm_name,
                turn=round_index,
                span_id=span_id,
                args={"round": round_index, "candidateHash": candidate_hash},
                ok=bool(candidate_hash),
                duration_ms=(time.monotonic_ns() - started_ns) / 1_000_000,
            ))

        classification = generation_round_classification(
            round_index=round_index,
            max_turns=max_turns,
            deterministic_error_count=len(latest_deterministic),
            semantic_error_count=len(latest_semantic),
            warning_count=len(latest_warnings),
        )
        if latest_blueprint is not None and classification in {
            "strictAccepted", "acceptedWithWarnings",
        }:
            # ``details`` is also retained by the current GenerationRound.  Do not
            # add the round history back to that same object: the current
            # round's ``validation`` would then point at a dictionary which
            # contains the current round again, making the terminal result
            # impossible to JSON-serialize ("Circular reference detected").
            final_details = _accepted_validation_details(
                details,
                rounds,
                classification=classification,
                deterministic_errors=latest_deterministic,
                semantic_errors=latest_semantic,
                warnings=latest_warnings,
            )
            final_details["staticGateMode"] = "shadow"
            final_details["semanticStaticGate"] = False
            final_details["semanticAuditMode"] = semantic_audit_mode
            latest_blueprint.generation_validation = final_details
            latest_blueprint.generation_history = [item.to_dict() for item in rounds]
            latest_blueprint.candidate_history = list(candidates)
            latest_blueprint.candidate_labels = list(labels)
            latest_blueprint.semantic_gate_results.append({
                "stage": "generation",
                "passed": True,
                "classification": classification,
                "issues": [
                    {**item, "severity": "warning"} for item in latest_warnings
                ],
            })
            if tracer:
                tracer.emit(TraceEvent(
                    kind="phase1FinalClassification",
                    thm_name=thm_name,
                    turn=round_index,
                    args={"classification": classification, "warningCount": len(latest_warnings)},
                    ok=True,
                ))
            return latest_blueprint

    deterministic_failed = bool(latest_deterministic)
    semantic_failed = bool(latest_semantic)
    failure_stage = (
        "phase1DeterministicAndSemantic"
        if deterministic_failed and semantic_failed else
        "phase1Deterministic"
        if deterministic_failed else
        "phase1Semantic"
    )
    final_details = validation_details(latest_validation) if latest_validation else {}
    final_details.update({
        "classification": "structuralRejected" if deterministic_failed else "semanticRejected",
        "terminalCategory": "mechanicalRejected" if deterministic_failed else "semanticRejected",
        "terminalMechanicalStage": (
            latest_validation.mechanical_failure_stage
            if deterministic_failed and latest_validation else None
        ),
        "staticGateMode": "shadow",
        "semanticStaticGate": False,
        "semanticAuditMode": semantic_audit_mode,
        "finalErrorKinds": [
            name for name, present in (
                ("deterministic", deterministic_failed), ("semantic", semantic_failed),
            ) if present
        ],
        "generationRounds": [item.to_dict() for item in rounds],
        "finalDeterministicErrors": list(latest_deterministic),
        "finalSemanticErrors": list(latest_semantic),
        "finalWarnings": list(latest_warnings),
    })
    if tracer:
        tracer.emit(TraceEvent(
            kind="phase1FinalClassification",
            thm_name=thm_name,
            turn=max_turns,
            args={
                "classification": final_details["classification"],
                "finalErrorKinds": final_details["finalErrorKinds"],
                "warningCount": len(latest_warnings),
            },
            ok=False,
        ))
    raise BlueprintGenerationError(
        "Phase 1 exhausted its full-regeneration turns.",
        last_candidate=previous_code,
        diagnostics=[previous_feedback],
        attempt=max_turns,
        failure_stage=failure_stage,
        candidate_history=candidates,
        candidate_labels=labels,
        validation_details=final_details,
        generation_history=[item.to_dict() for item in rounds],
    )


__all__ = [
    "generation_request_budget",
    "generation_round_classification",
    "generate_blueprint",
]

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
    ROBUSTPA_BLUEPRINT_SYSTEM_PROMPT,
    ROBUSTPA_BLUEPRINT_USER_TEMPLATE,
    LEAN_COMPILE_TOOL,
    _emit_llm_response,
    _emit_usage,
    _extract_target_name,
    _load_phase1_tokenizer,
    _parse_blueprint,
    canonicalize_blueprint,
    _render_step_grounded_proof,
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
    StrictComparatorResult,
    WholeCotComparatorResult,
    WHOLE_COT_PROMPT_VERSION,
    build_formal_view,
    comparator_defects,
    run_formal_decompiler,
    run_strict_comparator,
    run_whole_cot_comparator,
    semantic_audit_cache_key,
    strict_comparator_messages,
    whole_cot_comparator_defects,
    whole_cot_comparator_messages,
)
from semantic_fidelity import (
    SemanticIssue,
    parse_cot_manifest,
    validate_blueprint_fidelity,
)
from tracer import TraceEvent


GENERATION_SYSTEM_SUFFIX = r"""

## Complete Blueprint generation contract

Return a complete replacement Blueprint on every round.  Do not emit or refer
to `PendingBlueprintClaim`: every definition body and every lemma/theorem
proposition must be concrete Lean.  Proof bodies remain exactly
`:= by sorry_using [...]`.  The supplied diagnostics describe the previous
complete candidate; repair all deterministic errors, semantic translation
errors, and warnings while preserving the COT even when the COT is wrong.

Every `@[blueprint]` declaration must include
`(title := "COT_STEP:Snnn")` for exactly one supplied Step and a non-empty
`statement` doc comment.  Every lemma/theorem must additionally include a
non-empty `proof` doc comment.  Cover every supplied Step with at least one
node, and map the root theorem to the final Step.  For example:

    @[blueprint (title := "COT_STEP:S003")
      (statement := /-- Exact formal content of this source Step. -/)
      (proof := /-- Derivation from the named Blueprint parents. -/)]
    lemma derived_relation : P := by sorry_using [problem_model]

Call `lean_compile` exactly once with the entire replacement file.  Do not
return prose or a partial declaration.
"""

WHOLE_COT_GENERATION_SYSTEM_SUFFIX = r"""

## Complete Whole-COT Blueprint generation contract

Return a complete replacement Blueprint on every round. Do not emit or refer
to `PendingBlueprintClaim`: every definition body and every lemma/theorem
proposition must be concrete Lean. Proof bodies remain exactly
`:= by sorry_using [...]`. Faithfully formalize the complete supplied COT even
when it is mathematically wrong. Preserve shared objects, relation directions,
quantifiers, dependencies, and the final target. Do not use `COT_STEP` titles;
Blueprint `title` metadata is optional and carries no semantic credit.

Every declaration must have a non-empty `statement` doc comment, and every
lemma/theorem must also have a non-empty `proof` doc comment. Call
`lean_compile` exactly once with the entire replacement file. Do not return
prose or a partial declaration.
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
    formal_decompiler_result: FormalDecompilerResult | None = None
    strict_comparator_result: StrictComparatorResult | WholeCotComparatorResult | None = None
    semantic_audit_protocol: str = "blueprint-semantic-audit-v1"

    @property
    def passed(self) -> bool:
        return (
            self.lean_result.success
            and not any(issue.severity == "error" for issue in self.semantic_issues)
            and not self.structural_errors
            and not self.standalone_report.issues
            and self.strict_comparator_result is not None
            and self.strict_comparator_result.passed
        )


def validation_details(validation: BlueprintValidation) -> dict[str, Any]:
    standalone = validation.standalone_report
    audit = None
    if validation.formal_decompiler_result and validation.strict_comparator_result:
        audit = {
            "formalDecompiler": validation.formal_decompiler_result.to_dict(),
            "protocol": validation.semantic_audit_protocol,
            "classification": "strictAccepted" if validation.passed else "semanticRejected",
        }
        comparator_key = (
            "wholeCotComparator"
            if validation.semantic_audit_protocol == WHOLE_COT_PROMPT_VERSION
            else "strictComparator"
        )
        audit[comparator_key] = validation.strict_comparator_result.to_dict()
    return {
        "passed": validation.passed,
        "wholeFileLeanSuccess": validation.lean_result.success,
        "leanErrors": list(validation.lean_result.diagnostics),
        "semanticErrors": [
            issue.to_dict() for issue in validation.semantic_issues
            if issue.severity == "error"
        ],
        "semanticWarnings": [
            issue.to_dict() for issue in validation.semantic_issues
            if issue.severity == "warning"
        ],
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
        "semanticAudit": audit,
    }


def _semantic_audit_eligible(validation: BlueprintValidation) -> bool:
    blocking_codes = {
        "emptyCotManifest", "missingRoot", "missingStepMapping",
        "multipleStepMappings", "malformedStepMapping", "unknownStepMapping",
        "rootNotFinalStep", "stepMappingAbsent",
    }
    return (
        not any(
            issue.severity == "error" and issue.code in blocking_codes
            for issue in validation.semantic_issues
        )
    )


def _with_semantic_audit(
    validation: BlueprintValidation,
    blueprint: Blueprint,
    *,
    client: Any,
    model: str,
    informal_statement: str,
    informal_proof: str,
    claimed_answer: str,
    semantic_manifest: Any,
    source_grounding_mode: str,
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
    decompiler_cache: dict[str, FormalDecompilerResult],
    comparator_cache: dict[str, Any],
    tracer: Any,
    thm_name: str,
    round_index: int,
) -> BlueprintValidation:
    whole_cot = source_grounding_mode == "whole_cot"
    view = build_formal_view(blueprint, include_step_ids=not whole_cot)
    decompiler = decompiler_cache.get(view.sha256)
    if decompiler is None:
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
        decompiler_cache[view.sha256] = decompiler
    if whole_cot:
        messages = whole_cot_comparator_messages(
            informal_statement, informal_proof, claimed_answer, view, decompiler,
        )
        cache_key = semantic_audit_cache_key(
            model, messages, version=WHOLE_COT_PROMPT_VERSION,
        )
    else:
        messages = strict_comparator_messages(
            informal_statement, claimed_answer, semantic_manifest, view, decompiler, (),
        )
        cache_key = semantic_audit_cache_key(model, messages)
    comparator = comparator_cache.get(cache_key)
    if comparator is None:
        if whole_cot:
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
        else:
            comparator = run_strict_comparator(
                client, model, informal_statement=informal_statement,
                claimed_answer=claimed_answer, manifest=semantic_manifest,
                view=view, decompiler=decompiler, open_obligations=(),
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
        comparator_cache[cache_key] = comparator
    return BlueprintValidation(
        lean_result=validation.lean_result,
        semantic_issues=validation.semantic_issues,
        structural_errors=validation.structural_errors,
        standalone_report=validation.standalone_report,
        formal_decompiler_result=decompiler,
        strict_comparator_result=comparator,
        semantic_audit_protocol=(
            WHOLE_COT_PROMPT_VERSION if whole_cot else "blueprint-semantic-audit-v1"
        ),
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
    if warning_count:
        return "acceptedWithWarnings" if round_index >= max_turns else None
    return "strictAccepted"


def _normalized_message(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text


def _issue(
    code: str,
    message: Any,
    *,
    node_name: str = "",
    step_id: str = "",
) -> dict[str, Any]:
    return {
        "code": code,
        "nodeName": node_name,
        "stepId": step_id,
        "message": _normalized_message(message),
    }


def _deduplicate(items: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        fingerprint = "|".join((
            str(item.get("code") or ""),
            str(item.get("nodeName") or ""),
            str(item.get("stepId") or ""),
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
        encoded = tokenizer.apply_chat_template(
            messages,
            tools=tools or [],
            tokenize=True,
            add_generation_prompt=True,
        )
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
            {"messages": messages, "tools": tools or []},
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
    sections: list[str] = []
    for title, values in (
        ("DETERMINISTIC_ERRORS", deterministic),
        ("SEMANTIC_ERRORS", semantic),
        ("WARNINGS", warnings),
    ):
        lines = [title]
        if not values:
            lines.append("- none")
        else:
            for item in values:
                lines.append(
                    f"- {item.get('code')} node={item.get('nodeName') or '<none>'} "
                    f"step={item.get('stepId') or '<none>'}: {item.get('message')}"
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
    source_grounding_mode: str = "formal_steps",
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
        {"role": "system", "content": ROBUSTPA_BLUEPRINT_SYSTEM_PROMPT + (
            WHOLE_COT_GENERATION_SYSTEM_SUFFIX
            if source_grounding_mode == "whole_cot"
            else GENERATION_SYSTEM_SUFFIX
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
        ))
        return "", problems
    call = calls[0]
    if str(call.function.name) != "lean_compile":
        problems.append(_issue(
            "phase1WrongTool", f"expected lean_compile, received {call.function.name}",
        ))
        return "", problems
    try:
        arguments = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError as exc:
        problems.append(_issue("phase1ToolArguments", str(exc)))
        return "", problems
    code = arguments.get("lean_code") if isinstance(arguments, dict) else None
    if not isinstance(code, str) or not code.strip():
        problems.append(_issue("phase1ToolArguments", "lean_code must be a non-empty string"))
        return "", problems
    return code.strip() + "\n", problems


def _contract_errors(
    blueprint: Blueprint, target_name: str, *, source_grounding_mode: str = "formal_steps",
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if not blueprint.nodes:
        errors.append(_issue("noBlueprintNodes", "no annotated declarations were parsed"))
        return errors
    if _extract_target_name(blueprint.lean_file, "") != target_name:
        errors.append(_issue("missingOrWrongRoot", f"root theorem must be named {target_name}"))
    if "PendingBlueprintClaim" in blueprint.lean_file:
        errors.append(_issue(
            "forbiddenPendingClaim", "Phase 1 forbids PendingBlueprintClaim everywhere",
        ))
    for raw in _unannotated_local_declaration_errors(blueprint):
        errors.append(_issue(raw.split(":", 1)[0], raw))
    for raw in phase2_contract_errors(blueprint):
        errors.append(_issue(raw.split(":", 1)[0], raw))
    for node in blueprint.nodes:
        if not node.statement.strip():
            errors.append(_issue(
                "missingStatementMetadata", "statement metadata is empty",
                node_name=node.name,
                step_id=node.source_step_id if source_grounding_mode == "formal_steps" else "",
            ))
        if node.kind in {"lemma", "theorem"} and not node.proof_sketch.strip():
            errors.append(_issue(
                "missingProofMetadata", "proof metadata is empty",
                node_name=node.name,
                step_id=node.source_step_id if source_grounding_mode == "formal_steps" else "",
            ))
    return errors


def _non_sorry_warnings(values: Sequence[str], code: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for value in values:
        lowered = value.lower()
        if "declaration uses" in lowered and ("sorry" in lowered or "admit" in lowered):
            continue
        result.append(_issue(code, value))
    return result


def _validate_round(
    code: str,
    *,
    target_name: str,
    compiler: KiminaLeanCompiler,
    semantic_manifest,
    informal_statement: str,
    informal_proof: str,
    claimed_answer: str,
    source_grounding_mode: str,
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
    tracer,
    thm_name: str,
    round_index: int,
    standalone_cache: dict[str, Any],
    decompiler_cache: dict[str, Any],
    comparator_cache: dict[str, Any],
) -> tuple[Blueprint, BlueprintValidation, tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    candidate = _parse_blueprint(code, target_name)
    deterministic: list[dict[str, Any]] = _contract_errors(
        candidate, target_name, source_grounding_mode=source_grounding_mode,
    )
    lean_result = compiler.check_blueprint(code, target_name)
    if lean_result.failure_kind == "infra":
        raise KiminaInfrastructureError(
            "\n".join(lean_result.diagnostics) or lean_result.raw_output[-2000:]
        )
    deterministic.extend(_issue("wholeFileLean", value) for value in lean_result.diagnostics)
    warnings: list[dict[str, Any]] = _non_sorry_warnings(lean_result.warnings, "leanWarning")

    semantic_issues = validate_blueprint_fidelity(
        candidate,
        semantic_manifest,
        claimed_answer=claimed_answer,
        require_step_bindings=source_grounding_mode == "formal_steps",
    ) if candidate.nodes else []
    for issue in semantic_issues:
        row = _issue(
            issue.code, issue.message, node_name=issue.node_name,
            step_id=issue.step_id if source_grounding_mode == "formal_steps" else "",
        )
        if issue.severity == "error":
            deterministic.append(row)
        elif issue.severity == "warning":
            warnings.append(row)

    formal_blueprint = candidate
    canonical_result = None
    if candidate.nodes:
        try:
            formal_blueprint = canonicalize_blueprint(candidate, list(candidate.nodes))
        except ValueError as exc:
            deterministic.append(_issue("canonicalRebuild", str(exc)))
        else:
            canonical_result = compiler.check_blueprint(formal_blueprint.lean_file, target_name)
            if canonical_result.failure_kind == "infra":
                raise KiminaInfrastructureError(
                    "\n".join(canonical_result.diagnostics)
                    or canonical_result.raw_output[-2000:]
                )
            deterministic.extend(
                _issue("canonicalLean", value) for value in canonical_result.diagnostics
            )
            warnings.extend(_non_sorry_warnings(canonical_result.warnings, "canonicalLeanWarning"))

    structural = phase2_contract_errors(formal_blueprint) if formal_blueprint.nodes else []
    if canonical_result is not None and canonical_result.success and not structural:
        standalone = phase2_standalone_contract_report(
            formal_blueprint,
            compiler,
            concurrency=standalone_concurrency,
            cache=standalone_cache,
            tracer=tracer,
            thm_name=thm_name,
            round_index=round_index,
        )
    else:
        standalone = Phase2StandaloneReport((), 0, 0, 0.0, "deterministicErrors")
    for raw in structural:
        deterministic.append(_issue("phase2Structural", raw))
    for item in standalone.issues:
        deterministic.append(_issue(
            "phase2Standalone",
            item.diagnostic,
            node_name=item.node_name,
            step_id=(
                getattr(item, "step_id", "")
                if source_grounding_mode == "formal_steps" else ""
            ),
        ))

    validation = BlueprintValidation(
        lean_result=lean_result,
        semantic_issues=semantic_issues,
        structural_errors=structural,
        standalone_report=standalone,
    )
    semantic: list[dict[str, Any]] = []
    if formal_blueprint.nodes and _semantic_audit_eligible(validation):
        validation = _with_semantic_audit(
            validation,
            formal_blueprint,
            client=client,
            model=model,
            informal_statement=informal_statement,
            informal_proof=informal_proof,
            claimed_answer=claimed_answer,
            semantic_manifest=semantic_manifest,
            source_grounding_mode=source_grounding_mode,
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
            decompiler_cache=decompiler_cache,
            comparator_cache=comparator_cache,
            tracer=tracer,
            thm_name=thm_name,
            round_index=round_index,
        )
        decompiler = validation.formal_decompiler_result
        comparator = validation.strict_comparator_result
        if decompiler is not None:
            for node_name in decompiler.vacuous_nodes:
                node = formal_blueprint.node_by_name(node_name)
                semantic.append(_issue(
                    "vacuousFormalNode",
                    "Formal Decompiler classified this node as vacuous",
                    node_name=node_name,
                    step_id=(
                        node.source_step_id
                        if node and source_grounding_mode == "formal_steps" else ""
                    ),
                ))
        if comparator is not None:
            defects = (
                whole_cot_comparator_defects(comparator)
                if source_grounding_mode == "whole_cot"
                else comparator_defects(comparator)
            )
            for defect in defects:
                names = list(defect.get("node_names") or ())
                semantic.append(_issue(
                    str(defect.get("category") or "semanticDefect"),
                    f"{defect.get('requirement')} Reason: {defect.get('reason')}",
                    node_name=",".join(names),
                    step_id=(
                        "" if source_grounding_mode == "whole_cot"
                        else str(defect.get("step_id") or "")
                    ),
                ))
            if not comparator.passed and not defects:
                semantic.append(_issue(
                    "semanticComparatorRejected",
                    "The strict semantic comparator rejected the candidate.",
                ))
            unreachable_items = (
                comparator.unreachable_nodes
                if source_grounding_mode == "whole_cot"
                else comparator.unreachable_steps
            )
            for item in unreachable_items:
                if item.get("justified_side_branch"):
                    warnings.append(_issue(
                        "justifiedSideBranch",
                        item.get("reason"),
                        node_name=(
                            str(item.get("node_name") or "")
                            if source_grounding_mode == "whole_cot" else ""
                        ),
                        step_id=(
                            "" if source_grounding_mode == "whole_cot"
                            else str(item.get("step_id") or "")
                        ),
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
    cot_manifest_json: str,
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
    source_grounding_mode: str = "formal_steps",
) -> Blueprint:
    if max_turns <= 0:
        raise ValueError("generation max_turns must be positive")
    if source_grounding_mode not in {"formal_steps", "whole_cot"}:
        raise ValueError("source_grounding_mode must be formal_steps or whole_cot")
    semantic_manifest = (
        parse_cot_manifest(cot_manifest_json)
        if source_grounding_mode == "formal_steps" else None
    )
    source_proof = informal_proof
    if source_grounding_mode == "formal_steps":
        source_proof = _render_step_grounded_proof(cot_manifest_json, include_ir=False)
    client = make_client(model)
    previous_code = ""
    previous_feedback = ""
    candidates: list[str] = []
    labels: list[str] = []
    rounds: list[GenerationRound] = []
    standalone_cache: dict[str, Any] = {}
    decompiler_cache: dict[str, Any] = {}
    comparator_cache: dict[str, Any] = {}
    latest_blueprint: Blueprint | None = None
    latest_validation: BlueprintValidation | None = None
    latest_deterministic: tuple[dict[str, Any], ...] = ()
    latest_semantic: tuple[dict[str, Any], ...] = ()
    latest_warnings: tuple[dict[str, Any], ...] = ()

    for round_index in range(1, max_turns + 1):
        messages = _messages(
            target_name=target_name,
            informal_statement=informal_statement,
            informal_proof=source_proof,
            claimed_answer=claimed_answer,
            previous_blueprint=previous_code,
            previous_feedback=previous_feedback,
            source_grounding_mode=source_grounding_mode,
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
                    "sourceGroundingMode": source_grounding_mode,
                    "splitterInvoked": source_grounding_mode == "formal_steps",
                    "inputTokens": input_tokens,
                    "maxCompletionTokens": completion_budget,
                    "enableThinking": enable_thinking,
                    "temperature": temperature,
                    "topP": top_p,
                    "topK": top_k,
                    "minP": min_p,
                    "presencePenalty": presence_penalty,
                    "repetitionPenalty": repetition_penalty,
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
            (
                latest_blueprint,
                latest_validation,
                latest_deterministic,
                latest_semantic,
                latest_warnings,
            ) = _validate_round(
                code,
                target_name=target_name,
                compiler=compiler,
                semantic_manifest=semantic_manifest,
                informal_statement=informal_statement,
                informal_proof=informal_proof,
                claimed_answer=claimed_answer,
                source_grounding_mode=source_grounding_mode,
                standalone_concurrency=standalone_concurrency,
                client=client,
                model=model,
                decompiler_max_tokens=decompiler_max_tokens,
                comparator_max_tokens=comparator_max_tokens,
                semantic_format_attempts=semantic_format_attempts,
                semantic_audit_enable_thinking=semantic_audit_enable_thinking,
                semantic_audit_temperature=semantic_audit_temperature,
                semantic_audit_top_p=semantic_audit_top_p,
                semantic_audit_top_k=semantic_audit_top_k,
                semantic_audit_min_p=semantic_audit_min_p,
                semantic_audit_presence_penalty=semantic_audit_presence_penalty,
                semantic_audit_repetition_penalty=semantic_audit_repetition_penalty,
                tracer=tracer,
                thm_name=thm_name,
                round_index=round_index,
                standalone_cache=standalone_cache,
                decompiler_cache=decompiler_cache,
                comparator_cache=comparator_cache,
            )
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
        for kind, inventory, ok in (
            ("phase1DeterministicValidation", latest_deterministic, not latest_deterministic),
            ("phase1SemanticAudit", latest_semantic, not latest_semantic),
            ("phase1WarningInventory", latest_warnings, not latest_warnings),
        ):
            if tracer:
                tracer.emit(TraceEvent(
                    kind=kind,
                    thm_name=thm_name,
                    turn=round_index,
                    args={"round": round_index, "count": len(inventory), "issues": list(inventory)},
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
            final_details["sourceGroundingMode"] = source_grounding_mode
            final_details["splitterInvoked"] = source_grounding_mode == "formal_steps"
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
        "sourceGroundingMode": source_grounding_mode,
        "splitterInvoked": source_grounding_mode == "formal_steps",
        "classification": "structuralRejected" if deterministic_failed else "semanticRejected",
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

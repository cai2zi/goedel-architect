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
from mathlib_retrieval import MathlibRetrieval
from semantic_audit import (
    CANONICAL_COMPACT_WHOLE_COT_PROMPT_VERSION,
    CANONICAL_DIRECT_WHOLE_COT_PROMPT_VERSION,
    COMPACT_WHOLE_COT_PROMPT_VERSION,
    DIRECT_WHOLE_COT_PROMPT_VERSION,
    FORMAL_DECOMPILER_PROMPT_VERSION,
    FormalDecompilerResult,
    CanonicalWholeCotComparatorResult,
    JointWholeCotAuditResult,
    WholeCotComparatorResult,
    JOINT_WHOLE_COT_PROMPT_VERSION,
    WHOLE_COT_PROMPT_VERSION,
    build_formal_view,
    compact_whole_cot_comparator_messages,
    canonical_compact_whole_cot_comparator_messages,
    canonical_direct_whole_cot_comparator_messages,
    direct_whole_cot_comparator_messages,
    joint_whole_cot_audit_messages,
    run_formal_decompiler,
    run_compact_whole_cot_comparator,
    run_canonical_compact_whole_cot_comparator,
    run_canonical_direct_whole_cot_comparator,
    run_direct_whole_cot_comparator,
    run_joint_whole_cot_audit,
    run_whole_cot_comparator,
    semantic_audit_cache_key,
    unreachable_proof_node_names,
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

Every Blueprint definition is global context shared by every proof node.
Definitions need not be listed in `sorry_using`; if a definition is already
listed, it may remain and is not a proof-graph dependency. Do not add a
definition to `sorry_using` merely to make it root-reachable. A definition's
mere existence does not ground the final theorem: required source objects and
relations must be referenced, constrained, or related by the root or by a
root-supporting proposition.

Use only these Blueprint declaration forms: `def`, `noncomputable def`,
`abbrev`, `lemma`, and `theorem`. The pipeline rejects `axiom`, `structure`,
`instance`, `class`, `inductive`, `variable`, `section`,
`noncomputable section`, `namespace`, `notation`, `macro`, `syntax`, and
`partial def`. Replace a structure with a tuple or type alias plus accessor
definitions; replace an instance with an explicit Bool predicate or a decision
procedure inside a definition; replace an axiom with theorem binders or a
Blueprint lemma; put variables in each declaration's explicit binders; and use
`noncomputable def` on each affected declaration instead of a noncomputable
section. `sorry_using` may name only local `@[blueprint]` nodes, never Mathlib
declarations.

For bounded operators use `∑ x ∈ s, f x`, `∏ x ∈ s, f x`,
`Finset.sum s (fun x => f x)`, or `Finset.prod s (fun x => f x)`. Never use
`∑ x in s, f x` or `∏ x in s, f x`: `open scoped BigOperators` is supplied by
the pipeline but does not make that binder syntax valid.

`ℝ × ℝ` and `ℚ × ℚ` are coordinate carriers. Their default product `dist` and
`norm` use the max/sup metric, not the Euclidean metric. Do not use unqualified
product `dist` or `norm` to formalize Euclidean length, circles, or angles.
Prefer `ℚ` and squared lengths when the COT permits, with explicit definitions
such as `(P.1 - Q.1)^2 + (P.2 - Q.2)^2` for squared Euclidean distance and
`u.1 * v.1 + u.2 * v.2` for the dot product. Use
`Real.sqrt (sqEuclideanDist2 P Q)` only when ordinary Euclidean distance is
required.

`title`, `statement`, and `proof` metadata are optional. Call `lean_compile`
exactly once with the entire replacement file. Do not return prose or a partial
declaration.
"""


ANONYMOUS_NODE_NAMING_SUFFIX = r"""

## Opaque Blueprint node naming contract

Top-level declaration names are opaque identifiers and must not describe
their mathematical meaning. Name definitions and abbrevs consecutively
`d1`, `d2`, ... in source order. Name every non-root lemma/theorem
consecutively `n1`, `n2`, ... in source order. The unique root theorem is
`n_final`. Use these same opaque names in every declaration reference and
`sorry_using` dependency. Do not add semantic namespaces or semantic
`title`/`statement`/`proof` metadata. Ordinary mathematical binders such as
`x`, `n`, `A`, or `B` are allowed, but do not introduce descriptive local
answer aliases such as `shortest_path` or `expected_value`.
"""


ANSWER_PREASSIGNED_REPAIR_GUIDANCE = r"""

## Semantic repair pattern: computed object was preassigned

This synthetic example is unrelated to the current problem. Do not copy its
names or numeric literals. Do not repair `computed := answer` by changing it to
`computed := 0`. The object to be computed must be an explicit lemma/root
binder, and the source constraints must occur in the theorem type. The claimed
answer may occur in a derived conclusion, but not in a definition that fixes
the computed object. Do not manufacture a reflexive equality by applying a
parameterized definition directly to the answer value.

```lean
@[blueprint]
def demo_relations (unknown target : ℤ) : Prop :=
  5 * unknown + 3 = 38 ∧ target = 2 * unknown

@[blueprint]
lemma demo_solve
    (unknown target : ℤ)
    (h : demo_relations unknown target) :
    unknown = 7 := by sorry_using []

@[blueprint]
theorem {{target_name}}
    (unknown target : ℤ)
    (h : demo_relations unknown target) :
    target = 14 := by sorry_using [demo_solve]
```
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
    semantic_stage: int = 1
    semantic_audit_ordinal: int | None = None
    semantic_audit_invoked: bool = False
    semantic_anchor_round: int | None = None
    structural_input_round: int | None = None
    attempt_role: str = "initial"
    submitted_candidate_hash: str = ""
    canonical_candidate_hash: str = ""
    context_fallback_applied: bool = False
    builder_message_content: str = ""
    finish_reason: str = ""

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
            "semanticStage": self.semantic_stage,
            "semanticAuditOrdinal": self.semantic_audit_ordinal,
            "semanticAuditInvoked": self.semantic_audit_invoked,
            "semanticAnchorRound": self.semantic_anchor_round,
            "structuralInputRound": self.structural_input_round,
            "attemptRole": self.attempt_role,
            "submittedCandidateHash": self.submitted_candidate_hash,
            "canonicalCandidateHash": self.canonical_candidate_hash,
            "contextFallbackApplied": self.context_fallback_applied,
            "builderMessageContent": self.builder_message_content,
            "finishReason": self.finish_reason,
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
    strict_comparator_result: WholeCotComparatorResult | CanonicalWholeCotComparatorResult | None = None
    joint_audit_result: JointWholeCotAuditResult | None = None
    semantic_audit_protocol: str = WHOLE_COT_PROMPT_VERSION
    graph_shadow_unreachable_nodes: tuple[str, ...] = ()

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

    def __init__(
        self,
        message: str,
        validation: BlueprintValidation,
        blueprint: Blueprint,
    ):
        super().__init__(message)
        self.validation = validation
        self.blueprint = blueprint


def validation_details(validation: BlueprintValidation) -> dict[str, Any]:
    standalone = validation.standalone_report
    audit = None
    if validation.strict_comparator_result:
        audit = {
            "protocol": validation.semantic_audit_protocol,
            "mode": validation.semantic_audit_mode,
            "actualRequestCount": validation.semantic_request_count,
            "cacheHits": dict(validation.semantic_cache_hits or {}),
            "outputBudget": validation.semantic_output_budget,
            "classification": "strictAccepted" if validation.passed else "semanticRejected",
        }
        if validation.formal_decompiler_result is not None:
            audit["formalDecompiler"] = validation.formal_decompiler_result.to_dict()
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
        "graphShadow": {
            "unreachableNodeNames": list(validation.graph_shadow_unreachable_nodes),
            "hasUnreachableNodes": bool(validation.graph_shadow_unreachable_nodes),
        },
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
    semantic_comparator_protocol: str,
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
    unreachable_shadow = unreachable_proof_node_names(view)
    validation.graph_shadow_unreachable_nodes = unreachable_shadow
    if tracer:
        tracer.emit(TraceEvent(
            kind="phase1GraphShadow", thm_name=thm_name, turn=round_index,
            args={
                "round": round_index,
                "unreachableNodeNames": list(unreachable_shadow),
                "hasUnreachableNodes": bool(unreachable_shadow),
            },
            ok=not bool(unreachable_shadow),
        ))

    if semantic_audit_mode == "direct":
        canonical_v2 = semantic_comparator_protocol == "canonical_v2"
        messages = (
            canonical_direct_whole_cot_comparator_messages(
                informal_statement, informal_proof, claimed_answer, view,
            ) if canonical_v2 else direct_whole_cot_comparator_messages(
                informal_statement, informal_proof, claimed_answer, view,
            )
        )
        protocol = (
            CANONICAL_DIRECT_WHOLE_COT_PROMPT_VERSION
            if canonical_v2 else DIRECT_WHOLE_COT_PROMPT_VERSION
        )
        cache_key = semantic_audit_cache_key(
            model, messages, version=protocol,
        )
        comparator = comparator_cache.get(cache_key)
        comparator_hit = comparator is not None
        validation.semantic_audit_invoked = True
        validation.semantic_audit_mode = "direct"
        validation.semantic_audit_protocol = protocol
        validation.semantic_cache_hits = {"wholeCotComparator": comparator_hit}
        if comparator is None:
            try:
                comparator_runner = (
                    run_canonical_direct_whole_cot_comparator
                    if canonical_v2 else run_direct_whole_cot_comparator
                )
                comparator = comparator_runner(
                    client, model,
                    informal_statement=informal_statement,
                    informal_proof=informal_proof,
                    claimed_answer=claimed_answer,
                    view=view, max_tokens=strict_comparator_max_tokens,
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
            semantic_audit_mode="direct",
            semantic_request_count=0 if comparator_hit else len(comparator.attempts),
            semantic_cache_hits={"wholeCotComparator": comparator_hit},
            semantic_output_budget=None,
            formal_decompiler_result=None,
            strict_comparator_result=comparator,
            semantic_audit_protocol=protocol,
            graph_shadow_unreachable_nodes=unreachable_shadow,
        )

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
            graph_shadow_unreachable_nodes=unreachable_shadow,
        )

    validation.semantic_audit_invoked = True
    compact = semantic_audit_mode == "compact_separate"
    canonical_v2 = compact and semantic_comparator_protocol == "canonical_v2"
    validation.semantic_audit_mode = semantic_audit_mode
    validation.semantic_audit_protocol = (
        COMPACT_WHOLE_COT_PROMPT_VERSION if compact else WHOLE_COT_PROMPT_VERSION
    )
    decompiler_cache_key = f"{FORMAL_DECOMPILER_PROMPT_VERSION}:{view.sha256}"
    decompiler = decompiler_cache.get(decompiler_cache_key)
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
                compact=compact,
            )
        except Exception as exc:
            validation.semantic_request_count = max(
                1, len(getattr(exc, "attempts", ()) or ()),
            )
            raise
        decompiler_cache[decompiler_cache_key] = decompiler
        validation.semantic_request_count = len(decompiler.attempts)
    messages = (
        canonical_compact_whole_cot_comparator_messages(
            informal_statement, informal_proof, claimed_answer, view, decompiler,
        ) if canonical_v2 else compact_whole_cot_comparator_messages(
            informal_statement, informal_proof, claimed_answer, view, decompiler,
        )
        if compact else whole_cot_comparator_messages(
            informal_statement, informal_proof, claimed_answer, view, decompiler,
        )
    )
    protocol = (
        CANONICAL_COMPACT_WHOLE_COT_PROMPT_VERSION if canonical_v2 else
        COMPACT_WHOLE_COT_PROMPT_VERSION if compact else WHOLE_COT_PROMPT_VERSION
    )
    cache_key = semantic_audit_cache_key(
        model, messages, version=protocol,
    )
    comparator = comparator_cache.get(cache_key)
    comparator_hit = comparator is not None
    validation.semantic_cache_hits["wholeCotComparator"] = comparator_hit
    if comparator is None:
        try:
            comparator_runner = (
                run_canonical_compact_whole_cot_comparator if canonical_v2 else
                run_compact_whole_cot_comparator if compact else
                run_whole_cot_comparator
            )
            comparator = comparator_runner(
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
        semantic_audit_mode=semantic_audit_mode,
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
        semantic_audit_protocol=protocol,
        graph_shadow_unreachable_nodes=unreachable_shadow,
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
    """Classify an audited candidate; structural failures are never terminal here."""
    if deterministic_error_count:
        return None
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
    *,
    blueprint: Blueprint | None = None,
    mathlib_search_context: Sequence[dict[str, Any]] = (),
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
    contract_context = _validated_contract_context(deterministic, blueprint)
    if contract_context:
        sections.append(
            "VALIDATED_CONTRACT_CONTEXT\n"
            + "\n".join(f"- {line}" for line in contract_context)
        )
    if mathlib_search_context:
        lines = [
            "MATHLIB_SEARCH_CONTEXT",
            "These search hits are candidates, not a repair conclusion. Use only a name "
            "whose displayed type fits, never add a Mathlib name to `sorry_using`, and "
            "submit a complete Blueprint that passes formal compilation again.",
        ]
        for item in mathlib_search_context:
            lines.append(f"- Compiler-reported symbol `{item['symbol']}`:")
            results = item.get("results") or []
            if not results:
                lines.append("  - no candidate returned")
            for result in results:
                lines.append(
                    f"  - `{result['name']}` : `{result.get('type') or '<type unavailable>'}`"
                )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _active_repair_codes(
    semantic: Sequence[dict[str, Any]],
) -> tuple[str, ...]:
    supported = {"answerPreassigned", "targetCoverageIncomplete"}
    codes = {
        str(item.get("code")) for item in semantic
        if item.get("code") in supported
    }
    for item in semantic:
        shortcut = item.get("shortcut")
        if (
            item.get("code") == "derivationShortcut"
            and isinstance(shortcut, dict)
            and shortcut.get("pattern") == "preassigned"
            and shortcut.get("object_role") == "target"
        ):
            codes.add("derivationShortcut:preassigned:target")
    return tuple(sorted(codes))


def _semantic_feedback_state(
    last_errors: Sequence[dict[str, Any]],
    last_round: int | None,
    *,
    semantic_audit_invoked: bool,
    current_errors: Sequence[dict[str, Any]],
    current_round: int,
) -> tuple[tuple[dict[str, Any], ...], int | None, tuple[str, ...], bool]:
    if semantic_audit_invoked:
        errors = tuple(current_errors)
        return errors, current_round, _active_repair_codes(errors), False
    errors = tuple(last_errors)
    return errors, last_round, _active_repair_codes(errors), bool(errors)


_FORBIDDEN_REPLACEMENTS = {
    "structure": "Use a tuple or type alias plus Blueprint accessor definitions.",
    "instance": "Use an explicit Bool predicate or a decision procedure inside a definition.",
    "axiom": "Move assumptions into theorem binders or state them as a Blueprint lemma.",
    "variable": "Put every variable in the explicit binders of each declaration.",
    "noncomputable section": "Mark each affected declaration individually as `noncomputable def`.",
    "section": "Remove the section and put parameters in explicit declaration binders.",
    "class": "Use an explicit tuple/type alias and ordinary definitions instead of a class.",
    "inductive": "Use an allowed existing type, tuple/type alias, or ordinary definition.",
    "namespace": "Use top-level Blueprint declarations with unique names.",
    "notation": "Use the underlying Lean expression directly.",
    "local notation": "Use the underlying Lean expression directly.",
    "macro": "Use the expanded Lean expression directly.",
    "syntax": "Use existing Lean syntax directly.",
    "partial def": "Use a terminating ordinary `def` or reformulate the finite object.",
}
_FORBIDDEN_DIAGNOSTIC_RE = re.compile(
    r"^Safeguard rejected: forbidden construct `([^`]+)` is not allowed\.$"
)
_NONCOMPUTABLE_SUGGESTION = "consider marking it as 'noncomputable'"
_MATHLIB_DIAGNOSTIC_PATTERNS = (
    re.compile(r"Unknown constant [`']([^`']+)[`']"),
    re.compile(r"Unknown identifier [`']([^`']+)[`']"),
    re.compile(r"Invalid field [`']([^`']+)[`']"),
)


def _diagnostic_json(message: Any) -> dict[str, Any] | None:
    if not isinstance(message, str):
        return None
    try:
        value = json.loads(message)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _diagnostic_text(message: Any) -> str:
    payload = _diagnostic_json(message)
    if payload is None:
        return str(message or "")
    return str(payload.get("data") or payload.get("message") or "")


def _diagnostic_line(message: Any) -> int | None:
    payload = _diagnostic_json(message)
    if payload is None:
        return None
    for key in ("pos", "startPos", "position"):
        position = payload.get(key)
        if isinstance(position, dict) and isinstance(position.get("line"), int):
            return int(position["line"])
    return None


def _validated_contract_context(
    deterministic: Sequence[dict[str, Any]],
    blueprint: Blueprint | None,
) -> tuple[str, ...]:
    """Add guidance only when a validator/compiler proved the exact contract."""
    context: list[str] = []
    for item in deterministic:
        message = str(item.get("message") or "")
        forbidden = _FORBIDDEN_DIAGNOSTIC_RE.fullmatch(message)
        if forbidden:
            construct = forbidden.group(1)
            replacement = _FORBIDDEN_REPLACEMENTS.get(construct)
            if replacement:
                context.append(
                    f"The safeguard proved that `{construct}` is forbidden. {replacement}"
                )
        if item.get("code") == "unannotatedLocalDeclaration":
            context.append(
                "Canonicalization keeps only imports, `open`/`open scoped`, `set_option`, "
                "and annotated Blueprint nodes. Convert each other local declaration into "
                "an allowed `@[blueprint]` node; replace `noncomputable section` with "
                "individual `noncomputable def` declarations."
            )
        diagnostic = _diagnostic_text(message)
        line = _diagnostic_line(message)
        if (
            blueprint is not None
            and _NONCOMPUTABLE_SUGGESTION in diagnostic
            and line is not None
        ):
            source_line = line
            if (
                "import Architect" in blueprint.lean_file
                and "import GoedelArch" not in blueprint.lean_file
            ):
                # KiminaLeanCompiler.check_blueprint inserts GoedelArch after
                # Architect before compilation, shifting declaration diagnostics.
                source_line -= 1
            node = next((
                candidate for candidate in blueprint.nodes
                if candidate.lean_start_line <= source_line <= candidate.lean_end_line
            ), None)
            if node is not None:
                context.append(
                    f"Lean explicitly recommends marking Blueprint node `{node.name}` "
                    "as `noncomputable`."
                )
    return tuple(dict.fromkeys(context))


def _eligible_mathlib_symbols(
    deterministic: Sequence[dict[str, Any]],
) -> tuple[str, ...]:
    symbols: list[str] = []
    for item in deterministic:
        if item.get("code") != "canonicalLean" or item.get("stage") != "canonical_lean":
            continue
        diagnostic = _diagnostic_text(item.get("message"))
        for pattern in _MATHLIB_DIAGNOSTIC_PATTERNS:
            match = pattern.search(diagnostic)
            if match:
                symbols.append(match.group(1).strip())
                break
    return tuple(dict.fromkeys(symbol for symbol in symbols if symbol))


def _search_result_dict(result: Any) -> dict[str, str] | None:
    name = getattr(result, "name", None)
    if name is None and isinstance(result, dict):
        name = result.get("name")
    if not name:
        return None
    type_sig = getattr(result, "type_sig", None)
    if type_sig is None and isinstance(result, dict):
        type_sig = result.get("type_sig", result.get("type", ""))
    return {"name": str(name), "type": str(type_sig or "")}


def _run_phase1_mathlib_search(
    deterministic: Sequence[dict[str, Any]],
    *,
    retrieval: Any,
    cache: dict[str, tuple[dict[str, str], ...]],
    max_queries: int,
    k: int,
) -> tuple[dict[str, Any], ...]:
    reports: list[dict[str, Any]] = []
    for symbol in _eligible_mathlib_symbols(deterministic)[:max_queries]:
        started = time.monotonic()
        cache_hit = symbol in cache
        error = ""
        if cache_hit:
            results = cache[symbol]
        else:
            try:
                raw_results = retrieval.search(symbol, k=k)
                results = tuple(
                    item for raw in raw_results[:k]
                    if (item := _search_result_dict(raw)) is not None
                )
                cache[symbol] = results
            except Exception as exc:  # Retrieval is optional evidence, never infrastructure.
                results = ()
                error = f"{type(exc).__name__}: {exc}"
        reports.append({
            "symbol": symbol,
            "query": symbol,
            "results": list(results),
            "latencyMs": (time.monotonic() - started) * 1000,
            "cacheHit": cache_hit,
            "error": error,
        })
    return tuple(reports)


def _messages(
    *,
    target_name: str,
    informal_statement: str,
    informal_proof: str,
    claimed_answer: str,
    previous_blueprint: str,
    previous_feedback: str,
    latest_structural_blueprint: str = "",
    latest_structural_feedback: str = "",
    structural_context_omitted: bool = False,
    active_repair_codes: Sequence[str] = (),
    prompt_profile: str = "whole_cot_minimal",
    node_naming: str = "semantic",
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
            "\n\n## Last semantically audited Blueprint\n```lean\n"
            + previous_blueprint.rstrip()
            + "\n```\n\n## Unresolved semantic diagnostics\n"
            + previous_feedback
        )
        if (
            "answerPreassigned" in active_repair_codes
            or "derivationShortcut:preassigned:target" in active_repair_codes
        ):
            user += render(
                ANSWER_PREASSIGNED_REPAIR_GUIDANCE,
                target_name=target_name,
            )
    if latest_structural_feedback:
        user += "\n\n## Latest structural diagnostics\n" + latest_structural_feedback
        if latest_structural_blueprint:
            user += (
                "\n\n## Latest structurally rejected Blueprint\n```lean\n"
                + latest_structural_blueprint.rstrip()
                + "\n```"
            )
        elif structural_context_omitted:
            user += (
                "\n\nThe latest structurally rejected Blueprint body was omitted because "
                "the full repair context exceeded the model context budget. Repair from "
                "the semantic anchor and both diagnostic inventories."
            )
    system = (
        ROBUSTPA_BLUEPRINT_MINIMAL_SYSTEM_PROMPT
        if prompt_profile == "whole_cot_minimal"
        else ROBUSTPA_BLUEPRINT_SYSTEM_PROMPT
    )
    system += WHOLE_COT_GENERATION_SYSTEM_SUFFIX
    if node_naming == "anonymous":
        system += ANONYMOUS_NODE_NAMING_SUFFIX
    return [
        {"role": "system", "content": (
            system
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


def _contract_errors(
    blueprint: Blueprint,
    target_name: str,
    *,
    node_naming: str = "semantic",
) -> list[dict[str, Any]]:
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
    if node_naming == "anonymous":
        definitions = [node.name for node in blueprint.nodes if node.kind == "definition"]
        expected_definitions = [f"d{index}" for index in range(1, len(definitions) + 1)]
        if definitions != expected_definitions:
            errors.append(_issue(
                "anonymousDefinitionNames",
                f"definition names must be consecutive in source order: {expected_definitions}",
                stage="parse_basic",
            ))
        proof_nodes = [
            node.name for node in blueprint.nodes
            if node.kind != "definition" and node.name != target_name
        ]
        expected_proofs = [f"n{index}" for index in range(1, len(proof_nodes) + 1)]
        if proof_nodes != expected_proofs:
            errors.append(_issue(
                "anonymousProofNames",
                f"non-root proof names must be consecutive in source order: {expected_proofs}",
                stage="parse_basic",
            ))
        root = blueprint.nodes_by_name().get(target_name)
        if target_name != "n_final" or root is None or root.kind != "theorem":
            errors.append(_issue(
                "anonymousRootName",
                "anonymous generation requires one theorem root named n_final",
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
    semantic_comparator_protocol: str,
    node_naming: str,
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
    deterministic: list[dict[str, Any]] = _contract_errors(
        candidate, target_name, node_naming=node_naming,
    )
    warnings: list[dict[str, Any]] = []
    if deterministic:
        validation = _mechanical_validation(
            whole_result=_failed_compile("whole_file_lean not run"),
            stage_reached="parse_basic", failure_stage="parse_basic",
            semantic_audit_mode=semantic_audit_mode,
        )
        return candidate, validation, _deduplicate(deterministic), (), ()

    try:
        formal_blueprint = canonicalize_blueprint(candidate, list(candidate.nodes))
    except ValueError as exc:
        deterministic.append(_issue("canonicalRebuild", str(exc), stage="canonical_rebuild"))
        validation = _mechanical_validation(
            whole_result=_failed_compile("formal blueprint Lean not run"),
            stage_reached="canonical_rebuild",
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
            whole_result=_failed_compile("formal blueprint Lean not run"),
            stage_reached="phase2_contract",
            failure_stage="phase2_contract", structural=structural,
            semantic_audit_mode=semantic_audit_mode,
        )
        return formal_blueprint, validation, _deduplicate(deterministic), (), _deduplicate(warnings)

    formal_result = compiler.check_blueprint(formal_blueprint.lean_file, target_name)
    if formal_result.failure_kind == "infra":
        raise KiminaInfrastructureError(
            "\n".join(formal_result.diagnostics) or formal_result.raw_output[-2000:]
        )
    warnings.extend(_non_sorry_warnings(
        formal_result.warnings, "canonicalLeanWarning", stage="canonical_lean",
    ))
    if not formal_result.success:
        deterministic.extend(
            _issue("canonicalLean", value, stage="canonical_lean")
            for value in formal_result.diagnostics
        )
        validation = _mechanical_validation(
            whole_result=formal_result, canonical_result=formal_result,
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
            whole_result=formal_result, canonical_result=formal_result,
            stage_reached="phase2_standalone", failure_stage="phase2_standalone",
            structural=structural, standalone=standalone,
            semantic_audit_mode=semantic_audit_mode,
        )
        return formal_blueprint, validation, _deduplicate(deterministic), (), _deduplicate(warnings)

    semantic_issues = validate_blueprint_fidelity(
        formal_blueprint, claimed_answer=claimed_answer,
    )
    validation = _mechanical_validation(
        whole_result=formal_result, canonical_result=formal_result,
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
            semantic_comparator_protocol=semantic_comparator_protocol,
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
        raise SemanticAuditExecutionError(str(exc), validation, formal_blueprint) from exc

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
            observed = str(defect.get("observed_formal_behavior") or "").strip()
            message = f"{defect.get('requirement')}"
            if observed:
                message += f" Observed formal behavior: {observed}."
            message += f" Reason: {defect.get('reason')}"
            issue = _issue(
                str(defect.get("category") or "semanticDefect"),
                message,
                stage="whole_cot_comparator", node_name=",".join(names),
            )
            if defect.get("issue_id") is not None:
                issue["canonicalIssueId"] = defect["issue_id"]
            if defect.get("shortcut") is not None:
                issue["shortcut"] = defect["shortcut"]
            semantic.append(issue)
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
    max_semantic_audits: int,
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
    semantic_comparator_protocol: str = "legacy_v1",
    joint_semantic_audit_max_tokens: int = 32768,
    prompt_profile: str = "whole_cot_minimal",
    node_naming: str = "semantic",
    phase1_mathlib_search_enabled: bool = False,
    phase1_mathlib_search_max_queries_per_round: int = 2,
    phase1_mathlib_search_k: int = 3,
    phase1_mathlib_search_timeout_s: float = 15.0,
    mathlib_retrieval: Any | None = None,
) -> Blueprint:
    if max_semantic_audits <= 0:
        raise ValueError("generation max_semantic_audits must be positive")
    if prompt_profile not in {"whole_cot_minimal", "standard"}:
        raise ValueError("prompt_profile must be whole_cot_minimal or standard")
    if semantic_audit_mode not in {"separate", "compact_separate", "direct", "joint"}:
        raise ValueError(
            "semantic_audit_mode must be separate, compact_separate, direct, or joint"
        )
    if semantic_comparator_protocol not in {"legacy_v1", "canonical_v2"}:
        raise ValueError(
            "semantic_comparator_protocol must be legacy_v1 or canonical_v2"
        )
    if semantic_comparator_protocol == "canonical_v2" and semantic_audit_mode not in {
        "direct", "compact_separate",
    }:
        raise ValueError(
            "canonical_v2 is supported only for direct and compact_separate"
        )
    if node_naming not in {"semantic", "anonymous"}:
        raise ValueError("node_naming must be semantic or anonymous")
    if joint_semantic_audit_max_tokens <= 0:
        raise ValueError("joint_semantic_audit_max_tokens must be positive")
    if phase1_mathlib_search_max_queries_per_round <= 0:
        raise ValueError("phase1_mathlib_search_max_queries_per_round must be positive")
    if phase1_mathlib_search_k <= 0:
        raise ValueError("phase1_mathlib_search_k must be positive")
    if phase1_mathlib_search_timeout_s <= 0:
        raise ValueError("phase1_mathlib_search_timeout_s must be positive")
    client = make_client(model)
    phase1_retrieval = mathlib_retrieval
    if phase1_mathlib_search_enabled and phase1_retrieval is None:
        phase1_retrieval = MathlibRetrieval(timeout=phase1_mathlib_search_timeout_s)

    candidates: list[str] = []
    labels: list[str] = []
    rounds: list[GenerationRound] = []
    standalone_cache: dict[str, Any] = {}
    decompiler_cache: dict[str, Any] = {}
    comparator_cache: dict[str, Any] = {}
    joint_cache: dict[str, JointWholeCotAuditResult] = {}
    mathlib_search_cache: dict[str, tuple[dict[str, str], ...]] = {}
    previous_search_symbols: tuple[str, ...] = ()

    generation_attempt_index = 0
    semantic_audit_count = 0
    semantic_anchor_blueprint: Blueprint | None = None
    semantic_anchor_validation: BlueprintValidation | None = None
    semantic_anchor_errors: tuple[dict[str, Any], ...] = ()
    semantic_anchor_warnings: tuple[dict[str, Any], ...] = ()
    semantic_anchor_round: int | None = None
    semantic_anchor_details: dict[str, Any] = {}
    latest_structural_blueprint = ""
    latest_structural_errors: tuple[dict[str, Any], ...] = ()
    latest_structural_warnings: tuple[dict[str, Any], ...] = ()
    latest_structural_round: int | None = None
    latest_structural_search: tuple[dict[str, Any], ...] = ()

    while semantic_audit_count < max_semantic_audits:
        generation_attempt_index += 1
        round_index = generation_attempt_index
        semantic_stage = semantic_audit_count + 1
        input_anchor_round = semantic_anchor_round
        input_structural_round = latest_structural_round
        active_repair_codes = _active_repair_codes(semantic_anchor_errors)
        anchor_feedback = _feedback(
            (), semantic_anchor_errors, semantic_anchor_warnings,
            blueprint=semantic_anchor_blueprint,
        ) if semantic_anchor_blueprint is not None else ""
        structural_feedback = _feedback(
            latest_structural_errors, (), latest_structural_warnings,
            mathlib_search_context=latest_structural_search,
        ) if latest_structural_round is not None else ""
        messages = _messages(
            target_name=target_name,
            informal_statement=informal_statement,
            informal_proof=informal_proof,
            claimed_answer=claimed_answer,
            previous_blueprint=(
                semantic_anchor_blueprint.lean_file
                if semantic_anchor_blueprint is not None else ""
            ),
            previous_feedback=anchor_feedback,
            latest_structural_blueprint=latest_structural_blueprint,
            latest_structural_feedback=structural_feedback,
            active_repair_codes=active_repair_codes,
            prompt_profile=prompt_profile,
            node_naming=node_naming,
        )
        input_tokens, completion_budget = generation_request_budget(
            messages, tokenizer_path=tokenizer_path,
            model_max_context=model_max_context,
            safety_margin=context_safety_margin, tools=[LEAN_COMPILE_TOOL],
        )
        context_fallback_applied = False
        if completion_budget <= 0 and latest_structural_blueprint:
            context_fallback_applied = True
            messages = _messages(
                target_name=target_name,
                informal_statement=informal_statement,
                informal_proof=informal_proof,
                claimed_answer=claimed_answer,
                previous_blueprint=(
                    semantic_anchor_blueprint.lean_file
                    if semantic_anchor_blueprint is not None else ""
                ),
                previous_feedback=anchor_feedback,
                latest_structural_blueprint="",
                latest_structural_feedback=structural_feedback,
                structural_context_omitted=True,
                active_repair_codes=active_repair_codes,
                prompt_profile=prompt_profile,
                node_naming=node_naming,
            )
            input_tokens, completion_budget = generation_request_budget(
                messages, tokenizer_path=tokenizer_path,
                model_max_context=model_max_context,
                safety_margin=context_safety_margin, tools=[LEAN_COMPILE_TOOL],
            )
        if completion_budget <= 0:
            last_candidate = (
                semantic_anchor_blueprint.lean_file
                if semantic_anchor_blueprint is not None else latest_structural_blueprint
            )
            raise BlueprintGenerationError(
                "Phase 1 input exhausts the model context window.",
                last_candidate=last_candidate,
                diagnostics=[
                    f"inputTokens={input_tokens} modelContext={model_max_context} "
                    f"safetyMargin={context_safety_margin}"
                ],
                attempt=round_index,
                failure_stage="phase1ContextBudgetExceeded",
                candidate_history=candidates,
                candidate_labels=labels,
                validation_details={
                    "semanticAuditCount": semantic_audit_count,
                    "generationRounds": [item.to_dict() for item in rounds],
                },
                generation_history=[item.to_dict() for item in rounds],
            )

        span_id = uuid.uuid4().hex
        started_ns = time.monotonic_ns()
        if tracer:
            tracer.emit(TraceEvent(
                kind="phase1GenerationStart", thm_name=thm_name,
                turn=round_index, span_id=span_id,
                args={
                    "round": round_index,
                    "semanticStage": semantic_stage,
                    "semanticAuditCount": semantic_audit_count,
                    "generationMaxSemanticAudits": max_semantic_audits,
                    "semanticAnchorRound": input_anchor_round,
                    "structuralInputRound": input_structural_round,
                    "contextFallbackApplied": context_fallback_applied,
                    "promptProfile": prompt_profile,
                    "inputTokens": input_tokens,
                    "maxCompletionTokens": completion_budget,
                    "enableThinking": enable_thinking,
                    "temperature": temperature,
                    "topP": top_p, "topK": top_k, "minP": min_p,
                    "presencePenalty": presence_penalty,
                    "repetitionPenalty": repetition_penalty,
                    "semanticAuditMode": semantic_audit_mode,
                    "semanticFeedbackSourceRound": input_anchor_round,
                    "semanticFeedbackRetained": bool(
                        input_anchor_round is not None and input_structural_round is not None
                    ),
                    "activeRepairCodes": list(active_repair_codes),
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
            client, tracer=tracer, thm_name=thm_name, phase="phase1",
            model_id=model, operation="blueprint_generation",
            trace_args={
                "round": round_index,
                "semantic_stage": semantic_stage,
                "max_completion_tokens": completion_budget,
            },
            model=model, messages=messages, tools=[LEAN_COMPILE_TOOL],
            tool_choice="required", parallel_tool_calls=False,
            temperature=temperature, top_p=top_p,
            presence_penalty=presence_penalty, seed=seed,
            max_completion_tokens=completion_budget,
            extra_body={
                "top_k": top_k, "min_p": min_p,
                "repetition_penalty": repetition_penalty,
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            },
        )
        _emit_usage(tracer, thm_name, "phase1", model, response)
        _emit_llm_response(
            tracer, thm_name=thm_name, phase="phase1", model=model,
            response=response, attempt=1, turn=round_index,
        )
        choice = response.choices[0]
        builder_message_content = str(getattr(choice.message, "content", None) or "")
        finish_reason = str(getattr(choice, "finish_reason", None) or "")
        code, submission_errors = _submitted_code(response)
        submitted_hash = hashlib.sha256(code.encode()).hexdigest() if code else ""

        current_blueprint: Blueprint | None = None
        current_validation: BlueprintValidation | None = None
        current_deterministic: tuple[dict[str, Any], ...] = ()
        current_semantic: tuple[dict[str, Any], ...] = ()
        current_warnings: tuple[dict[str, Any], ...] = ()
        canonical_hash = ""
        candidate_hash = ""
        search_reports: tuple[dict[str, Any], ...] = ()
        previous_status: list[dict[str, Any]] = []

        if submission_errors:
            current_deterministic = _deduplicate(submission_errors)
            details: dict[str, Any] = {
                "phase1MathlibSearch": {
                    "enabled": phase1_mathlib_search_enabled,
                    "eligibleSymbols": [], "queries": [],
                    "previousDiagnosticStatus": [],
                    "notRunReason": "generationSubmissionInvalid",
                }
            }
        else:
            candidates.append(code)
            labels.append(f"generation_round_{round_index}_submitted")
            try:
                (
                    current_blueprint, current_validation, current_deterministic,
                    current_semantic, current_warnings,
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
                    semantic_comparator_protocol=semantic_comparator_protocol,
                    node_naming=node_naming,
                    joint_semantic_audit_max_tokens=joint_semantic_audit_max_tokens,
                    tokenizer_path=tokenizer_path,
                    model_max_context=model_max_context,
                    context_safety_margin=context_safety_margin,
                    tracer=tracer, thm_name=thm_name, round_index=round_index,
                    standalone_cache=standalone_cache,
                    decompiler_cache=decompiler_cache,
                    comparator_cache=comparator_cache, joint_cache=joint_cache,
                )
            except SemanticAuditExecutionError as exc:
                current_blueprint = exc.blueprint
                current_validation = exc.validation
                canonical_code = current_blueprint.lean_file
                canonical_hash = hashlib.sha256(canonical_code.encode()).hexdigest()
                candidate_hash = canonical_hash
                candidates.extend([canonical_code, canonical_code])
                labels.extend([
                    f"generation_round_{round_index}_canonical",
                    f"generation_round_{round_index}",
                ])
                details = validation_details(current_validation)
                audit_error = _issue(
                    "semanticAuditError", exc,
                    stage=(
                        "joint_semantic_audit" if semantic_audit_mode == "joint"
                        else "formal_decompiler_or_comparator"
                    ),
                )
                details.update({
                    "semanticAuditError": audit_error,
                    "semanticAuditCount": semantic_audit_count,
                    "generationMaxSemanticAudits": max_semantic_audits,
                    "semanticFeedbackInputRound": input_anchor_round,
                    "structuralInputRound": input_structural_round,
                    "contextFallbackApplied": context_fallback_applied,
                })
                rounds.append(GenerationRound(
                    round_index=round_index, candidate_hash=candidate_hash,
                    input_tokens=input_tokens,
                    max_completion_tokens=completion_budget,
                    deterministic_errors=(), semantic_errors=(), warnings=(),
                    validation=dict(details), semantic_stage=semantic_stage,
                    semantic_audit_invoked=True,
                    semantic_anchor_round=input_anchor_round,
                    structural_input_round=input_structural_round,
                    attempt_role="semanticAuditError",
                    submitted_candidate_hash=submitted_hash,
                    canonical_candidate_hash=canonical_hash,
                    context_fallback_applied=context_fallback_applied,
                    builder_message_content=builder_message_content,
                    finish_reason=finish_reason,
                ))
                details.update({
                    "classification": "semanticAuditError",
                    "terminalCategory": "semanticAuditError",
                    "generationRounds": [item.to_dict() for item in rounds],
                })
                if tracer:
                    tracer.emit(TraceEvent(
                        kind="phase1FinalClassification", thm_name=thm_name,
                        turn=round_index,
                        args={"classification": "semanticAuditError", "error": audit_error},
                        ok=False,
                    ))
                raise BlueprintGenerationError(
                    "Phase 1 semantic audit exhausted its request/schema retries.",
                    last_candidate=canonical_code, diagnostics=[str(exc)],
                    attempt=round_index, failure_stage="phase1SemanticAuditError",
                    candidate_history=candidates, candidate_labels=labels,
                    validation_details=details,
                    generation_history=[item.to_dict() for item in rounds],
                ) from exc

            canonical_available = current_validation.mechanical_stage_reached not in {
                "parse_basic", "canonical_rebuild",
            }
            persisted_code = current_blueprint.lean_file
            if canonical_available:
                canonical_hash = hashlib.sha256(persisted_code.encode()).hexdigest()
                candidates.append(persisted_code)
                labels.append(f"generation_round_{round_index}_canonical")
            candidate_hash = hashlib.sha256(persisted_code.encode()).hexdigest()
            candidates.append(persisted_code)
            labels.append(f"generation_round_{round_index}")

            eligible_symbols = _eligible_mathlib_symbols(current_deterministic)
            if phase1_mathlib_search_enabled and phase1_retrieval is not None:
                search_reports = _run_phase1_mathlib_search(
                    current_deterministic, retrieval=phase1_retrieval,
                    cache=mathlib_search_cache,
                    max_queries=phase1_mathlib_search_max_queries_per_round,
                    k=phase1_mathlib_search_k,
                )
            compiler_diagnostic_available = bool(
                current_validation.canonical_lean_result is not None
            )
            previous_status = [{
                "symbol": symbol,
                "diagnosticDisappeared": symbol not in eligible_symbols,
            } for symbol in previous_search_symbols] if compiler_diagnostic_available else []
            previous_search_symbols = tuple(
                str(item["symbol"]) for item in search_reports
            )
            details = validation_details(current_validation)
            details["phase1MathlibSearch"] = {
                "enabled": phase1_mathlib_search_enabled,
                "eligibleSymbols": list(eligible_symbols),
                "queries": list(search_reports),
                "previousDiagnosticStatus": previous_status,
                "notRunReason": (
                    "disabled" if not phase1_mathlib_search_enabled else
                    "notEligible" if not eligible_symbols else ""
                ),
            }
            if tracer:
                tracer.emit(TraceEvent(
                    kind="phase1MathlibSearch", thm_name=thm_name,
                    turn=round_index,
                    args={"round": round_index, **details["phase1MathlibSearch"]},
                    ok=not any(item.get("error") for item in search_reports),
                ))

        semantic_invoked = bool(
            current_validation and current_validation.semantic_audit_invoked
        )
        semantic_audit_ordinal: int | None = None
        if semantic_invoked:
            semantic_audit_ordinal = semantic_stage
            semantic_audit_count += 1
            active_repair_codes = _active_repair_codes(current_semantic)
            semantic_anchor_blueprint = current_blueprint
            semantic_anchor_validation = current_validation
            semantic_anchor_errors = current_semantic
            semantic_anchor_warnings = current_warnings
            semantic_anchor_round = round_index
            latest_structural_blueprint = ""
            latest_structural_errors = ()
            latest_structural_warnings = ()
            latest_structural_round = None
            latest_structural_search = ()
            attempt_role = "semanticAudited"
        else:
            if current_blueprint is not None:
                latest_structural_blueprint = current_blueprint.lean_file
            else:
                latest_structural_blueprint = ""
            latest_structural_errors = current_deterministic
            latest_structural_warnings = current_warnings
            latest_structural_round = round_index
            latest_structural_search = search_reports
            attempt_role = "initial" if round_index == 1 else "structuralRetry"

        details.update({
            "semanticAuditCount": semantic_audit_count,
            "generationMaxSemanticAudits": max_semantic_audits,
            "semanticStage": semantic_stage,
            "semanticAuditOrdinal": semantic_audit_ordinal,
            "semanticFeedbackInputRound": input_anchor_round,
            "semanticFeedbackSourceRound": (
                semantic_anchor_round if semantic_invoked else input_anchor_round
            ),
            "semanticFeedbackRetained": bool(
                not semantic_invoked and input_anchor_round is not None
            ),
            "structuralInputRound": input_structural_round,
            "activeRepairCodes": list(active_repair_codes),
            "contextFallbackApplied": context_fallback_applied,
        })
        round_row = GenerationRound(
            round_index=round_index, candidate_hash=candidate_hash,
            input_tokens=input_tokens, max_completion_tokens=completion_budget,
            deterministic_errors=current_deterministic,
            semantic_errors=current_semantic, warnings=current_warnings,
            validation=details, semantic_stage=semantic_stage,
            semantic_audit_ordinal=semantic_audit_ordinal,
            semantic_audit_invoked=semantic_invoked,
            semantic_anchor_round=input_anchor_round,
            structural_input_round=input_structural_round,
            attempt_role=attempt_role,
            submitted_candidate_hash=submitted_hash,
            canonical_candidate_hash=canonical_hash,
            context_fallback_applied=context_fallback_applied,
            builder_message_content=builder_message_content,
            finish_reason=finish_reason,
        )
        rounds.append(round_row)
        if semantic_invoked:
            semantic_anchor_details = dict(details)

        for kind, inventory, ok in (
            ("phase1DeterministicValidation", current_deterministic, not current_deterministic),
            (
                "phase1SemanticAudit" if semantic_invoked else "phase1SemanticAuditSkipped",
                current_semantic,
                not current_semantic if semantic_invoked else True,
            ),
            ("phase1WarningInventory", current_warnings, not current_warnings),
        ):
            if tracer:
                tracer.emit(TraceEvent(
                    kind=kind, thm_name=thm_name, turn=round_index,
                    args={
                        "round": round_index, "semanticStage": semantic_stage,
                        "semanticAuditOrdinal": semantic_audit_ordinal,
                        "semanticAuditCount": semantic_audit_count,
                        "invoked": semantic_invoked, "count": len(inventory),
                        "issues": list(inventory),
                        "semanticAuditMode": semantic_audit_mode,
                        "actualRequestCount": (
                            current_validation.semantic_request_count
                            if current_validation else 0
                        ),
                        "cacheHits": dict(
                            current_validation.semantic_cache_hits or {}
                        ) if current_validation else {},
                        "outputBudget": (
                            current_validation.semantic_output_budget
                            if current_validation else None
                        ),
                        "semanticFeedbackSourceRound": details.get(
                            "semanticFeedbackSourceRound"
                        ),
                        "semanticFeedbackRetained": details.get(
                            "semanticFeedbackRetained", False
                        ),
                        "activeRepairCodes": details.get("activeRepairCodes", []),
                    }, ok=ok,
                ))
        if tracer:
            tracer.emit(TraceEvent(
                kind="phase1RoundAssessment", thm_name=thm_name, turn=round_index,
                args={
                    "round": round_index, "semanticStage": semantic_stage,
                    "semanticAuditOrdinal": semantic_audit_ordinal,
                    "semanticAuditCount": semantic_audit_count,
                    "generationMaxSemanticAudits": max_semantic_audits,
                    "attemptRole": attempt_role,
                    "deterministicErrorCount": len(current_deterministic),
                    "semanticErrorCount": len(current_semantic),
                    "warningCount": len(current_warnings),
                    "mechanicalStageReached": (
                        current_validation.mechanical_stage_reached
                        if current_validation else "submission"
                    ),
                    "mechanicalFailureStage": (
                        current_validation.mechanical_failure_stage
                        if current_validation else "submission"
                    ),
                    "semanticEligible": bool(
                        current_validation
                        and current_validation.mechanical_failure_stage is None
                    ),
                    "semanticAuditMode": semantic_audit_mode,
                    "semanticAuditInvoked": semantic_invoked,
                    "semanticRequestCount": (
                        current_validation.semantic_request_count
                        if current_validation else 0
                    ),
                    "semanticAnchorRound": input_anchor_round,
                    "resultingSemanticAnchorRound": semantic_anchor_round,
                    "structuralInputRound": input_structural_round,
                    "contextFallbackApplied": context_fallback_applied,
                    "semanticFeedbackSourceRound": details.get(
                        "semanticFeedbackSourceRound"
                    ),
                    "semanticFeedbackRetained": details.get(
                        "semanticFeedbackRetained", False
                    ),
                    "activeRepairCodes": details.get("activeRepairCodes", []),
                },
                ok=not current_deterministic and not current_semantic,
            ))
            tracer.emit(TraceEvent(
                kind="phase1GenerationEnd", thm_name=thm_name,
                turn=round_index, span_id=span_id,
                args={
                    "round": round_index, "candidateHash": candidate_hash,
                    "submittedCandidateHash": submitted_hash,
                    "canonicalCandidateHash": canonical_hash,
                    "semanticAuditCount": semantic_audit_count,
                },
                ok=bool(candidate_hash),
                duration_ms=(time.monotonic_ns() - started_ns) / 1_000_000,
            ))

        if semantic_invoked and not current_semantic and current_blueprint is not None:
            classification = "strictAccepted"
            final_details = _accepted_validation_details(
                details, rounds, classification=classification,
                deterministic_errors=current_deterministic,
                semantic_errors=current_semantic, warnings=current_warnings,
            )
            final_details.update({
                "staticGateMode": "shadow", "semanticStaticGate": False,
                "semanticAuditMode": semantic_audit_mode,
                "semanticAuditCount": semantic_audit_count,
                "generationMaxSemanticAudits": max_semantic_audits,
            })
            current_blueprint.generation_validation = final_details
            current_blueprint.generation_history = [item.to_dict() for item in rounds]
            current_blueprint.candidate_history = list(candidates)
            current_blueprint.candidate_labels = list(labels)
            current_blueprint.semantic_gate_results.append({
                "stage": "generation", "passed": True,
                "classification": classification,
                "issues": [
                    {**item, "severity": "warning"} for item in current_warnings
                ],
            })
            if tracer:
                tracer.emit(TraceEvent(
                    kind="phase1FinalClassification", thm_name=thm_name,
                    turn=round_index,
                    args={
                        "classification": classification,
                        "warningCount": len(current_warnings),
                        "semanticAuditCount": semantic_audit_count,
                    }, ok=True,
                ))
            return current_blueprint

    if semantic_anchor_blueprint is None or semantic_anchor_validation is None:
        raise RuntimeError("semantic audit budget exhausted without an audited candidate")
    final_details = dict(semantic_anchor_details)
    audit = dict(final_details.get("semanticAudit") or {})
    audit["classification"] = "semanticRejected"
    final_details.update({
        "semanticAudit": audit,
        "classification": "semanticRejected",
        "terminalCategory": "semanticRejected",
        "terminalMechanicalStage": None,
        "staticGateMode": "shadow", "semanticStaticGate": False,
        "semanticAuditMode": semantic_audit_mode,
        "semanticAuditCount": semantic_audit_count,
        "generationMaxSemanticAudits": max_semantic_audits,
        "finalErrorKinds": ["semantic"],
        "generationRounds": [item.to_dict() for item in rounds],
        "finalDeterministicErrors": [],
        "finalSemanticErrors": list(semantic_anchor_errors),
        "finalWarnings": list(semantic_anchor_warnings),
    })
    if tracer:
        tracer.emit(TraceEvent(
            kind="phase1FinalClassification", thm_name=thm_name,
            turn=generation_attempt_index,
            args={
                "classification": "semanticRejected",
                "finalErrorKinds": ["semantic"],
                "warningCount": len(semantic_anchor_warnings),
                "semanticAuditCount": semantic_audit_count,
            }, ok=False,
        ))
    raise BlueprintGenerationError(
        "Phase 1 exhausted its semantic-audit budget.",
        last_candidate=semantic_anchor_blueprint.lean_file,
        diagnostics=[_feedback(
            (), semantic_anchor_errors, semantic_anchor_warnings,
            blueprint=semantic_anchor_blueprint,
        )],
        attempt=generation_attempt_index,
        failure_stage="phase1Semantic",
        candidate_history=candidates, candidate_labels=labels,
        validation_details=final_details,
        generation_history=[item.to_dict() for item in rounds],
    )


__all__ = [
    "generation_request_budget",
    "generation_round_classification",
    "generate_blueprint",
]

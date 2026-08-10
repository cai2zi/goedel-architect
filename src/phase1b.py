"""Phase 1B Blueprint repair strategies.

This module owns the editable-DAG repair loop.  ``blueprint.py`` remains
responsible for Phase 1A generation and the shared Blueprint data model.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import difflib
import hashlib
import json
import re
import time
import uuid
from typing import Any, Sequence

from blueprint import (
    Blueprint,
    BlueprintGenerationError,
    _apply_node_edits,
    _emit_lean_check_result,
    _emit_llm_response,
    _emit_pending_summary,
    _emit_semantic_check,
    _emit_usage,
    _enabled_semantic_issues,
    _lean_source_contexts,
    _node_hash,
    _parse_blueprint,
    _strip_pending_helper,
    _validate_node_edit,
    phase2_contract_errors,
    phase2_standalone_contract_report,
)
from kimina_lean_compiler import CompilerResult, KiminaInfrastructureError, KiminaLeanCompiler
from llm_client import chat_completion_with_retry
from semantic_audit import (
    FormalDecompilerResult,
    SemanticAuditFormatError,
    StrictComparatorResult,
    build_formal_view,
    comparator_defects,
    run_formal_decompiler,
    run_strict_comparator,
    semantic_audit_cache_key,
    strict_comparator_messages,
)
from semantic_fidelity import SemanticIssue, validate_blueprint_fidelity
from tracer import TraceEvent


REPAIR_STRATEGIES = {"progressController", "planDirect", "directEdit"}


def _phase1_semantic_issues(
    blueprint: Blueprint,
    semantic_manifest,
    *,
    claimed_answer: str,
    semantic_fidelity_enabled: bool,
    semantic_require_step_ids: bool,
    semantic_static_gate: bool,
    allow_pending_claims: bool,
) -> list[SemanticIssue]:
    if not semantic_fidelity_enabled:
        pending: list[SemanticIssue] = []
        for node in blueprint.nodes:
            if "PendingBlueprintClaim" not in node.lean_declaration:
                continue
            exact = re.search(
                rf':\s*PendingBlueprintClaim\s+"{re.escape(node.name)}"\s*:=',
                node.lean_declaration,
            )
            malformed = node.kind == "definition" or not exact
            pending.append(SemanticIssue(
                "malformedPendingClaim" if malformed else "unresolvedPendingClaim",
                "PendingBlueprintClaim is malformed." if malformed else
                "The Phase-1A placeholder has not been replaced.",
                node_name=node.name,
                step_id=node.source_step_id,
            ))
        return [] if allow_pending_claims else pending
    issues = validate_blueprint_fidelity(
        blueprint,
        semantic_manifest,
        claimed_answer=claimed_answer,
        require_step_bindings=semantic_require_step_ids,
        allow_pending_claims=allow_pending_claims,
    )
    return _enabled_semantic_issues(
        issues,
        require_step_ids=semantic_require_step_ids,
        static_gate=semantic_static_gate,
    )

EDIT_SUBGRAPH_TOOL = {
    "type": "function",
    "function": {
        "name": "editBlueprintSubgraph",
        "description": (
            "Atomically add, replace, or delete one or more Blueprint declarations. "
            "Return only declarations that really change."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array", "minItems": 1, "maxItems": 8,
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["add", "replace", "delete"]},
                            "node_name": {"type": "string"},
                            "expected_node_hash": {"type": "string"},
                            "replacement": {"type": "string"},
                        },
                        "required": ["action", "node_name", "expected_node_hash", "replacement"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["edits"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class Phase1BPlan:
    target_obligations: tuple[str, ...]
    edit_nodes: tuple[str, ...]
    new_nodes: tuple[str, ...]
    text: str
    raw_content: str = ""
    attempts: tuple[dict[str, Any], ...] = ()

    def stable_hash(self) -> str:
        payload = {
            "target_obligations": list(self.target_obligations),
            "edit_nodes": list(self.edit_nodes),
            "new_nodes": list(self.new_nodes),
            "text": self.text,
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()


class Phase1BPlanFormatError(ValueError):
    def __init__(self, reason: str, *, attempts: Sequence[dict[str, Any]] = ()) -> None:
        super().__init__(reason)
        self.reason = reason
        self.attempts = tuple(attempts)


@dataclass(frozen=True)
class ProgressDecision:
    decision: str
    reason: str
    raw_content: str = ""
    attempts: tuple[dict[str, Any], ...] = ()


class ProgressDecisionFormatError(ValueError):
    def __init__(self, reason: str, *, attempts: Sequence[dict[str, Any]] = ()) -> None:
        super().__init__(reason)
        self.reason = reason
        self.attempts = tuple(attempts)


@dataclass
class Phase1BValidation:
    lean_result: CompilerResult
    semantic_issues: list[SemanticIssue]
    source_contexts: list[dict[str, Any]]
    structural_errors: list[str]
    standalone_report: Any
    pending_node_names: tuple[str, ...]
    formal_decompiler_result: FormalDecompilerResult | None = None
    strict_comparator_result: StrictComparatorResult | None = None
    open_semantic_obligations: tuple[dict[str, Any], ...] = ()
    semantic_obligation_ledger: tuple[dict[str, Any], ...] = ()
    semantic_audit_required: bool = False

    @property
    def base_passed(self) -> bool:
        return (
            self.lean_result.success
            and not any(issue.severity == "error" for issue in self.semantic_issues)
            and not self.structural_errors
            and not self.standalone_report.issues
            and self.standalone_report.skipped_pending_node_count == 0
            and not self.pending_node_names
        )

    @property
    def passed(self) -> bool:
        return self.base_passed and (
            not self.semantic_audit_required
            or (
                self.strict_comparator_result is not None
                and self.strict_comparator_result.passed
                and not self.open_semantic_obligations
            )
        )


PLANNER_PROMPT = r"""
Create one short repair plan for the supplied Lean Blueprint. Preserve the COT
even if its mathematics is wrong. Focus on formal object binding, exact clauses,
relation direction, and dependency paths. Do not emit Lean or call tools.

Return exactly these headings and no Markdown fence:

TARGET_OBLIGATIONS:
- <exact supplied obligation ID, or none when no IDs exist>

EDIT_NODES:
- <existing declaration name>
- +<new declaration name>

PLAN:
<concrete natural-language plan within the supplied character limit>

Select at most the supplied node limit. Existing names must come from the
inventory. Prefix planned new declarations with `+`. A later Editor may modify
any subset of these nodes and may add at most two other helper declarations.
"""


EDITOR_PROMPT = r"""
Edit a Lean Blueprint by calling `editBlueprintSubgraph` exactly once. You are
translating the supplied COT faithfully, not repairing its mathematics.

Each edit is add, replace, or delete. Use the current hash for replace/delete
and an empty hash for add. A replacement contains exactly one complete
`@[blueprint]` declaration. Never delete or rename the root or change it from a
theorem. Every lemma/theorem proof body must be exactly
`:= by sorry_using [...]`; do not prove nodes with tactics. Dependencies must
name Blueprint nodes. Do not resend unchanged declarations.

When a Plan is supplied, you may edit any subset of its nodes and add at most
two additional helper nodes, but may not modify an unplanned existing node.
Comments do not constitute a semantic repair. Rebuild shared formal objects,
relations, Step claims, and root dependencies in the declaration types/bodies.
"""


CONTROLLER_PROMPT = r"""
Judge whether the candidate is a useful implementation of the fixed repair
Plan. COMMIT means it is worth using as the next repair-turn baseline; it need
not already pass Lean or all final checks. RETRY_EDIT means the candidate does
not genuinely implement the Plan and the Editor should retry from the original
turn baseline. Do not judge whether the source COT is mathematically correct.

Prefer RETRY_EDIT for comment-only changes, reflexive claims, answer-only
witnesses, disconnected nodes, wrong target objects, or diagnostics caused by
an incomplete mechanical implementation that can be fixed under the same Plan.
Allow COMMIT for a real reusable object/relation/dependency improvement even if
some explicit Lean, binder, standalone, or semantic diagnostics remain.

Return exactly:

DECISION: COMMIT | RETRY_EDIT
REASON:
<specific reason; keep it concise>
"""


def parse_phase1b_plan(
    content: str,
    *,
    known_obligation_ids: Sequence[str],
    known_node_names: Sequence[str],
    max_nodes: int,
    max_chars: int,
) -> Phase1BPlan:
    match = re.fullmatch(
        r"\s*TARGET_OBLIGATIONS:\s*\n(?P<obligations>.*?)\n\s*"
        r"EDIT_NODES:\s*\n(?P<nodes>.*?)\n\s*PLAN:\s*\n(?P<plan>.*?)\s*",
        content,
        re.DOTALL,
    )
    if match is None:
        raise Phase1BPlanFormatError(
            "missing or reordered TARGET_OBLIGATIONS/EDIT_NODES/PLAN headings"
        )

    def bullets(block: str, label: str) -> tuple[str, ...]:
        values: list[str] = []
        for raw in block.splitlines():
            line = raw.strip()
            if not line:
                continue
            if not line.startswith("-") or not line[1:].strip():
                raise Phase1BPlanFormatError(f"{label} entries must be non-empty bullets")
            values.append(line[1:].strip())
        return tuple(values)

    obligations = bullets(match.group("obligations"), "TARGET_OBLIGATIONS")
    raw_nodes = bullets(match.group("nodes"), "EDIT_NODES")
    text = match.group("plan").strip()
    if not text:
        raise Phase1BPlanFormatError("PLAN must be non-empty")
    if len(text) > max_chars:
        raise Phase1BPlanFormatError(f"PLAN exceeds {max_chars} characters")
    if not raw_nodes or len(raw_nodes) > max_nodes:
        raise Phase1BPlanFormatError(f"EDIT_NODES must contain 1..{max_nodes} nodes")
    if len(set(raw_nodes)) != len(raw_nodes):
        raise Phase1BPlanFormatError("EDIT_NODES contains duplicates")

    known_nodes = set(known_node_names)
    existing: list[str] = []
    new: list[str] = []
    for raw in raw_nodes:
        is_new = raw.startswith("+")
        name = raw[1:] if is_new else raw
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_']*", name):
            raise Phase1BPlanFormatError(f"invalid Blueprint node name: {raw}")
        if is_new:
            if name in known_nodes:
                raise Phase1BPlanFormatError(f"new node already exists: {name}")
            new.append(name)
        else:
            if name not in known_nodes:
                raise Phase1BPlanFormatError(
                    f"unknown existing node `{name}`; use `+{name}` for a new node"
                )
            existing.append(name)

    known_obligations = set(known_obligation_ids)
    if known_obligations:
        if not obligations or "none" in obligations:
            raise Phase1BPlanFormatError("select at least one open obligation")
        unknown = [item for item in obligations if item not in known_obligations]
        if unknown:
            raise Phase1BPlanFormatError("unknown obligation IDs: " + ", ".join(unknown))
    elif obligations != ("none",):
        raise Phase1BPlanFormatError("use only `none` when no obligations are open")
    if len(set(obligations)) != len(obligations):
        raise Phase1BPlanFormatError("TARGET_OBLIGATIONS contains duplicates")
    return Phase1BPlan(
        obligations, tuple(existing), tuple(new), text, raw_content=content,
    )


def parse_progress_decision(content: str) -> ProgressDecision:
    match = re.fullmatch(
        r"\s*DECISION:\s*(COMMIT|RETRY_EDIT)\s*\n\s*REASON:\s*(?P<reason>.*?)\s*",
        content,
        re.DOTALL,
    )
    if match is None:
        raise ProgressDecisionFormatError("expected DECISION and REASON headings")
    reason = match.group("reason").strip()
    if not reason:
        raise ProgressDecisionFormatError("REASON must be non-empty")
    # The API request bounds the response to the configured token budget.  Do
    # not impose a second, character-based limit on a valid control decision.
    return ProgressDecision(match.group(1), reason, raw_content=content)


def validation_details(validation: Phase1BValidation) -> dict[str, Any]:
    semantic_errors = [
        issue.to_dict() for issue in validation.semantic_issues if issue.severity == "error"
    ]
    semantic_warnings = [
        issue.to_dict() for issue in validation.semantic_issues if issue.severity == "warning"
    ]
    standalone = validation.standalone_report
    audit = None
    if validation.formal_decompiler_result and validation.strict_comparator_result:
        audit = {
            "formalDecompiler": validation.formal_decompiler_result.to_dict(),
            "strictComparator": validation.strict_comparator_result.to_dict(),
            "obligations": list(validation.semantic_obligation_ledger),
            "openObligations": list(validation.open_semantic_obligations),
            "classification": "strictAccepted" if validation.passed else "semanticRejected",
        }
    return {
        "passed": validation.passed,
        "wholeFileLeanSuccess": validation.lean_result.success,
        "leanErrors": list(validation.lean_result.diagnostics),
        "semanticErrors": semantic_errors,
        "semanticWarnings": semantic_warnings,
        "phase2StructuralErrors": list(validation.structural_errors),
        "phase2StandaloneErrors": [issue.to_dict() for issue in standalone.issues],
        "phase2StandaloneSummary": {
            "checkedNodeCount": standalone.checked_node_count,
            "cachedNodeCount": standalone.cached_node_count,
            "skippedPendingNodeCount": standalone.skipped_pending_node_count,
            "failedNodeCount": len(standalone.issues),
            "notRunReason": standalone.not_run_reason,
            "durationMs": standalone.duration_ms,
        },
        "pendingNodes": list(validation.pending_node_names),
        "pendingNodeCount": len(validation.pending_node_names),
        "semanticAuditRequired": validation.semantic_audit_required,
        "semanticAudit": audit,
    }


def _emit_validation(
    tracer, *, thm_name: str, round_index: int, attempt: int,
    validation: Phase1BValidation, event_kind: str = "phase1BSoftDiagnostics",
) -> None:
    if tracer is None:
        return
    details = validation_details(validation)
    tracer.emit(TraceEvent(
        kind=event_kind,
        thm_name=thm_name,
        turn=round_index,
        args={
            "round": round_index,
            "attempt": attempt,
            "wholeFileLeanSuccess": details["wholeFileLeanSuccess"],
            "leanErrorCount": len(details["leanErrors"]),
            "semanticErrorCount": len(details["semanticErrors"]),
            "semanticWarningCount": len(details["semanticWarnings"]),
            "phase2StructuralErrorCount": len(details["phase2StructuralErrors"]),
            "phase2StandaloneErrorCount": len(details["phase2StandaloneErrors"]),
            "pendingNodeCount": details["pendingNodeCount"],
            "semanticAuditPassed": (
                details["semanticAudit"]["strictComparator"]["passed"]
                if details["semanticAudit"] else None
            ),
            "openSemanticObligationCount": len(
                details["semanticAudit"]["openObligations"]
                if details["semanticAudit"] else ()
            ),
            **details["phase2StandaloneSummary"],
        },
        ok=validation.passed,
    ))


def _skipped_standalone(tracer, *, thm_name: str, round_index: int, reason: str):
    from blueprint import Phase2StandaloneReport, _emit_standalone_report

    report = Phase2StandaloneReport((), 0, 0, 0, 0.0, reason)
    _emit_standalone_report(
        tracer, thm_name=thm_name, round_index=round_index, report=report,
    )
    return report


def validate_candidate(
    blueprint: Blueprint,
    *,
    compiler: KiminaLeanCompiler,
    semantic_manifest,
    claimed_answer: str,
    semantic_fidelity_enabled: bool,
    semantic_require_step_ids: bool,
    semantic_static_gate: bool,
    standalone_concurrency: int,
    standalone_cache: dict[str, CompilerResult],
    tracer,
    thm_name: str,
    round_index: int,
    attempt: int,
    skip_pending: bool,
    semantic_audit_required: bool,
) -> Phase1BValidation:
    lean_result = compiler.check_blueprint(blueprint.lean_file, blueprint.target_theorem)
    if lean_result.failure_kind == "infra":
        raise KiminaInfrastructureError(
            "\n".join(lean_result.diagnostics) or lean_result.raw_output[-2000:]
        )
    semantic_issues = _phase1_semantic_issues(
        blueprint,
        semantic_manifest,
        claimed_answer=claimed_answer,
        semantic_fidelity_enabled=semantic_fidelity_enabled,
        semantic_require_step_ids=semantic_require_step_ids,
        semantic_static_gate=semantic_static_gate,
        allow_pending_claims=False,
    )
    structural_errors = phase2_contract_errors(blueprint)
    source_contexts = _lean_source_contexts(lean_result, blueprint, semantic_manifest)
    _emit_semantic_check(
        tracer, thm_name=thm_name, phase="phase1B", attempt=attempt,
        turn=round_index, issues=semantic_issues,
    )
    _emit_lean_check_result(
        tracer, thm_name=thm_name, phase="phase1B", attempt=attempt,
        target=blueprint.target_theorem, result=lean_result,
        source_contexts=source_contexts,
    )
    if not lean_result.success:
        standalone = _skipped_standalone(
            tracer, thm_name=thm_name, round_index=round_index,
            reason="wholeFileCompileFailed",
        )
    elif structural_errors:
        standalone = _skipped_standalone(
            tracer, thm_name=thm_name, round_index=round_index,
            reason="phase2StructuralContractFailed",
        )
    else:
        standalone = phase2_standalone_contract_report(
            blueprint,
            compiler,
            concurrency=standalone_concurrency,
            skip_pending=skip_pending,
            cache=standalone_cache,
            tracer=tracer,
            thm_name=thm_name,
            round_index=round_index,
        )
    result = Phase1BValidation(
        lean_result=lean_result,
        semantic_issues=semantic_issues,
        source_contexts=source_contexts,
        structural_errors=structural_errors,
        standalone_report=standalone,
        pending_node_names=tuple(
            node.name for node in blueprint.nodes
            if "PendingBlueprintClaim" in node.lean_declaration
        ),
        semantic_audit_required=semantic_audit_required,
    )
    _emit_validation(
        tracer, thm_name=thm_name, round_index=round_index,
        attempt=attempt, validation=result,
    )
    return result


BLOCKING_BINDING_CODES = {
    "emptyCotManifest", "missingRoot", "missingStepMapping",
    "multipleStepMappings", "malformedStepMapping", "unknownStepMapping",
    "rootNotFinalStep", "stepMappingAbsent",
}


def semantic_audit_eligible(validation: Phase1BValidation) -> bool:
    """Audit syntactically valid candidates even when Lean currently fails."""
    return (
        not validation.pending_node_names
        and not any(
            issue.severity == "error" and issue.code in BLOCKING_BINDING_CODES
            for issue in validation.semantic_issues
        )
    )


def _obligation_id(defect: dict[str, Any], semantic_manifest=None) -> str:
    step_id = str(defect.get("step_id") or "")
    source_hash = ""
    if semantic_manifest is not None and step_id:
        step = getattr(semantic_manifest, "by_id", {}).get(step_id.split(".", 1)[0])
        source_hash = str(getattr(step, "source_sha256", "") or "") if step else ""
    if not source_hash:
        source_hash = hashlib.sha256(
            str(defect.get("requirement") or "").encode()
        ).hexdigest()
    payload = {
        "category": str(defect.get("category") or "semanticDefect"),
        "step_id": step_id,
        "node_names": sorted(str(name) for name in defect.get("node_names") or ()),
        "source_clause_hash": source_hash,
    }
    return "semantic:" + payload["category"] + ":" + hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:20]


def _open_obligations(ledger: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    return tuple(
        dict(ledger[key]) for key in sorted(ledger)
        if ledger[key].get("status") == "open"
    )


def _update_obligations(
    ledger: dict[str, dict[str, Any]], *, view, decompiler, comparator,
    semantic_manifest, round_index: int,
) -> tuple[dict[str, Any], ...]:
    defects = comparator_defects(comparator)
    grouped: dict[str, dict[str, Any]] = {}
    for defect in defects:
        obligation_id = _obligation_id(defect, semantic_manifest)
        row = grouped.get(obligation_id)
        if row is None:
            grouped[obligation_id] = {**defect, "obligation_id": obligation_id}
        else:
            row["node_names"] = sorted(
                set(row.get("node_names") or ())
                | set(defect.get("node_names") or ())
            )
    current_ids = set(grouped)
    for obligation_id, row in ledger.items():
        if row.get("status") == "open" and obligation_id not in current_ids:
            row.update({"status": "resolved", "lastCheckedRound": round_index})
    for obligation_id, defect in grouped.items():
        old = ledger.get(obligation_id)
        ledger[obligation_id] = {
            "obligation_id": obligation_id,
            "category": str(defect.get("category") or "semanticDefect"),
            "step_id": str(defect.get("step_id") or ""),
            "node_names": list(defect.get("node_names") or ()),
            "requirement": str(defect.get("requirement") or ""),
            "reason": str(defect.get("reason") or ""),
            "firstRound": int(old.get("firstRound", round_index)) if old else round_index,
            "lastCheckedRound": round_index,
            "status": "open",
        }
    return _open_obligations(ledger)


def with_semantic_audit(
    validation: Phase1BValidation,
    blueprint: Blueprint,
    *, client, model: str, informal_statement: str, claimed_answer: str,
    semantic_manifest, formal_decompiler_max_tokens: int,
    strict_comparator_max_tokens: int, format_max_attempts: int,
    decompiler_cache: dict[str, FormalDecompilerResult],
    comparator_cache: dict[str, StrictComparatorResult],
    obligation_ledger: dict[str, dict[str, Any]], tracer, thm_name: str,
    round_index: int,
) -> Phase1BValidation:
    view = build_formal_view(blueprint)
    decompiler = decompiler_cache.get(view.sha256)
    if decompiler is None:
        decompiler = run_formal_decompiler(
            client, model, view=view, max_tokens=formal_decompiler_max_tokens,
            max_attempts=format_max_attempts, tracer=tracer,
            thm_name=thm_name, round_index=round_index,
        )
        decompiler_cache[view.sha256] = decompiler
    messages = strict_comparator_messages(
        informal_statement, claimed_answer, semantic_manifest, view,
        decompiler, _open_obligations(obligation_ledger),
    )
    cache_key = semantic_audit_cache_key(model, messages)
    comparator = comparator_cache.get(cache_key)
    if comparator is None:
        comparator = run_strict_comparator(
            client, model, informal_statement=informal_statement,
            claimed_answer=claimed_answer, manifest=semantic_manifest,
            view=view, decompiler=decompiler,
            open_obligations=_open_obligations(obligation_ledger),
            max_tokens=strict_comparator_max_tokens,
            max_attempts=format_max_attempts, tracer=tracer,
            thm_name=thm_name, round_index=round_index,
        )
        comparator_cache[cache_key] = comparator
    open_after = _update_obligations(
        obligation_ledger, view=view, decompiler=decompiler,
        comparator=comparator, semantic_manifest=semantic_manifest,
        round_index=round_index,
    )
    return Phase1BValidation(
        lean_result=validation.lean_result,
        semantic_issues=validation.semantic_issues,
        source_contexts=validation.source_contexts,
        structural_errors=validation.structural_errors,
        standalone_report=validation.standalone_report,
        pending_node_names=validation.pending_node_names,
        formal_decompiler_result=decompiler,
        strict_comparator_result=comparator,
        open_semantic_obligations=open_after,
        semantic_obligation_ledger=tuple(dict(x) for x in obligation_ledger.values()),
        semantic_audit_required=True,
    )


def _semantic_snapshot(validation: Phase1BValidation) -> dict[str, Any]:
    comparator = validation.strict_comparator_result
    root = comparator.root if comparator is not None else {}
    required_disconnections = 0
    if comparator is not None:
        required_disconnections = sum(
            not bool(item.get("justified_side_branch"))
            for item in comparator.unreachable_steps
        )
    return {
        "openObligationIds": sorted(
            str(item.get("obligation_id") or "")
            for item in validation.open_semantic_obligations
        ),
        "rootTargetObject": root.get("target_object_preserved"),
        "rootAnswerGrounding": root.get("answer_grounded"),
        "requiredPathDisconnections": required_disconnections,
    }


def _semantic_delta(
    before: dict[str, Any], after: dict[str, Any], *,
    target_obligations: Sequence[str], changed_nodes: Sequence[str],
    no_op_nodes: Sequence[str],
) -> dict[str, Any]:
    old = set(before.get("openObligationIds") or ())
    new = set(after.get("openObligationIds") or ())
    targets = set(target_obligations) - {"none"}
    return {
        "resolvedTargetObligations": sorted(targets - new),
        "stillOpenTargetObligations": sorted(targets & new),
        "newObligations": sorted(new - old),
        "reopenedObligations": [],
        "rootTargetObject": {
            "before": before.get("rootTargetObject"),
            "after": after.get("rootTargetObject"),
        },
        "rootAnswerGrounding": {
            "before": before.get("rootAnswerGrounding"),
            "after": after.get("rootAnswerGrounding"),
        },
        "requiredPathDisconnections": {
            "before": before.get("requiredPathDisconnections"),
            "after": after.get("requiredPathDisconnections"),
        },
        "changedFormalNodes": list(changed_nodes),
        "noOpNodes": list(no_op_nodes),
    }


def _compact_validation(validation: Phase1BValidation) -> dict[str, Any]:
    details = validation_details(validation)
    comparator = validation.strict_comparator_result
    defects: list[dict[str, Any]] = []
    if comparator is not None:
        for step in comparator.steps:
            for key in ("missing_clauses", "weakened_clauses", "unbound_objects", "wrong_relations"):
                if step.get(key):
                    defects.append({
                        "step_id": step.get("step_id"), "category": key,
                        "items": step.get(key),
                    })
        defects.extend({"category": "dependency", **item} for item in comparator.dependency_issues)
    return {
        "whole_file_lean_success": details["wholeFileLeanSuccess"],
        "lean_errors": details["leanErrors"][:12],
        "semantic_errors": details["semanticErrors"],
        "structural_errors": details["phase2StructuralErrors"],
        "standalone_errors": details["phase2StandaloneErrors"],
        "pending_nodes": details["pendingNodes"],
        "open_obligations": list(validation.open_semantic_obligations),
        "strict_defects": defects,
        "semantic_snapshot": _semantic_snapshot(validation),
        "strict_passed": validation.passed,
    }


def _relevant_view(blueprint: Blueprint, node_names: Sequence[str]) -> dict[str, Any]:
    view = build_formal_view(blueprint)
    wanted = set(node_names) | {view.root_name}
    return {
        "root_name": view.root_name,
        "root_closure": list(view.root_closure),
        "nodes": [asdict(node) for node in view.nodes if node.node_name in wanted],
    }


def _formal_diff(before: Blueprint, after: Blueprint, *, limit: int = 16000) -> str:
    diff = "\n".join(difflib.unified_diff(
        before.lean_file.splitlines(), after.lean_file.splitlines(),
        fromfile="committed", tofile="candidate", lineterm="",
    ))
    return diff[:limit]


def _node_inventory(blueprint: Blueprint) -> list[dict[str, Any]]:
    return [{
        "node_name": node.name,
        "hash": _node_hash(node),
        "kind": node.kind,
        "step_id": node.source_step_id,
        "dependencies": list(node.dependencies),
    } for node in blueprint.nodes]


def _planner_messages(
    *, informal_statement: str, claimed_answer: str, prompt_proof: str,
    blueprint: Blueprint, validation: Phase1BValidation,
    previous_turn: dict[str, Any] | None, max_nodes: int,
) -> list[dict[str, str]]:
    payload = {
        "problem": informal_statement,
        "claimed_answer": claimed_answer,
        "cot_steps": prompt_proof,
        "root": blueprint.target_theorem,
        "nodes": _node_inventory(blueprint),
        "formal_view": build_formal_view(blueprint).to_dict(),
        "diagnostics": _compact_validation(validation),
        "maximum_edit_nodes": max_nodes,
    }
    if previous_turn:
        payload["previous_turn"] = previous_turn
    return [
        {"role": "system", "content": PLANNER_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def run_planner(
    client, model: str, messages: list[dict[str, str]], *,
    known_obligation_ids: Sequence[str], known_node_names: Sequence[str],
    max_tokens: int, max_attempts: int, max_nodes: int, max_chars: int,
    round_index: int, tracer, thm_name: str,
) -> Phase1BPlan:
    base = list(messages)
    attempts: list[dict[str, Any]] = []
    span_id = uuid.uuid4().hex
    started = time.monotonic_ns()
    if tracer:
        tracer.emit(TraceEvent(
            kind="phase1BPlanStart", thm_name=thm_name, turn=round_index,
            span_id=span_id, args={"round": round_index},
        ))
    for attempt in range(1, max_attempts + 1):
        response = chat_completion_with_retry(
            client, tracer=tracer, thm_name=thm_name, phase="phase1BPlanner",
            model_id=model, operation="phase1b_light_plan", model=model,
            messages=messages, temperature=0, max_completion_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        _emit_usage(tracer, thm_name, "phase1BPlanner", model, response)
        _emit_llm_response(
            tracer, thm_name=thm_name, phase="phase1BPlanner", model=model,
            response=response, attempt=attempt, turn=round_index,
        )
        choice = response.choices[0]
        content = str(choice.message.content or "")
        record = {"attempt": attempt, "rawContent": content,
                  "finishReason": getattr(choice, "finish_reason", None)}
        try:
            if str(record["finishReason"] or "").lower() == "length":
                raise Phase1BPlanFormatError("planner response truncated")
            plan = parse_phase1b_plan(
                content, known_obligation_ids=known_obligation_ids,
                known_node_names=known_node_names, max_nodes=max_nodes,
                max_chars=max_chars,
            )
        except Phase1BPlanFormatError as exc:
            record["error"] = exc.reason
            attempts.append(record)
            if attempt == max_attempts:
                if tracer:
                    tracer.emit(TraceEvent(
                        kind="phase1BPlanEnd", thm_name=thm_name,
                        turn=round_index, span_id=span_id, ok=False,
                        result=exc.reason,
                        duration_ms=(time.monotonic_ns() - started) / 1_000_000,
                    ))
                raise Phase1BPlanFormatError(exc.reason, attempts=attempts) from exc
            messages = [*base, {"role": "user", "content": (
                f"Rejected: {exc.reason}. Return only the three required headings. "
                f"Known obligations: {list(known_obligation_ids) or ['none']}. "
                f"Known nodes: {list(known_node_names)}. Prefix new nodes with +."
            )}]
            continue
        attempts.append(record)
        plan = Phase1BPlan(
            plan.target_obligations, plan.edit_nodes, plan.new_nodes,
            plan.text, raw_content=content, attempts=tuple(attempts),
        )
        if tracer:
            tracer.emit(TraceEvent(
                kind="phase1BPlanResult", thm_name=thm_name, turn=round_index,
                span_id=span_id, result=content, ok=True,
                args={"round": round_index,
                      "targetObligations": list(plan.target_obligations),
                      "editNodes": list(plan.edit_nodes),
                      "newNodes": list(plan.new_nodes), "plan": plan.text,
                      "planHash": plan.stable_hash()},
            ))
            tracer.emit(TraceEvent(
                kind="phase1BPlanEnd", thm_name=thm_name, turn=round_index,
                span_id=span_id, ok=True,
                duration_ms=(time.monotonic_ns() - started) / 1_000_000,
            ))
        return plan
    raise AssertionError("unreachable")


def _call_editor(
    client, model: str, *, blueprint: Blueprint, plan: Phase1BPlan | None,
    informal_statement: str, prompt_proof: str, claimed_answer: str,
    validation: Phase1BValidation, retry_feedback: dict[str, Any] | None,
    round_index: int, attempt: int, tracer, thm_name: str,
):
    payload: dict[str, Any] = {
        "problem": informal_statement,
        "claimed_answer": claimed_answer,
        "cot_steps": prompt_proof,
        "plan": ({
            "target_obligations": list(plan.target_obligations),
            "edit_nodes": list(plan.edit_nodes),
            "new_nodes": list(plan.new_nodes),
            "text": plan.text,
        } if plan else None),
        "node_inventory": _node_inventory(blueprint),
        "current_blueprint": blueprint.lean_file,
        "diagnostics": _compact_validation(validation),
        "retry_feedback": retry_feedback,
    }
    messages = [
        {"role": "system", "content": EDITOR_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    response = chat_completion_with_retry(
        client, tracer=tracer, thm_name=thm_name, phase="phase1B",
        model_id=model, operation="phase1b_v9_subgraph_edit", model=model,
        messages=messages, tools=[EDIT_SUBGRAPH_TOOL], tool_choice="required",
        temperature=0, max_completion_tokens=16384,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    _emit_usage(tracer, thm_name, "phase1B", model, response)
    _emit_llm_response(
        tracer, thm_name=thm_name, phase="phase1B", model=model,
        response=response, attempt=attempt, turn=round_index,
    )
    return response


def _apply_hard_checked_edit(
    committed: Blueprint,
    response,
    *, plan: Phase1BPlan | None, max_edits: int,
) -> tuple[Blueprint | None, dict[str, Any]]:
    calls = list(response.choices[0].message.tool_calls or [])
    result: dict[str, Any] = {
        "actualNodes": [], "effectiveNodes": [], "noOpNodes": [],
        "hardErrors": [], "edits": [],
    }
    if len(calls) != 1:
        result["hardErrors"].append(
            "missingSubgraphCall" if not calls else "multipleSubgraphCalls"
        )
        return None, result
    call = calls[0]
    if str(call.function.name) != "editBlueprintSubgraph":
        result["hardErrors"].append("notAllowed")
        return None, result
    try:
        args = json.loads(call.function.arguments or "{}")
        raw_edits = args.get("edits") if isinstance(args, dict) else None
        if not isinstance(raw_edits, list) or not 1 <= len(raw_edits) <= max_edits:
            raise ValueError(f"edits must contain 1..{max_edits} items")
        actual_names = [
            str(item.get("node_name") or "")
            for item in raw_edits if isinstance(item, dict)
        ]
        if len(actual_names) != len(raw_edits):
            raise ValueError("every edit must be an object")
        if len(set(actual_names)) != len(actual_names):
            raise ValueError("duplicateNodeInSubgraph")
        if plan is not None:
            current_names = set(committed.nodes_by_name())
            allowed_existing = set(plan.edit_nodes)
            unplanned_existing = sorted(
                name for name in actual_names
                if name in current_names and name not in allowed_existing
            )
            if unplanned_existing:
                raise ValueError(
                    "unplannedExistingNodeEdit:" + ",".join(unplanned_existing)
                )
            planned_new = set(plan.new_nodes)
            extra_new = {
                name for name in actual_names
                if name not in current_names and name not in planned_new
            }
            if len(extra_new) > 2:
                raise ValueError("tooManyUnplannedHelperNodes")
        edits = []
        for item in raw_edits:
            allowed = {"action", "node_name", "expected_node_hash", "replacement"}
            if set(item) != allowed:
                raise ValueError("invalidEditSchema")
            action = str(item.get("action") or "")
            node_name = str(item.get("node_name") or "")
            result["actualNodes"].append(node_name)
            edit, reason = _validate_node_edit(
                committed, action=action, node_name=node_name,
                expected_hash=str(item.get("expected_node_hash") or ""),
                replacement=str(item.get("replacement") or ""),
            )
            if reason == "identicalReplacement":
                result["noOpNodes"].append(node_name)
                continue
            if reason or edit is None:
                raise ValueError(reason or "invalidEdit")
            edits.append(edit)
            result["effectiveNodes"].append(node_name)
            result["edits"].append({"action": action, "nodeName": node_name})
        if not edits:
            raise ValueError("noEffectiveEdit")
        candidate = _apply_node_edits(committed, edits)
        result["candidateHash"] = hashlib.sha256(candidate.lean_file.encode()).hexdigest()
        return candidate, result
    except (json.JSONDecodeError, ValueError) as exc:
        result["hardErrors"].append(str(exc))
        return None, result


def _controller_messages(
    *, informal_statement: str, claimed_answer: str, prompt_proof: str,
    plan: Phase1BPlan, committed: Blueprint, candidate: Blueprint,
    baseline_validation: Phase1BValidation,
    candidate_validation: Phase1BValidation,
    hard_result: dict[str, Any], semantic_delta: dict[str, Any],
    previous_reason: str,
) -> list[dict[str, str]]:
    relevant_nodes = list(dict.fromkeys([
        *plan.edit_nodes, *plan.new_nodes, *hard_result.get("actualNodes", ()),
        committed.target_theorem,
    ]))
    payload = {
        "problem": informal_statement,
        "claimed_answer": claimed_answer,
        "cot_steps": prompt_proof,
        "plan": {
            "target_obligations": list(plan.target_obligations),
            "edit_nodes": list(plan.edit_nodes), "new_nodes": list(plan.new_nodes),
            "text": plan.text,
        },
        "baseline_formal_view": _relevant_view(committed, relevant_nodes),
        "candidate_formal_view": _relevant_view(candidate, relevant_nodes),
        "formal_diff": _formal_diff(committed, candidate),
        "actual_edits": hard_result.get("edits", []),
        "no_op_nodes": hard_result.get("noOpNodes", []),
        "baseline_diagnostics": _compact_validation(baseline_validation),
        "candidate_diagnostics": _compact_validation(candidate_validation),
        "semantic_delta": semantic_delta,
        "previous_controller_reason": previous_reason,
    }
    return [
        {"role": "system", "content": CONTROLLER_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def run_progress_controller(
    client, model: str, messages: list[dict[str, str]], *,
    max_tokens: int, max_attempts: int, round_index: int, attempt_index: int,
    tracer, thm_name: str,
) -> ProgressDecision:
    base = list(messages)
    attempts: list[dict[str, Any]] = []
    span_id = uuid.uuid4().hex
    started = time.monotonic_ns()
    if tracer:
        tracer.emit(TraceEvent(
            kind="phase1BProgressControllerStart", thm_name=thm_name,
            turn=round_index, span_id=span_id,
            args={"round": round_index, "attempt": attempt_index},
        ))
    for format_attempt in range(1, max_attempts + 1):
        response = chat_completion_with_retry(
            client, tracer=tracer, thm_name=thm_name,
            phase="phase1BProgressController", model_id=model,
            operation="phase1b_progress_controller", model=model,
            messages=messages, temperature=0, max_completion_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        _emit_usage(tracer, thm_name, "phase1BProgressController", model, response)
        _emit_llm_response(
            tracer, thm_name=thm_name, phase="phase1BProgressController",
            model=model, response=response, attempt=format_attempt,
            turn=round_index,
        )
        choice = response.choices[0]
        content = str(choice.message.content or "")
        record = {"attempt": format_attempt, "rawContent": content,
                  "finishReason": getattr(choice, "finish_reason", None)}
        try:
            if str(record["finishReason"] or "").lower() == "length":
                raise ProgressDecisionFormatError("controller response truncated")
            decision = parse_progress_decision(content)
        except ProgressDecisionFormatError as exc:
            record["error"] = exc.reason
            attempts.append(record)
            if format_attempt == max_attempts:
                if tracer:
                    tracer.emit(TraceEvent(
                        kind="phase1BProgressControllerEnd", thm_name=thm_name,
                        turn=round_index, span_id=span_id, ok=False,
                        result=exc.reason,
                        duration_ms=(time.monotonic_ns() - started) / 1_000_000,
                    ))
                raise ProgressDecisionFormatError(exc.reason, attempts=attempts) from exc
            messages = [*base, {"role": "user", "content": (
                f"Rejected: {exc.reason}. Return only DECISION and REASON; "
                "DECISION must be COMMIT or RETRY_EDIT."
            )}]
            continue
        attempts.append(record)
        decision = ProgressDecision(
            decision.decision, decision.reason, raw_content=content,
            attempts=tuple(attempts),
        )
        if tracer:
            tracer.emit(TraceEvent(
                kind="phase1BProgressControllerResult", thm_name=thm_name,
                turn=round_index, span_id=span_id, result=content, ok=True,
                args={"round": round_index, "attempt": attempt_index,
                      "decision": decision.decision, "reason": decision.reason},
            ))
            tracer.emit(TraceEvent(
                kind="phase1BProgressControllerEnd", thm_name=thm_name,
                turn=round_index, span_id=span_id, ok=True,
                duration_ms=(time.monotonic_ns() - started) / 1_000_000,
            ))
        return decision
    raise AssertionError("unreachable")


def _feedback(validation: Phase1BValidation) -> str:
    return json.dumps(_compact_validation(validation), ensure_ascii=False)


def _audit_candidate(
    validation: Phase1BValidation,
    blueprint: Blueprint,
    *, ledger: dict[str, dict[str, Any]], client, model: str,
    informal_statement: str, claimed_answer: str, semantic_manifest,
    formal_decompiler_max_tokens: int, strict_comparator_max_tokens: int,
    semantic_format_max_attempts: int,
    decompiler_cache: dict[str, FormalDecompilerResult],
    comparator_cache: dict[str, StrictComparatorResult], tracer,
    thm_name: str, round_index: int,
) -> Phase1BValidation:
    if not validation.semantic_audit_required:
        return validation
    if not semantic_audit_eligible(validation):
        validation.open_semantic_obligations = _open_obligations(ledger)
        validation.semantic_obligation_ledger = tuple(dict(x) for x in ledger.values())
        return validation
    return with_semantic_audit(
        validation, blueprint, client=client, model=model,
        informal_statement=informal_statement, claimed_answer=claimed_answer,
        semantic_manifest=semantic_manifest,
        formal_decompiler_max_tokens=formal_decompiler_max_tokens,
        strict_comparator_max_tokens=strict_comparator_max_tokens,
        format_max_attempts=semantic_format_max_attempts,
        decompiler_cache=decompiler_cache, comparator_cache=comparator_cache,
        obligation_ledger=ledger, tracer=tracer, thm_name=thm_name,
        round_index=round_index,
    )


def run_phase1b_patch_session(
    client,
    model: str,
    blueprint: Blueprint,
    *,
    compiler: KiminaLeanCompiler,
    informal_statement: str,
    prompt_proof: str,
    claimed_answer: str,
    semantic_manifest,
    semantic_fidelity_enabled: bool,
    semantic_require_step_ids: bool,
    semantic_static_gate: bool,
    max_rounds: int,
    phase2_contract_check_concurrency: int,
    tracer,
    thm_name: str,
    candidate_history: list[str],
    candidate_labels: list[str],
    semantic_audit_enabled: bool = False,
    formal_decompiler_max_tokens: int = 4096,
    strict_comparator_max_tokens: int = 4096,
    semantic_format_max_attempts: int = 2,
    repair_strategy: str = "directEdit",
    editor_attempts_per_turn: int = 3,
    plan_max_tokens: int = 768,
    plan_format_attempts: int = 2,
    plan_max_chars: int = 600,
    progress_controller_max_tokens: int = 2048,
    progress_controller_format_attempts: int = 2,
    subgraph_max_edits: int = 8,
) -> Blueprint:
    if repair_strategy not in REPAIR_STRATEGIES:
        raise ValueError(
            "phase1b_repair_strategy must be one of: "
            + ", ".join(sorted(REPAIR_STRATEGIES))
        )
    committed = blueprint
    initial_pending_names = tuple(
        node.name for node in committed.nodes
        if "PendingBlueprintClaim" in node.lean_declaration
    )
    previous_pending_names = initial_pending_names
    standalone_cache: dict[str, CompilerResult] = {}
    decompiler_cache: dict[str, FormalDecompilerResult] = {}
    comparator_cache: dict[str, StrictComparatorResult] = {}
    ledger: dict[str, dict[str, Any]] = {}
    rejected_candidate_hashes: set[str] = set()
    edit_history: list[dict[str, Any]] = []
    previous_turn: dict[str, Any] | None = None

    validation = validate_candidate(
        committed, compiler=compiler, semantic_manifest=semantic_manifest,
        claimed_answer=claimed_answer,
        semantic_fidelity_enabled=semantic_fidelity_enabled,
        semantic_require_step_ids=semantic_require_step_ids,
        semantic_static_gate=semantic_static_gate,
        standalone_concurrency=phase2_contract_check_concurrency,
        standalone_cache=standalone_cache, tracer=tracer, thm_name=thm_name,
        round_index=0, attempt=0, skip_pending=True,
        semantic_audit_required=semantic_audit_enabled,
    )
    try:
        validation = _audit_candidate(
            validation, committed, ledger=ledger, client=client, model=model,
            informal_statement=informal_statement, claimed_answer=claimed_answer,
            semantic_manifest=semantic_manifest,
            formal_decompiler_max_tokens=formal_decompiler_max_tokens,
            strict_comparator_max_tokens=strict_comparator_max_tokens,
            semantic_format_max_attempts=semantic_format_max_attempts,
            decompiler_cache=decompiler_cache, comparator_cache=comparator_cache,
            tracer=tracer, thm_name=thm_name, round_index=0,
        )
    except SemanticAuditFormatError as exc:
        raise BlueprintGenerationError(
            f"Initial Phase 1B semantic audit response was invalid: {exc.reason}",
            last_candidate=committed.lean_file, diagnostics=[exc.reason],
            failure_stage="phase1BSemanticAuditFormat",
            candidate_history=candidate_history, candidate_labels=candidate_labels,
        ) from exc
    _emit_pending_summary(
        tracer, thm_name=thm_name, phase="phase1B", round_index=0,
        blueprint=committed, initial_names=initial_pending_names,
    )

    for round_index in range(1, max_rounds + 1):
        if validation.passed:
            break
        plan: Phase1BPlan | None = None
        if repair_strategy != "directEdit":
            try:
                plan = run_planner(
                    client, model,
                    _planner_messages(
                        informal_statement=informal_statement,
                        claimed_answer=claimed_answer, prompt_proof=prompt_proof,
                        blueprint=committed, validation=validation,
                        previous_turn=previous_turn, max_nodes=subgraph_max_edits,
                    ),
                    known_obligation_ids=[
                        str(item.get("obligation_id") or "")
                        for item in validation.open_semantic_obligations
                    ],
                    known_node_names=[node.name for node in committed.nodes],
                    max_tokens=plan_max_tokens,
                    max_attempts=plan_format_attempts,
                    max_nodes=subgraph_max_edits, max_chars=plan_max_chars,
                    round_index=round_index, tracer=tracer, thm_name=thm_name,
                )
            except Phase1BPlanFormatError as exc:
                row = {
                    "round": round_index, "strategy": repair_strategy,
                    "plan": "", "planHash": "", "targetObligations": [],
                    "plannedNodes": [], "newNodes": [], "attempts": [],
                    "committed": False, "rollbackReason": "planFormat:" + exc.reason,
                    "committedHash": hashlib.sha256(committed.lean_file.encode()).hexdigest(),
                }
                edit_history.append(row)
                previous_turn = row
                if tracer:
                    tracer.emit(TraceEvent(
                        kind="phase1BTurnRollback", thm_name=thm_name,
                        turn=round_index, args=row, ok=False,
                    ))
                continue

        baseline = committed
        baseline_validation = validation
        semantic_before = _semantic_snapshot(baseline_validation)
        retry_feedback: dict[str, Any] | None = None
        attempt_rows: list[dict[str, Any]] = []
        committed_this_turn = False
        for attempt in range(1, editor_attempts_per_turn + 1):
            span_id = uuid.uuid4().hex
            started = time.monotonic_ns()
            if tracer:
                tracer.emit(TraceEvent(
                    kind="phase1BEditorAttemptStart", thm_name=thm_name,
                    turn=round_index, span_id=span_id,
                    args={"round": round_index, "attempt": attempt,
                          "strategy": repair_strategy},
                ))
            response = _call_editor(
                client, model, blueprint=baseline, plan=plan,
                informal_statement=informal_statement, prompt_proof=prompt_proof,
                claimed_answer=claimed_answer, validation=baseline_validation,
                retry_feedback=retry_feedback, round_index=round_index,
                attempt=attempt, tracer=tracer, thm_name=thm_name,
            )
            if tracer:
                tracer.emit(TraceEvent(
                    kind="phase1BEditorAttemptResult", thm_name=thm_name,
                    turn=round_index, span_id=span_id,
                    args={"round": round_index, "attempt": attempt},
                    ok=True,
                ))
            candidate, hard = _apply_hard_checked_edit(
                baseline, response, plan=plan, max_edits=subgraph_max_edits,
            )
            candidate_hash = str(hard.get("candidateHash") or "")
            if candidate_hash and candidate_hash in rejected_candidate_hashes:
                hard["hardErrors"].append("repeatedRejectedCandidateHash")
                candidate = None
            hard_ok = candidate is not None and not hard["hardErrors"]
            if hard["noOpNodes"] and tracer:
                tracer.emit(TraceEvent(
                    kind="phase1BNoOpFiltered", thm_name=thm_name,
                    turn=round_index, span_id=span_id,
                    args={"round": round_index, "attempt": attempt,
                          "noOpNodes": hard["noOpNodes"],
                          "effectiveNodes": hard["effectiveNodes"]},
                    ok=hard_ok,
                ))
            if tracer:
                tracer.emit(TraceEvent(
                    kind="phase1BHardValidation", thm_name=thm_name,
                    turn=round_index, span_id=span_id, args={
                        "round": round_index, "attempt": attempt,
                        "actualNodes": hard["actualNodes"],
                        "effectiveNodes": hard["effectiveNodes"],
                        "noOpNodes": hard["noOpNodes"],
                        "hardErrors": hard["hardErrors"],
                        "candidateHash": candidate_hash,
                    }, ok=hard_ok,
                ))
            attempt_row: dict[str, Any] = {
                "attempt": attempt, "actualNodes": hard["actualNodes"],
                "effectiveNodes": hard["effectiveNodes"],
                "noOpNodes": hard["noOpNodes"], "hardErrors": hard["hardErrors"],
                "candidateHash": candidate_hash, "controllerDecision": None,
                "controllerReason": "", "softDiagnostics": None,
            }
            if not hard_ok:
                retry_feedback = {
                    "kind": "hardValidation", "errors": hard["hardErrors"],
                    "actual_nodes": hard["actualNodes"],
                    "effective_nodes": hard["effectiveNodes"],
                }
                attempt_rows.append(attempt_row)
                if tracer:
                    tracer.emit(TraceEvent(
                        kind="phase1BEditorAttemptEnd", thm_name=thm_name,
                        turn=round_index, span_id=span_id, ok=False,
                        args={"round": round_index, "attempt": attempt,
                              "outcome": "hardRetry"},
                        duration_ms=(time.monotonic_ns() - started) / 1_000_000,
                    ))
                continue

            assert candidate is not None
            candidate_history.append(candidate.lean_file)
            candidate_labels.append(f"phase1b_round_{round_index}_attempt_{attempt}")
            candidate_validation = validate_candidate(
                candidate, compiler=compiler, semantic_manifest=semantic_manifest,
                claimed_answer=claimed_answer,
                semantic_fidelity_enabled=semantic_fidelity_enabled,
                semantic_require_step_ids=semantic_require_step_ids,
                semantic_static_gate=semantic_static_gate,
                standalone_concurrency=phase2_contract_check_concurrency,
                standalone_cache=standalone_cache, tracer=tracer,
                thm_name=thm_name, round_index=round_index, attempt=attempt,
                skip_pending=True, semantic_audit_required=semantic_audit_enabled,
            )
            candidate_ledger = copy.deepcopy(ledger)
            try:
                candidate_validation = _audit_candidate(
                    candidate_validation, candidate, ledger=candidate_ledger,
                    client=client, model=model,
                    informal_statement=informal_statement,
                    claimed_answer=claimed_answer,
                    semantic_manifest=semantic_manifest,
                    formal_decompiler_max_tokens=formal_decompiler_max_tokens,
                    strict_comparator_max_tokens=strict_comparator_max_tokens,
                    semantic_format_max_attempts=semantic_format_max_attempts,
                    decompiler_cache=decompiler_cache,
                    comparator_cache=comparator_cache, tracer=tracer,
                    thm_name=thm_name, round_index=round_index,
                )
            except SemanticAuditFormatError as exc:
                raise BlueprintGenerationError(
                    f"Phase 1B semantic audit response was invalid: {exc.reason}",
                    last_candidate=candidate.lean_file, diagnostics=[exc.reason],
                    failure_stage="phase1BSemanticAuditFormat",
                    candidate_history=candidate_history,
                    candidate_labels=candidate_labels,
                    node_edit_rounds=edit_history,
                ) from exc
            _emit_validation(
                tracer, thm_name=thm_name, round_index=round_index,
                attempt=attempt, validation=candidate_validation,
            )
            semantic_after = _semantic_snapshot(candidate_validation)
            delta = _semantic_delta(
                semantic_before, semantic_after,
                target_obligations=(plan.target_obligations if plan else ()),
                changed_nodes=hard["effectiveNodes"],
                no_op_nodes=hard["noOpNodes"],
            )
            attempt_row["softDiagnostics"] = _compact_validation(candidate_validation)
            attempt_row["semanticDelta"] = delta

            decision = "COMMIT"
            reason = "hard-valid candidate committed by simple strategy"
            if repair_strategy == "progressController" and not candidate_validation.passed:
                assert plan is not None
                try:
                    progress = run_progress_controller(
                        client, model,
                        _controller_messages(
                            informal_statement=informal_statement,
                            claimed_answer=claimed_answer, prompt_proof=prompt_proof,
                            plan=plan, committed=baseline, candidate=candidate,
                            baseline_validation=baseline_validation,
                            candidate_validation=candidate_validation,
                            hard_result=hard, semantic_delta=delta,
                            previous_reason=(retry_feedback or {}).get("reason", ""),
                        ),
                        max_tokens=progress_controller_max_tokens,
                        max_attempts=progress_controller_format_attempts,
                        round_index=round_index, attempt_index=attempt,
                        tracer=tracer, thm_name=thm_name,
                    )
                except ProgressDecisionFormatError as exc:
                    raise BlueprintGenerationError(
                        "Phase 1B Progress Controller response was invalid: "
                        + exc.reason,
                        last_candidate=baseline.lean_file,
                        diagnostics=[exc.reason],
                        failure_stage="phase1BProgressControllerFormat",
                        candidate_history=candidate_history,
                        candidate_labels=candidate_labels,
                        node_edit_rounds=edit_history,
                    ) from exc
                decision, reason = progress.decision, progress.reason
            elif candidate_validation.passed:
                reason = "candidate satisfies the final strict gate"
            attempt_row["controllerDecision"] = decision
            attempt_row["controllerReason"] = reason
            attempt_rows.append(attempt_row)
            if decision == "COMMIT":
                committed = candidate
                validation = candidate_validation
                ledger = candidate_ledger
                previous_turn = None
                rejected_candidate_hashes.clear()
                committed_this_turn = True
                if tracer:
                    tracer.emit(TraceEvent(
                        kind="phase1BTurnCommit", thm_name=thm_name,
                        turn=round_index, args={
                            "round": round_index, "attempt": attempt,
                            "strategy": repair_strategy,
                            "committedHashBefore": hashlib.sha256(
                                baseline.lean_file.encode()
                            ).hexdigest(),
                            "committedHashAfter": candidate_hash,
                            "semanticDelta": delta, "reason": reason,
                        }, ok=True,
                    ))
                if tracer:
                    tracer.emit(TraceEvent(
                        kind="phase1BEditorAttemptEnd", thm_name=thm_name,
                        turn=round_index, span_id=span_id, ok=True,
                        args={"round": round_index, "attempt": attempt,
                              "outcome": "commit"},
                        duration_ms=(time.monotonic_ns() - started) / 1_000_000,
                    ))
                break

            rejected_candidate_hashes.add(candidate_hash)
            retry_feedback = {
                "kind": "progressController", "reason": reason,
                "candidate_formal_diff": _formal_diff(baseline, candidate, limit=10000),
                "actual_nodes": hard["actualNodes"],
                "effective_nodes": hard["effectiveNodes"],
                "no_op_nodes": hard["noOpNodes"],
                "soft_diagnostics": _compact_validation(candidate_validation),
                "semantic_delta": delta,
            }
            if tracer:
                tracer.emit(TraceEvent(
                    kind="phase1BEditorAttemptEnd", thm_name=thm_name,
                    turn=round_index, span_id=span_id, ok=False,
                    args={"round": round_index, "attempt": attempt,
                          "outcome": "controllerRetry", "reason": reason},
                    duration_ms=(time.monotonic_ns() - started) / 1_000_000,
                ))

        turn_row = {
            "round": round_index, "strategy": repair_strategy,
            "plan": plan.text if plan else "",
            "planHash": plan.stable_hash() if plan else "",
            "targetObligations": list(plan.target_obligations) if plan else [],
            "plannedNodes": list(plan.edit_nodes) if plan else [],
            "newNodes": list(plan.new_nodes) if plan else [],
            "attempts": attempt_rows, "committed": committed_this_turn,
            "rollbackReason": "" if committed_this_turn else "editorAttemptsExhausted",
            "committedHash": hashlib.sha256(committed.lean_file.encode()).hexdigest(),
        }
        edit_history.append(turn_row)
        if not committed_this_turn:
            previous_turn = turn_row
            if tracer:
                tracer.emit(TraceEvent(
                    kind="phase1BTurnRollback", thm_name=thm_name,
                    turn=round_index, args=turn_row, ok=False,
                ))
        _emit_pending_summary(
            tracer, thm_name=thm_name, phase="phase1B",
            round_index=round_index, blueprint=committed,
            initial_names=initial_pending_names,
            previous_names=previous_pending_names,
        )
        previous_pending_names = tuple(
            node.name for node in committed.nodes
            if "PendingBlueprintClaim" in node.lean_declaration
        )

    if not validation.passed:
        raise BlueprintGenerationError(
            f"Phase 1B failed after {max_rounds} repair turns.",
            last_candidate=committed.lean_file,
            diagnostics=[_feedback(validation)], attempt=max_rounds,
            failure_stage="phase1BFailed", candidate_history=candidate_history,
            candidate_labels=candidate_labels,
            validation_details=validation_details(validation),
            node_edit_rounds=edit_history,
        )

    final_code = _strip_pending_helper(committed.lean_file)
    final = _parse_blueprint(final_code, committed.target_theorem)
    final_validation = validate_candidate(
        final, compiler=compiler, semantic_manifest=semantic_manifest,
        claimed_answer=claimed_answer,
        semantic_fidelity_enabled=semantic_fidelity_enabled,
        semantic_require_step_ids=semantic_require_step_ids,
        semantic_static_gate=semantic_static_gate,
        standalone_concurrency=phase2_contract_check_concurrency,
        standalone_cache=standalone_cache, tracer=tracer, thm_name=thm_name,
        round_index=max_rounds + 1, attempt=1, skip_pending=False,
        semantic_audit_required=semantic_audit_enabled,
    )
    try:
        final_validation = _audit_candidate(
            final_validation, final, ledger=ledger, client=client, model=model,
            informal_statement=informal_statement, claimed_answer=claimed_answer,
            semantic_manifest=semantic_manifest,
            formal_decompiler_max_tokens=formal_decompiler_max_tokens,
            strict_comparator_max_tokens=strict_comparator_max_tokens,
            semantic_format_max_attempts=semantic_format_max_attempts,
            decompiler_cache=decompiler_cache, comparator_cache=comparator_cache,
            tracer=tracer, thm_name=thm_name, round_index=max_rounds + 1,
        )
    except SemanticAuditFormatError as exc:
        raise BlueprintGenerationError(
            f"Final Phase 1B semantic audit response was invalid: {exc.reason}",
            last_candidate=final_code, diagnostics=[exc.reason],
            failure_stage="phase1BSemanticAuditFormat",
            candidate_history=candidate_history,
            candidate_labels=candidate_labels,
            node_edit_rounds=edit_history,
        ) from exc
    if not final_validation.passed:
        raise BlueprintGenerationError(
            "Final Phase 1B strict validation failed.",
            last_candidate=final_code, diagnostics=[_feedback(final_validation)],
            failure_stage="phase1BFinalValidation",
            candidate_history=candidate_history, candidate_labels=candidate_labels,
            validation_details=validation_details(final_validation),
            node_edit_rounds=edit_history,
        )
    candidate_history.append(final_code)
    candidate_labels.append("phase1b_final")
    final.semantic_gate_results.append({
        "stage": "phase1B_final", "passed": True,
        "issues": [issue.to_dict() for issue in final_validation.semantic_issues],
        "warning_count": sum(
            issue.severity == "warning" for issue in final_validation.semantic_issues
        ),
        "phase2_structural_error_count": 0,
        "phase2_standalone_error_count": 0,
    })
    final.phase1b_validation = validation_details(final_validation)
    final.phase1b_edit_history = edit_history
    return final

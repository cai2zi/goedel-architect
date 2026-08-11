"""Phase 1B Blueprint repair strategies.

This module owns the editable-DAG repair loop.  ``blueprint.py`` remains
responsible for Phase 1A generation and the shared Blueprint data model.
"""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
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
from mathlib_retrieval import MathlibRetrieval
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
from semantic_fidelity import effective_blueprint_dependencies
from tracer import TraceEvent


REPAIR_STRATEGIES = {"planDirect", "directEdit"}
OBJECT_OBLIGATION_CATEGORIES = {
    "unbound_objects", "rootTargetObject", "rootAnswerGrounding",
}
CRITICAL_ROOT_REGRESSION_CODES = {
    "vacuousTrueRoot", "vacuousPropRoot", "vacuousTrueShellRoot",
    "reflexiveRoot", "unboundAnswerWitnessRoot", "unconstrainedExistsRoot",
}


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

MATHLIB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "mathlib_search",
        "description": (
            "Search Mathlib once for exact names, type signatures, or existing "
            "formal constructions needed by the planned Blueprint edit."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "target_node_names": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["query", "target_node_names"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["queries"],
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

The later Editor has access to one batched Mathlib search call. When exact
Mathlib names, types, or constructions are uncertain, PLAN may mention the
concept that the Editor should look up. Do not output search queries, tool calls,
or guessed theorem names; semantic repair remains the purpose of this Plan.

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
Translate the supplied COT faithfully, not its mathematical correctness. When
search is available, either call `mathlib_search` once or call
`editBlueprintSubgraph` exactly once. Search only for exact Mathlib names,
types, or existing constructions; it cannot replace object or relation modeling.
After search results are returned, call `editBlueprintSubgraph` exactly once.

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
Mathlib names may be used in formal types and definition bodies, but
`sorry_using [...]` may contain Blueprint node names only.
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


def _semantic_issue_fingerprints(validation: Phase1BValidation) -> set[str]:
    return {
        f"{issue.code}|{issue.node_name}|{issue.step_id}"
        for issue in validation.semantic_issues if issue.severity == "error"
    }


def _standalone_issue_fingerprints(validation: Phase1BValidation) -> set[str]:
    values: set[str] = set()
    for issue in validation.standalone_report.issues:
        row = issue.to_dict() if hasattr(issue, "to_dict") else dict(issue)
        values.add(
            f"{row.get('code', '')}|{row.get('nodeName', '')}|"
            f"{row.get('stepId', '')}"
        )
    return values


def _structural_issue_fingerprints(validation: Phase1BValidation) -> set[str]:
    """Normalize diagnostics so wording and source locations are not debt."""
    values: set[str] = set()
    for raw in validation.structural_errors:
        text = str(raw)
        code = text.split(":", 1)[0].strip()
        node_match = re.search(r"node\s+`([^`]+)`", text)
        step_match = re.search(r"(?:Step|step)\s+`?([A-Za-z0-9_.-]+)`?", text)
        values.add(
            f"{code}|{node_match.group(1) if node_match else ''}|"
            f"{step_match.group(1) if step_match else ''}"
        )
    return values


def _standalone_failed_nodes(validation: Phase1BValidation) -> set[str]:
    nodes: set[str] = set()
    for issue in validation.standalone_report.issues:
        row = issue.to_dict() if hasattr(issue, "to_dict") else dict(issue)
        if row.get("nodeName"):
            nodes.add(str(row["nodeName"]))
    return nodes


def _deterministic_debt(validation: Phase1BValidation) -> dict[str, Any]:
    semantic = _semantic_issue_fingerprints(validation)
    structural = _structural_issue_fingerprints(validation)
    standalone = _standalone_issue_fingerprints(validation)
    pending = set(validation.pending_node_names)
    return {
        "leanSuccess": validation.lean_result.success,
        "semanticErrors": sorted(semantic),
        "structuralErrors": sorted(structural),
        "standaloneErrors": sorted(standalone),
        "pendingNodes": sorted(pending),
        "count": (
            (0 if validation.lean_result.success else 1)
            + len(semantic) + len(structural) + len(standalone) + len(pending)
        ),
    }


def _stable_gate(
    baseline: Phase1BValidation,
    candidate: Phase1BValidation,
    *,
    changed_nodes: Sequence[str],
) -> dict[str, Any]:
    before = _deterministic_debt(baseline)
    after = _deterministic_debt(candidate)
    errors: list[str] = []
    if not candidate.lean_result.success:
        errors.append("wholeFileLeanFailed")
    for label in ("semanticErrors", "structuralErrors", "standaloneErrors", "pendingNodes"):
        added = sorted(set(after[label]) - set(before[label]))
        if added:
            errors.append(f"new{label[0].upper() + label[1:]}:" + ",".join(added))
    pending = set(candidate.pending_node_names)
    changed_concrete = set(changed_nodes) - pending
    changed_standalone_failures = sorted(
        changed_concrete & _standalone_failed_nodes(candidate)
    )
    if changed_standalone_failures:
        errors.append(
            "changedConcreteStandaloneFailed:" + ",".join(changed_standalone_failures)
        )
    return {
        "passed": not errors,
        "errors": errors,
        "baselineDebt": before,
        "candidateDebt": after,
        "deterministicDebtDecreased": int(after["count"]) < int(before["count"]),
    }


def _source_step_hash(semantic_manifest, step_id: str) -> str:
    if not semantic_manifest or not step_id:
        return ""
    base = step_id.split(".", 1)[0]
    step = getattr(semantic_manifest, "by_id", {}).get(base)
    return str(getattr(step, "source_sha256", "") or "") if step else ""


def _normalized_obligation_signature(item: dict[str, Any], semantic_manifest) -> str:
    category = str(item.get("category") or "semanticDefect")
    step_id = str(item.get("step_id") or "").split(".", 1)[0]
    if category in {"rootTargetObject", "rootAnswerGrounding"}:
        return f"{category}|<root>|<global>"
    source_hash = _source_step_hash(semantic_manifest, step_id)
    if not source_hash:
        source_hash = hashlib.sha256(
            str(item.get("requirement") or "").encode()
        ).hexdigest()[:20]
    return f"{category}|{step_id or '<root>'}|{source_hash}"


def _normalized_obligations(
    validation: Phase1BValidation,
    semantic_manifest,
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for item in validation.open_semantic_obligations:
        signature = _normalized_obligation_signature(item, semantic_manifest)
        grouped.setdefault(signature, []).append(str(item.get("obligation_id") or ""))
    return {key: sorted(values) for key, values in sorted(grouped.items())}


def _strict_defect_count(validation: Phase1BValidation) -> int:
    count = int(_deterministic_debt(validation)["count"])
    if validation.formal_decompiler_result is not None:
        count += len(validation.formal_decompiler_result.vacuous_nodes)
    if validation.strict_comparator_result is not None:
        count += len(comparator_defects(validation.strict_comparator_result))
    return count


def _formal_reference_consumers(
    blueprint: Blueprint,
    foundation_nodes: Sequence[str],
    changed_nodes: Sequence[str],
) -> tuple[dict[str, list[str]], set[str]]:
    view = build_formal_view(blueprint)
    declarations = {node.node_name: node.declaration for node in view.nodes}
    node_map = blueprint.nodes_by_name()
    root_closure: set[str] = set()
    stack = [blueprint.target_theorem]
    while stack:
        name = stack.pop()
        if name in root_closure or name not in node_map:
            continue
        root_closure.add(name)
        stack.extend(effective_blueprint_dependencies(node_map[name], node_map))
    changed = set(changed_nodes)
    consumers: dict[str, list[str]] = {}
    for foundation in foundation_nodes:
        matches: list[str] = []
        for name, declaration in declarations.items():
            if name == foundation or name not in root_closure or name not in changed:
                continue
            identifiers = set(re.findall(r"[A-Za-z_][A-Za-z0-9_']*", declaration))
            if foundation in identifiers:
                matches.append(name)
        consumers[foundation] = sorted(matches)
    return consumers, root_closure


def _foundation_nodes(
    baseline: Blueprint,
    candidate: Blueprint,
    hard_result: dict[str, Any],
) -> tuple[list[str], list[str]]:
    actions = {
        str(item.get("nodeName") or ""): str(item.get("action") or "")
        for item in hard_result.get("edits") or ()
    }
    added = sorted(name for name, action in actions.items() if action == "add")
    foundations = set(added)
    for name in hard_result.get("effectiveNodes") or ():
        before = baseline.node_by_name(name)
        after = candidate.node_by_name(name)
        if before is not None and after is not None and after.kind == "definition":
            foundations.add(name)
    return sorted(foundations), added


def _critical_semantic_regressions(
    baseline: Phase1BValidation,
    candidate: Phase1BValidation,
    *,
    root_name: str,
) -> list[str]:
    before = _semantic_snapshot(baseline)
    after = _semantic_snapshot(candidate)
    errors: list[str] = []
    for field in ("rootTargetObject", "rootAnswerGrounding"):
        if before.get(field) is True and after.get(field) is False:
            errors.append(field + "Regressed")
    if int(after.get("requiredPathDisconnections") or 0) > int(
        before.get("requiredPathDisconnections") or 0
    ):
        errors.append("requiredPathDisconnectionsIncreased")
    before_issues = {
        (issue.code, issue.node_name)
        for issue in baseline.semantic_issues if issue.severity == "error"
    }
    for issue in candidate.semantic_issues:
        if (
            issue.severity == "error"
            and issue.node_name == root_name
            and issue.code in CRITICAL_ROOT_REGRESSION_CODES
            and (issue.code, issue.node_name) not in before_issues
        ):
            errors.append("newCriticalRootError:" + issue.code)
    before_vacuous = set(
        baseline.formal_decompiler_result.vacuous_nodes
        if baseline.formal_decompiler_result else ()
    )
    after_vacuous = set(
        candidate.formal_decompiler_result.vacuous_nodes
        if candidate.formal_decompiler_result else ()
    )
    if root_name in after_vacuous - before_vacuous:
        errors.append("rootBecameVacuous")
    return errors


def _commit_assessment(
    baseline_blueprint: Blueprint,
    candidate_blueprint: Blueprint,
    baseline: Phase1BValidation,
    candidate: Phase1BValidation,
    *,
    stable: dict[str, Any],
    hard_result: dict[str, Any],
    plan: Phase1BPlan | None,
    semantic_manifest,
    closure_mode: bool,
    foundation_debt_open: bool,
) -> dict[str, Any]:
    before_snapshot = _semantic_snapshot(baseline)
    after_snapshot = _semantic_snapshot(candidate)
    before_raw = {str(x.get("obligation_id") or "") for x in baseline.open_semantic_obligations}
    after_raw = {str(x.get("obligation_id") or "") for x in candidate.open_semantic_obligations}
    before_normalized = _normalized_obligations(baseline, semantic_manifest)
    after_normalized = _normalized_obligations(candidate, semantic_manifest)
    before_signatures = set(before_normalized)
    after_signatures = set(after_normalized)
    resolved_normalized = sorted(before_signatures - after_signatures)
    new_normalized = sorted(after_signatures - before_signatures)
    target_ids = set(plan.target_obligations if plan else ()) - {"none"}
    resolved_targets = sorted(target_ids - after_raw)
    strict_before = _strict_defect_count(baseline)
    strict_after = _strict_defect_count(candidate)
    root_improved = any(
        before_snapshot.get(field) is False and after_snapshot.get(field) is True
        for field in ("rootTargetObject", "rootAnswerGrounding")
    )
    path_improved = int(after_snapshot.get("requiredPathDisconnections") or 0) < int(
        before_snapshot.get("requiredPathDisconnections") or 0
    )
    progress_flags = {
        "deterministicDebtDecreased": bool(stable["deterministicDebtDecreased"]),
        "targetObligationResolved": bool(resolved_targets),
        "openObligationCountDecreased": len(after_signatures) < len(before_signatures),
        "strictDefectCountDecreased": strict_after < strict_before,
        "rootImproved": root_improved,
        "requiredPathImproved": path_improved,
    }
    ordinary_progress = any(progress_flags.values())
    foundations, added = _foundation_nodes(
        baseline_blueprint, candidate_blueprint, hard_result,
    )
    consumers, root_closure = _formal_reference_consumers(
        candidate_blueprint, foundations, hard_result.get("effectiveNodes") or (),
    )
    vacuous = set(
        candidate.formal_decompiler_result.vacuous_nodes
        if candidate.formal_decompiler_result else ()
    )
    foundation_integrated = bool(foundations) and all(
        consumers.get(name) and name not in vacuous for name in foundations
    )
    foundation_only = bool(foundations) and not ordinary_progress
    critical = _critical_semantic_regressions(
        baseline, candidate, root_name=candidate_blueprint.target_theorem,
    )
    object_before = {
        str(item.get("obligation_id") or "")
        for item in baseline.open_semantic_obligations
        if item.get("category") in OBJECT_OBLIGATION_CATEGORIES
    }
    object_after = {
        str(item.get("obligation_id") or "")
        for item in candidate.open_semantic_obligations
        if item.get("category") in OBJECT_OBLIGATION_CATEGORIES
    }
    resolved_object_obligations = sorted(object_before - object_after)
    closure_auxiliary_allowed = (
        bool(added)
        and bool(resolved_object_obligations)
        and all(consumers.get(name) and name not in vacuous for name in added)
        and not new_normalized
        and (
            len(after_signatures) < len(before_signatures)
            or strict_after < strict_before
        )
    )

    errors = list(critical)
    reason = "semanticProgress"
    if not candidate.semantic_audit_required:
        reason = "semanticAuditDisabled"
    elif closure_mode:
        if foundation_only:
            errors.append("foundationOnlyNotAllowedInClosure")
        if new_normalized:
            errors.append("newObligationsInClosure:" + ",".join(new_normalized))
        if not (
            len(after_signatures) < len(before_signatures)
            or strict_after < strict_before
        ):
            errors.append("closureRequiresStrictProgress")
        if added and not closure_auxiliary_allowed:
            errors.append("closureAuxiliaryNodeNotJustified:" + ",".join(added))
        reason = "closureProgress"
    elif not ordinary_progress:
        if not foundation_only:
            errors.append("noSemanticProgress")
        elif foundation_debt_open:
            errors.append("consecutiveFoundationOnlyCommit")
        elif not foundation_integrated:
            errors.append("foundationNotIntegrated")
        else:
            reason = "integratedFoundation"

    passed = not errors
    return {
        "passed": passed,
        "reason": reason if passed else errors[0],
        "errors": errors,
        "closureMode": closure_mode,
        "baselineDebt": stable["baselineDebt"],
        "candidateDebt": stable["candidateDebt"],
        "strictDefectBefore": strict_before,
        "strictDefectAfter": strict_after,
        "rawOpenBefore": sorted(before_raw),
        "rawOpenAfter": sorted(after_raw),
        "normalizedOpenBefore": before_normalized,
        "normalizedOpenAfter": after_normalized,
        "resolvedNormalizedObligations": resolved_normalized,
        "newNormalizedObligations": new_normalized,
        "resolvedTargetObligations": resolved_targets,
        "progressFlags": progress_flags,
        "criticalRegressions": critical,
        "foundationNodes": foundations,
        "addedNodes": added,
        "foundationConsumers": consumers,
        "rootClosure": sorted(root_closure),
        "foundationOnly": foundation_only,
        "foundationIntegrated": foundation_integrated,
        "foundationDebtBefore": foundation_debt_open,
        "resolvedObjectObligations": resolved_object_obligations,
        "closureAuxiliaryAllowed": closure_auxiliary_allowed,
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
    search_state: dict[str, Any] | None = None,
    search_cache: dict[str, list[dict[str, str]]] | None = None,
    search_max_queries: int = 0,
    search_max_results: int = 5,
    search_policy: str = "leanErrorsOnly",
):
    search_eligibility = _mathlib_search_eligibility(
        validation,
        retry_feedback=retry_feedback,
        blueprint_node_names=tuple(blueprint.nodes_by_name()),
        policy=search_policy,
    )
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
        "mathlib_search": {
            "available": bool(
                search_max_queries > 0
                and search_state is not None
                and not search_state.get("used")
                and search_eligibility["eligible"]
            ),
            "policy": search_policy,
            "eligibility_reasons": search_eligibility["reasons"],
            "maximum_queries": search_max_queries,
            "maximum_results_per_query": search_max_results,
            "results_for_this_turn": (
                list(search_state.get("results") or ()) if search_state else []
            ),
        },
    }

    def request(*, edit_only: bool, operation: str):
        current_payload = dict(payload)
        current_payload["mode"] = "EDIT ONLY" if edit_only else "SEARCH OR EDIT"
        messages = [
            {"role": "system", "content": EDITOR_PROMPT},
            {"role": "user", "content": json.dumps(current_payload, ensure_ascii=False)},
        ]
        tools = [EDIT_SUBGRAPH_TOOL]
        if not edit_only and current_payload["mathlib_search"]["available"]:
            tools.append(MATHLIB_SEARCH_TOOL)
        response = chat_completion_with_retry(
            client, tracer=tracer, thm_name=thm_name, phase="phase1B",
            model_id=model, operation=operation, model=model,
            messages=messages, tools=tools, tool_choice="required",
            temperature=0, max_completion_tokens=16384,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        _emit_usage(tracer, thm_name, "phase1B", model, response)
        _emit_llm_response(
            tracer, thm_name=thm_name, phase="phase1B", model=model,
            response=response, attempt=attempt, turn=round_index,
        )
        return response

    can_search = bool(
        search_max_queries > 0
        and search_state is not None
        and not search_state.get("used")
        and search_eligibility["eligible"]
    )
    if tracer:
        tracer.emit(TraceEvent(
            kind="phase1BMathlibSearchEligibility",
            thm_name=thm_name,
            turn=round_index,
            args={
                "round": round_index,
                "attempt": attempt,
                "policy": search_policy,
                "configured": search_max_queries > 0,
                "eligible": bool(search_eligibility["eligible"]),
                "reasons": list(search_eligibility["reasons"]),
                "alreadyUsed": bool(search_state and search_state.get("used")),
            },
            ok=True,
        ))
    response = request(edit_only=not can_search, operation="phase1b_subgraph_edit")
    if not can_search:
        return response
    calls = list(response.choices[0].message.tool_calls or ())
    search_calls = [
        call for call in calls if str(call.function.name) == "mathlib_search"
    ]
    if not search_calls:
        return response

    assert search_state is not None
    search_state["used"] = True
    search_rows = _execute_phase1b_mathlib_search(
        search_calls,
        blueprint=blueprint,
        plan=plan,
        max_queries=search_max_queries,
        max_results=search_max_results,
        cache=search_cache if search_cache is not None else {},
        tracer=tracer,
        thm_name=thm_name,
        round_index=round_index,
        attempt=attempt,
    )
    search_state["results"] = search_rows
    payload["mathlib_search"] = {
        "available": False,
        "maximum_queries": search_max_queries,
        "maximum_results_per_query": search_max_results,
        "results_for_this_turn": search_rows,
    }
    if any(str(call.function.name) == "editBlueprintSubgraph" for call in calls):
        if tracer:
            tracer.emit(TraceEvent(
                kind="phase1BEditDeferredUntilAfterSearch",
                thm_name=thm_name,
                turn=round_index,
                args={"round": round_index, "attempt": attempt},
            ))
    return request(edit_only=True, operation="phase1b_subgraph_edit_after_search")


_SEARCHABLE_LEAN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("unknownIdentifier", re.compile(r"Unknown identifier", re.I)),
    ("unknownConstant", re.compile(r"Unknown constant", re.I)),
    ("invalidFieldNotation", re.compile(r"Invalid field notation", re.I)),
    ("synthesisFailure", re.compile(r"failed to synthesize", re.I)),
)


def _mathlib_search_eligibility(
    validation: Phase1BValidation,
    *,
    retry_feedback: dict[str, Any] | None,
    blueprint_node_names: Sequence[str],
    policy: str,
) -> dict[str, Any]:
    """Return high-confidence Mathlib/API lookup reasons for one Editor attempt.

    Unknown identifiers that are already Blueprint nodes are dependency/modeling
    errors, not Mathlib lookup problems, and therefore do not unlock search.
    """
    if policy != "leanErrorsOnly":
        raise ValueError("phase1b_mathlib_search_policy must be leanErrorsOnly")
    node_names = set(blueprint_node_names)
    reasons: list[str] = []

    def add_diagnostic(origin: str, diagnostic: str) -> None:
        text = str(diagnostic or "")
        for code, pattern in _SEARCHABLE_LEAN_PATTERNS:
            if not pattern.search(text):
                continue
            identifiers = set(re.findall(
                r"Unknown (?:identifier|constant) [`']([^`']+)[`']", text,
                flags=re.I,
            ))
            if code in {"unknownIdentifier", "unknownConstant"} and identifiers:
                external = sorted(identifiers - node_names)
                if not external:
                    continue
                reasons.append(f"{origin}:{code}:" + ",".join(external))
            else:
                reasons.append(f"{origin}:{code}")

    for diagnostic in validation.lean_result.diagnostics:
        add_diagnostic("committedLean", diagnostic)
    for issue in validation.standalone_report.issues:
        external = sorted(set(issue.identifiers) - node_names)
        if issue.error_kind in {"unknownIdentifier", "unknownConstant"}:
            if external or not issue.identifiers:
                suffix = ":" + ",".join(external) if external else ""
                reasons.append(f"committedStandalone:{issue.error_kind}{suffix}")
        elif issue.error_kind == "synthesisFailure":
            reasons.append("committedStandalone:synthesisFailure")
        elif issue.error_kind == "typeMismatch" and re.search(
            r"Invalid field notation", issue.diagnostic, re.I
        ):
            reasons.append("committedStandalone:invalidFieldNotation")

    diagnostics = (
        retry_feedback.get("deterministic_diagnostics")
        if isinstance(retry_feedback, dict) else None
    )
    if isinstance(diagnostics, dict):
        for diagnostic in diagnostics.get("lean_errors") or ():
            add_diagnostic("retryLean", str(diagnostic))
        for issue in diagnostics.get("standalone_errors") or ():
            if not isinstance(issue, dict):
                continue
            kind = str(issue.get("errorKind") or "")
            identifiers = set(str(x) for x in issue.get("identifiers") or ())
            external = sorted(identifiers - node_names)
            diagnostic = str(issue.get("diagnostic") or "")
            if kind in {"unknownIdentifier", "unknownConstant"}:
                if external or not identifiers:
                    suffix = ":" + ",".join(external) if external else ""
                    reasons.append(f"retryStandalone:{kind}{suffix}")
            elif kind == "synthesisFailure":
                reasons.append("retryStandalone:synthesisFailure")
            elif kind == "typeMismatch" and re.search(
                r"Invalid field notation", diagnostic, re.I
            ):
                reasons.append("retryStandalone:invalidFieldNotation")

    unique = tuple(dict.fromkeys(reasons))
    return {"eligible": bool(unique), "reasons": list(unique)}


def _execute_phase1b_mathlib_search(
    calls: Sequence[Any],
    *,
    blueprint: Blueprint,
    plan: Phase1BPlan | None,
    max_queries: int,
    max_results: int,
    cache: dict[str, list[dict[str, str]]],
    tracer,
    thm_name: str,
    round_index: int,
    attempt: int,
) -> list[dict[str, Any]]:
    """Execute one batched search call and return bounded current-turn context."""
    span_id = uuid.uuid4().hex
    started_ns = time.monotonic_ns()
    protocol_errors: list[str] = []
    queries: list[dict[str, Any]] = []
    if len(calls) != 1:
        protocol_errors.append("multipleMathlibSearchCalls")
    call = calls[0]
    try:
        args = json.loads(call.function.arguments or "{}")
        raw_queries = args.get("queries") if isinstance(args, dict) else None
        if set(args) != {"queries"} or not isinstance(raw_queries, list):
            raise ValueError("invalidSearchSchema")
        if not 1 <= len(raw_queries) <= max_queries:
            raise ValueError(f"search queries must contain 1..{max_queries} items")
        known_nodes = set(blueprint.nodes_by_name()) | set(plan.new_nodes if plan else ())
        seen: set[str] = set()
        for item in raw_queries:
            if not isinstance(item, dict) or set(item) != {"query", "target_node_names"}:
                raise ValueError("invalidSearchQuerySchema")
            query = str(item.get("query") or "").strip()
            targets = item.get("target_node_names")
            if not query or not isinstance(targets, list) or not all(
                isinstance(name, str) and name for name in targets
            ):
                raise ValueError("invalidSearchQuery")
            unknown = sorted(set(targets) - known_nodes)
            if unknown:
                raise ValueError("unknownSearchTarget:" + ",".join(unknown))
            key = " ".join(query.casefold().split())
            if key in seen:
                continue
            seen.add(key)
            queries.append({"query": query, "target_node_names": list(dict.fromkeys(targets)), "cache_key": key})
        if not queries:
            raise ValueError("noUniqueSearchQueries")
    except (json.JSONDecodeError, ValueError) as exc:
        protocol_errors.append(str(exc))
        queries = []

    if tracer:
        tracer.emit(TraceEvent(
            kind="phase1BMathlibSearchStart",
            thm_name=thm_name,
            turn=round_index,
            span_id=span_id,
            args={
                "round": round_index,
                "attempt": attempt,
                "queryCount": len(queries),
                "protocolErrors": protocol_errors,
            },
        ))

    def fetch(item: dict[str, Any]) -> tuple[list[dict[str, str]], bool]:
        key = f"{item['cache_key']}|{max_results}"
        if key in cache:
            return cache[key], True
        try:
            with MathlibRetrieval() as retrieval:
                hits = retrieval.search(item["query"], max_results)
            seen_names: set[str] = set()
            rows: list[dict[str, str]] = []
            for hit in hits:
                if not hit.name or hit.name in seen_names:
                    continue
                seen_names.add(hit.name)
                rows.append({
                    "name": hit.name,
                    "type_signature": str(hit.type_sig or "")[:800],
                    "docstring": str(hit.docstring or "")[:240],
                })
                if len(rows) >= max_results:
                    break
        except Exception as exc:  # Search failure is diagnostic, not infrastructure.
            rows = [{"name": "", "type_signature": "", "docstring": f"Search failed: {exc}"}]
        cache[key] = rows
        return rows, False

    fetched: list[tuple[list[dict[str, str]], bool]] = []
    if queries:
        with ThreadPoolExecutor(max_workers=len(queries)) as executor:
            fetched = list(executor.map(fetch, queries))

    rendered: list[dict[str, Any]] = []
    character_budget = 4096
    used_chars = 0
    for item, (results, cache_hit) in zip(queries, fetched, strict=True):
        row = {
            "query": item["query"],
            "target_node_names": item["target_node_names"],
            "cache_hit": cache_hit,
            "results": results,
        }
        encoded = json.dumps(row, ensure_ascii=False)
        if used_chars + len(encoded) > character_budget:
            row["results"] = []
            row["truncated"] = True
            encoded = json.dumps(row, ensure_ascii=False)
        used_chars += len(encoded)
        rendered.append(row)
        if tracer:
            tracer.emit(TraceEvent(
                kind="phase1BMathlibSearchResult",
                thm_name=thm_name,
                turn=round_index,
                span_id=span_id,
                args={
                    "round": round_index,
                    "attempt": attempt,
                    "query": item["query"],
                    "targetNodeNames": item["target_node_names"],
                    "cacheHit": cache_hit,
                    "resultCount": len(row["results"]),
                    "truncated": bool(row.get("truncated")),
                },
                result=json.dumps(row["results"], ensure_ascii=False),
                ok=True,
            ))
    if protocol_errors:
        rendered.append({"protocol_errors": protocol_errors, "results": []})
    if tracer:
        tracer.emit(TraceEvent(
            kind="phase1BMathlibSearchEnd",
            thm_name=thm_name,
            turn=round_index,
            span_id=span_id,
            args={
                "round": round_index,
                "attempt": attempt,
                "queryCount": len(queries),
                "protocolErrors": protocol_errors,
                "outputCharacters": used_chars,
            },
            ok=not protocol_errors,
            duration_ms=(time.monotonic_ns() - started_ns) / 1_000_000,
        ))
    return rendered


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
    subgraph_max_edits: int = 8,
    closure_rounds: int = 0,
    mathlib_search_policy: str = "leanErrorsOnly",
    mathlib_search_max_queries_per_turn: int = 0,
    mathlib_search_max_results_per_query: int = 5,
) -> Blueprint:
    if repair_strategy not in REPAIR_STRATEGIES:
        raise ValueError(
            "phase1b_repair_strategy must be one of: "
            + ", ".join(sorted(REPAIR_STRATEGIES))
        )
    if closure_rounds < 0 or closure_rounds > max_rounds:
        raise ValueError("phase1b_closure_rounds must be between 0 and max_rounds")
    if mathlib_search_max_queries_per_turn < 0:
        raise ValueError("Phase-1B Mathlib search query limit must be non-negative")
    if mathlib_search_policy != "leanErrorsOnly":
        raise ValueError("phase1b_mathlib_search_policy must be leanErrorsOnly")
    if mathlib_search_max_results_per_query <= 0:
        raise ValueError("Phase-1B Mathlib search result limit must be positive")
    committed = blueprint
    initial_pending_names = tuple(
        node.name for node in committed.nodes
        if "PendingBlueprintClaim" in node.lean_declaration
    )
    previous_pending_names = initial_pending_names
    standalone_cache: dict[str, CompilerResult] = {}
    decompiler_cache: dict[str, FormalDecompilerResult] = {}
    comparator_cache: dict[str, StrictComparatorResult] = {}
    mathlib_search_cache: dict[str, list[dict[str, str]]] = {}
    ledger: dict[str, dict[str, Any]] = {}
    rejected_candidate_hashes: set[str] = set()
    edit_history: list[dict[str, Any]] = []
    previous_turn: dict[str, Any] | None = None
    foundation_debt_open = False

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
        closure_mode = closure_rounds > 0 and round_index > max_rounds - closure_rounds
        if closure_mode and tracer:
            tracer.emit(TraceEvent(
                kind="phase1BClosureModeStart", thm_name=thm_name,
                turn=round_index, args={
                    "round": round_index, "maxRounds": max_rounds,
                    "closureRounds": closure_rounds,
                },
            ))
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
        search_state: dict[str, Any] = {"used": False, "results": []}
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
                search_state=search_state,
                search_cache=mathlib_search_cache,
                search_max_queries=mathlib_search_max_queries_per_turn,
                search_max_results=mathlib_search_max_results_per_query,
                search_policy=mathlib_search_policy,
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
                "candidateHash": candidate_hash, "softDiagnostics": None,
                "mathlibSearchUsed": bool(search_state["used"]),
                "mathlibSearchResults": list(search_state["results"]),
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
            stable = _stable_gate(
                baseline_validation, candidate_validation,
                changed_nodes=hard["effectiveNodes"],
            )
            if tracer:
                tracer.emit(TraceEvent(
                    kind="phase1BStableGate", thm_name=thm_name,
                    turn=round_index, span_id=span_id,
                    args={
                        "round": round_index, "attempt": attempt,
                        "closureMode": closure_mode, **stable,
                    }, ok=bool(stable["passed"]),
                ))
            attempt_row["stableGate"] = stable
            if not stable["passed"]:
                rejected_candidate_hashes.add(candidate_hash)
                retry_feedback = {
                    "kind": "stableGate", "errors": stable["errors"],
                    "actual_nodes": hard["actualNodes"],
                    "effective_nodes": hard["effectiveNodes"],
                    "deterministic_diagnostics": _compact_validation(candidate_validation),
                }
                attempt_rows.append(attempt_row)
                if tracer:
                    tracer.emit(TraceEvent(
                        kind="phase1BEditorAttemptEnd", thm_name=thm_name,
                        turn=round_index, span_id=span_id, ok=False,
                        args={"round": round_index, "attempt": attempt,
                              "outcome": "stableRetry", "errors": stable["errors"]},
                        duration_ms=(time.monotonic_ns() - started) / 1_000_000,
                    ))
                continue
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
            assessment = _commit_assessment(
                baseline, candidate, baseline_validation, candidate_validation,
                stable=stable, hard_result=hard, plan=plan,
                semantic_manifest=semantic_manifest, closure_mode=closure_mode,
                foundation_debt_open=foundation_debt_open,
            )
            attempt_row["commitAssessment"] = assessment
            if tracer:
                tracer.emit(TraceEvent(
                    kind="phase1BSemanticProgressGate", thm_name=thm_name,
                    turn=round_index, span_id=span_id,
                    args={"round": round_index, "attempt": attempt, **assessment},
                    ok=bool(assessment["passed"]),
                ))
                if assessment["foundationNodes"]:
                    tracer.emit(TraceEvent(
                        kind="phase1BFoundationDetected", thm_name=thm_name,
                        turn=round_index, span_id=span_id,
                        args={"round": round_index, "attempt": attempt,
                              **{key: assessment[key] for key in (
                                  "foundationNodes", "foundationConsumers",
                                  "foundationOnly", "foundationIntegrated",
                                  "foundationDebtBefore",
                              )}},
                        ok=bool(assessment["foundationIntegrated"]),
                    ))
                if closure_mode and assessment["addedNodes"]:
                    tracer.emit(TraceEvent(
                        kind="phase1BClosureAuxiliaryValidation",
                        thm_name=thm_name, turn=round_index, span_id=span_id,
                        args={"round": round_index, "attempt": attempt,
                              **{key: assessment[key] for key in (
                                  "addedNodes", "resolvedObjectObligations",
                                  "foundationConsumers", "closureAuxiliaryAllowed",
                              )}},
                        ok=bool(assessment["closureAuxiliaryAllowed"]),
                    ))
                tracer.emit(TraceEvent(
                    kind="phase1BCommitAssessment", thm_name=thm_name,
                    turn=round_index, span_id=span_id,
                    args={"round": round_index, "attempt": attempt, **assessment},
                    ok=bool(assessment["passed"]),
                ))
            attempt_rows.append(attempt_row)
            if assessment["passed"]:
                committed = candidate
                validation = candidate_validation
                ledger = candidate_ledger
                previous_turn = None
                rejected_candidate_hashes.clear()
                committed_this_turn = True
                foundation_debt_before = foundation_debt_open
                if assessment["foundationOnly"]:
                    foundation_debt_open = True
                elif any(assessment["progressFlags"].values()):
                    foundation_debt_open = False
                if tracer and foundation_debt_before != foundation_debt_open:
                    tracer.emit(TraceEvent(
                        kind="phase1BFoundationDebtUpdated", thm_name=thm_name,
                        turn=round_index, args={
                            "round": round_index, "attempt": attempt,
                            "before": foundation_debt_before,
                            "after": foundation_debt_open,
                            "reason": assessment["reason"],
                        },
                    ))
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
                            "semanticDelta": delta,
                            "commitAssessment": assessment,
                            "reason": assessment["reason"],
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
                "kind": "semanticProgressGate",
                "reason": assessment["reason"],
                "errors": assessment["errors"],
                "candidate_formal_diff": _formal_diff(baseline, candidate, limit=10000),
                "actual_nodes": hard["actualNodes"],
                "effective_nodes": hard["effectiveNodes"],
                "no_op_nodes": hard["noOpNodes"],
                "soft_diagnostics": _compact_validation(candidate_validation),
                "semantic_delta": delta,
                "commit_assessment": assessment,
            }
            if tracer:
                tracer.emit(TraceEvent(
                    kind="phase1BEditorAttemptEnd", thm_name=thm_name,
                    turn=round_index, span_id=span_id, ok=False,
                    args={"round": round_index, "attempt": attempt,
                          "outcome": "semanticRetry",
                          "reason": assessment["reason"]},
                    duration_ms=(time.monotonic_ns() - started) / 1_000_000,
                ))

        turn_row = {
            "round": round_index, "strategy": repair_strategy,
            "closureMode": closure_mode,
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

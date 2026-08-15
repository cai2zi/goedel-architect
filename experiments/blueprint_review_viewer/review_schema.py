"""Versioned, read-only review artifacts for full Blueprint generations."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from . import REVIEW_SCHEMA_VERSION


_CANDIDATE_RE = re.compile(
    r"^(?P<kind>generation_round|phase1_failed_last)(?:_(?P<round>\d+))?"
    r"(?:_(?P<variant>submitted|canonical))?\.lean$"
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return default
    return value if value is not None else default


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return ""


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _cot_steps(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw = _json(row.get("cot_manifest_json"), {})
    steps = raw.get("steps", []) if isinstance(raw, dict) else []
    result: list[dict[str, Any]] = []
    for item in steps:
        if not isinstance(item, dict):
            continue
        result.append({
            "stepId": str(item.get("step_id", "")),
            "sourceStart": item.get("source_start"),
            "sourceEnd": item.get("source_end"),
            "sourceText": str(item.get("source_text", "")),
            "sourceSha256": str(item.get("source_sha256", "")),
        })
    return result


def _declaration_nodes(lean: str, cot_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse native Blueprint nodes if available; fall back gracefully.

    Importing the parser here means emitted artifacts carry exact source lines
    and dependencies, while the viewer itself still works for old/corrupt
    artifacts without requiring Lean or a model server.
    """
    step_by_id = {str(step["stepId"]): step for step in cot_steps}
    try:
        from blueprint import _parse_blueprint  # type: ignore

        blueprint = _parse_blueprint(lean, "root")
        result = []
        for node in blueprint.nodes:
            step = step_by_id.get(node.source_step_id, {})
            declaration = node.lean_declaration or node.full_declaration()
            result.append({
                "nodeName": node.name,
                "kind": node.kind,
                "stepId": node.source_step_id,
                "dependencies": list(node.dependencies),
                "declaration": declaration,
                "declarationSha256": _sha256(declaration),
                "leanRange": {"startLine": node.lean_start_line, "endLine": node.lean_end_line},
                "cotSource": step,
            })
        return result
    except Exception as exc:  # A broken candidate should remain reviewable.
        pattern = re.compile(r"(?m)^\s*(?:def|lemma|theorem)\s+([A-Za-z_][\w']*)")
        matches = list(pattern.finditer(lean))
        result = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(lean)
            declaration = lean[match.start():end].strip()
            start_line = lean.count("\n", 0, match.start()) + 1
            end_line = lean.count("\n", 0, end) + 1
            result.append({
                "nodeName": match.group(1), "kind": "unknown", "stepId": "",
                "dependencies": [], "declaration": declaration,
                "declarationSha256": _sha256(declaration),
                "leanRange": {"startLine": start_line, "endLine": end_line},
                "cotSource": {}, "parseWarning": str(exc),
            })
        return result


def _candidate_files(blueprint_dir: Path) -> list[Path]:
    if not blueprint_dir.is_dir():
        return []
    files = [path for path in blueprint_dir.glob("*.lean") if _CANDIDATE_RE.match(path.name)]
    rank = {"generation_round": 0, "phase1_failed_last": 1}
    def sort_key(path: Path) -> tuple[int, int, str]:
        match = _CANDIDATE_RE.match(path.name)
        assert match is not None
        return (rank.get(match.group("kind"), 99), int(match.group("round") or 0), path.name)
    return sorted(files, key=sort_key)


def _candidate_id(path: Path) -> str:
    return path.stem.replace("_", "-")


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _whole_graph_feedback(
    validation: dict[str, Any],
    deterministic_errors: list[dict[str, Any]],
) -> dict[str, Any]:
    failure_stage = str(validation.get("mechanicalFailureStage") or "")
    stage_reached = str(validation.get("mechanicalStageReached") or "")
    lean_success = validation.get("wholeFileLeanSuccess")
    canonical_success = validation.get("canonicalLeanSuccess")
    structural_errors = _string_list(validation.get("phase2StructuralErrors"))
    lean_errors = _string_list(validation.get("leanErrors"))
    canonical_errors = _string_list(validation.get("canonicalLeanErrors"))
    graph_errors = [
        item for item in deterministic_errors
        if str(item.get("stage") or "") != "phase2_standalone"
    ]
    compile_reached = stage_reached in {
        "canonical_lean", "phase2_standalone", "static_shadow",
        "formal_decompiler_or_comparator", "joint_semantic_audit",
    }
    if graph_errors or failure_stage in {
        "parse_basic", "canonical_rebuild", "phase2_contract", "canonical_lean",
    }:
        status = "failed"
    elif compile_reached and lean_success is True:
        status = "passed"
    elif not stage_reached:
        status = "missing"
    else:
        status = "notRun"
    return {
        "status": status,
        "stageReached": stage_reached,
        "failureStage": failure_stage,
        "compileReached": compile_reached,
        "wholeFileLeanSuccess": lean_success if isinstance(lean_success, bool) else None,
        "canonicalLeanSuccess": canonical_success if isinstance(canonical_success, bool) else None,
        "contractErrors": structural_errors,
        "leanErrors": lean_errors,
        "canonicalLeanErrors": canonical_errors,
        "errors": graph_errors,
    }


def _standalone_feedback(
    validation: dict[str, Any],
    deterministic_errors: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = validation.get("phase2StandaloneSummary")
    summary = dict(summary) if isinstance(summary, dict) else {}
    issues = _dict_list(validation.get("phase2StandaloneErrors"))
    errors = [
        item for item in deterministic_errors
        if str(item.get("stage") or "") == "phase2_standalone"
    ]
    not_run_reason = str(summary.get("notRunReason") or "")
    earlier_failure = next((
        str(item.get("stage") or "") for item in deterministic_errors
        if str(item.get("stage") or "") != "phase2_standalone"
    ), "")
    if issues or errors:
        status = "failed"
    elif not_run_reason or earlier_failure:
        status = "notRun"
    elif summary:
        status = "passed"
    else:
        status = "missing"
    return {
        "status": status,
        "notRunReason": not_run_reason or earlier_failure,
        "checkedNodeCount": summary.get("checkedNodeCount"),
        "cachedNodeCount": summary.get("cachedNodeCount"),
        "failedNodeCount": summary.get("failedNodeCount"),
        "durationMs": summary.get("durationMs"),
        "issues": issues,
        "errors": errors,
    }


def _semantic_feedback(
    validation: dict[str, Any],
    deterministic_errors: list[dict[str, Any]],
    semantic_errors: list[dict[str, Any]],
) -> dict[str, Any]:
    invoked = bool(validation.get("semanticAuditInvoked"))
    audit_error = validation.get("semanticAuditError")
    audit_error = dict(audit_error) if isinstance(audit_error, dict) else None
    audit = validation.get("semanticAudit")
    audit = dict(audit) if isinstance(audit, dict) else {}
    if audit_error is not None:
        status = "executionError"
    elif deterministic_errors:
        status = "notRun"
    elif invoked and semantic_errors:
        status = "failed"
    elif invoked:
        status = "passed"
    else:
        status = "notRun"
    failure_stage = str(validation.get("mechanicalFailureStage") or "")
    return {
        "status": status,
        "invoked": invoked,
        "notRunReason": failure_stage if status == "notRun" else "",
        "errors": semantic_errors,
        "executionError": audit_error,
        "audit": {
            "protocol": audit.get("protocol"),
            "mode": audit.get("mode", validation.get("semanticAuditMode")),
            "actualRequestCount": audit.get(
                "actualRequestCount", validation.get("semanticActualRequestCount")
            ),
            "cacheHits": audit.get("cacheHits", validation.get("semanticCacheHits")),
            "outputBudget": audit.get(
                "outputBudget", validation.get("semanticOutputBudget")
            ),
            "classification": audit.get("classification"),
        },
    }


def _answer_parts(value: Any) -> dict[str, Any]:
    payload = dict(value) if isinstance(value, dict) else {}
    thinking = payload.get("reasoning_content", payload.get("reasoningContent", ""))
    answer = payload.get("raw_content", payload.get("rawContent", ""))
    return {
        "available": bool(thinking or answer),
        "thinking": str(thinking or ""),
        "answer": str(answer or ""),
    }


def _builder_responses(events: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    responses: dict[int, dict[str, Any]] = {}
    for event in events:
        if event.get("kind") != "llm_response":
            continue
        args = event.get("args")
        if not isinstance(args, dict) or str(args.get("phase") or "") != "phase1":
            continue
        try:
            round_index = int(event.get("turn") or args.get("round"))
        except (TypeError, ValueError):
            continue
        responses[round_index] = {
            "thinking": str(args.get("reasoning_content") or ""),
            "messageContent": str(event.get("result") or ""),
            "finishReason": str(args.get("finish_reason") or ""),
        }
    return responses


def _generation_rounds(
    row: dict[str, Any], candidates: list[dict[str, Any]], events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    raw_history = row.get("generation_history")
    if not isinstance(raw_history, list):
        validation = row.get("generation_validation")
        raw_history = validation.get("generationRounds", []) if isinstance(validation, dict) else []
    generation_candidates = [
        candidate for candidate in candidates
        if candidate.get("kind") == "generation_round"
        and isinstance(candidate.get("round"), int)
    ]
    candidate_by_round = {
        int(candidate["round"]): candidate for candidate in generation_candidates
        if not candidate.get("variant")
    }
    submitted_by_round = {
        int(candidate["round"]): candidate for candidate in generation_candidates
        if candidate.get("variant") == "submitted"
    }
    canonical_by_round = {
        int(candidate["round"]): candidate for candidate in generation_candidates
        if candidate.get("variant") == "canonical"
    }
    builder_responses = _builder_responses(events)
    history_by_round = {
        int(item["round"]): item for item in raw_history
        if isinstance(item, dict) and str(item.get("round") or "").isdigit()
    }
    rounds: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw in raw_history:
        if not isinstance(raw, dict):
            continue
        try:
            round_index = int(raw.get("round"))
        except (TypeError, ValueError):
            continue
        if round_index <= 0 or round_index in seen:
            continue
        seen.add(round_index)
        validation = raw.get("validation")
        validation = dict(validation) if isinstance(validation, dict) else {}
        deterministic_errors = _dict_list(raw.get("deterministicErrors"))
        semantic_errors = _dict_list(raw.get("semanticErrors"))
        warnings = _dict_list(raw.get("warnings"))
        candidate = candidate_by_round.get(round_index)
        expected_hash = str(raw.get("candidateHash") or "")
        actual_hash = str(candidate.get("leanSha256") or "") if candidate else ""
        hash_matches = (
            actual_hash == expected_hash if candidate and expected_hash else None
        )
        whole_graph = _whole_graph_feedback(validation, deterministic_errors)
        standalone = _standalone_feedback(validation, deterministic_errors)
        audit = validation.get("semanticAudit")
        audit = dict(audit) if isinstance(audit, dict) else {}
        decompile = _answer_parts(audit.get("formalDecompiler"))
        compact = _answer_parts(audit.get("wholeCotComparator"))
        try:
            anchor_round = int(raw.get("semanticAnchorRound"))
        except (TypeError, ValueError):
            anchor_round = None
        try:
            structural_round = int(raw.get("structuralInputRound"))
        except (TypeError, ValueError):
            structural_round = None
        anchor_history = history_by_round.get(anchor_round, {}) if anchor_round else {}
        structural_history = (
            history_by_round.get(structural_round, {}) if structural_round else {}
        )
        anchor_candidate = candidate_by_round.get(anchor_round) if anchor_round else None
        structural_candidate = (
            candidate_by_round.get(structural_round) if structural_round else None
        )
        builder_trace = builder_responses.get(round_index, {})
        submitted_candidate = submitted_by_round.get(round_index)
        canonical_candidate = canonical_by_round.get(round_index)
        builder_answer = str(candidate.get("lean") or "") if candidate else ""
        submitted_blueprint = (
            str(submitted_candidate.get("lean") or "") if submitted_candidate else ""
        )
        canonical_blueprint = (
            str(canonical_candidate.get("lean") or "") if canonical_candidate else ""
        )
        if submitted_blueprint:
            displayed_submission = submitted_blueprint
            displayed_submission_source = "submittedArtifact"
            displayed_submission_exact = True
        elif builder_answer:
            # Legacy tool-call traces redact lean_code.  The same-round persisted
            # artifact is still the best reviewable representation of that
            # submission, although it may already be canonicalized.
            displayed_submission = builder_answer
            displayed_submission_source = "persistedArtifactFallback"
            displayed_submission_exact = False
        elif canonical_blueprint:
            displayed_submission = canonical_blueprint
            displayed_submission_source = "canonicalArtifactFallback"
            displayed_submission_exact = False
        else:
            displayed_submission = ""
            displayed_submission_source = "unavailable"
            displayed_submission_exact = False
        submitted_expected_hash = str(raw.get("submittedCandidateHash") or "")
        canonical_expected_hash = str(raw.get("canonicalCandidateHash") or "")
        if deterministic_errors:
            deterministic_status = "failed"
        elif whole_graph["status"] == "passed" and standalone["status"] == "passed":
            deterministic_status = "passed"
        elif "missing" in {whole_graph["status"], standalone["status"]}:
            deterministic_status = "missing"
        else:
            deterministic_status = "notRun"
        rounds.append({
            "round": round_index,
            "candidateId": candidate.get("candidateId") if candidate else "",
            "candidateAvailable": candidate is not None,
            "candidateHash": expected_hash,
            "candidateHashMatches": hash_matches,
            "semanticStage": int(raw.get("semanticStage") or round_index),
            "semanticAuditOrdinal": raw.get("semanticAuditOrdinal"),
            "semanticAuditInvoked": bool(raw.get("semanticAuditInvoked")),
            "semanticAnchorRound": anchor_round,
            "structuralInputRound": structural_round,
            "attemptRole": str(raw.get("attemptRole") or "legacy"),
            "contextFallbackApplied": bool(raw.get("contextFallbackApplied")),
            "inputTokens": raw.get("inputTokens"),
            "maxCompletionTokens": raw.get("maxCompletionTokens"),
            "feedback": {
                "deterministic": {
                    "status": deterministic_status,
                    "errorCount": len(deterministic_errors),
                    "errors": deterministic_errors,
                    "wholeGraph": whole_graph,
                    "phase2Standalone": standalone,
                },
                "semantic": _semantic_feedback(
                    validation, deterministic_errors, semantic_errors,
                ),
                "warnings": warnings,
            },
            "artifacts": {
                "decompileAnswer": decompile,
                "compactAnswer": compact,
                "builderInput": {
                    "semanticAnchorRound": anchor_round,
                    "semanticAnchorBlueprint": (
                        str(anchor_candidate.get("lean") or "")
                        if anchor_candidate else ""
                    ),
                    "semanticErrors": _dict_list(anchor_history.get("semanticErrors")),
                    "structuralInputRound": structural_round,
                    "structuralBlueprint": (
                        str(structural_candidate.get("lean") or "")
                        if structural_candidate else ""
                    ),
                    "structuralErrors": _dict_list(
                        structural_history.get("deterministicErrors")
                    ),
                    "contextFallbackApplied": bool(raw.get("contextFallbackApplied")),
                },
                "builderAnswer": {
                    "available": bool(
                        builder_trace or displayed_submission or canonical_blueprint
                        or builder_answer
                    ),
                    "thinking": str(builder_trace.get("thinking") or ""),
                    "messageContent": str(
                        builder_trace.get("messageContent")
                        or raw.get("builderMessageContent") or ""
                    ),
                    "submittedBlueprint": displayed_submission,
                    "submittedAvailable": bool(displayed_submission),
                    "submittedExact": displayed_submission_exact,
                    "submittedSource": displayed_submission_source,
                    "submittedHash": submitted_expected_hash,
                    "submittedHashMatches": (
                        str(submitted_candidate.get("leanSha256") or "")
                        == submitted_expected_hash
                        if submitted_candidate and submitted_expected_hash else None
                    ),
                    "canonicalBlueprint": canonical_blueprint,
                    "canonicalAvailable": canonical_candidate is not None,
                    "canonicalHash": canonical_expected_hash,
                    "canonicalHashMatches": (
                        str(canonical_candidate.get("leanSha256") or "")
                        == canonical_expected_hash
                        if canonical_candidate and canonical_expected_hash else None
                    ),
                    "persistedBlueprint": builder_answer,
                    "answer": builder_answer,
                    "finishReason": str(
                        builder_trace.get("finishReason") or raw.get("finishReason") or ""
                    ),
                },
            },
        })
    rounds.sort(key=lambda item: item["round"])
    for selected in rounds:
        stage = selected["semanticStage"]
        selected["semanticStageAttempts"] = [{
            "round": attempt["round"],
            "semanticStage": attempt["semanticStage"],
            "semanticAuditOrdinal": attempt["semanticAuditOrdinal"],
            "semanticAuditInvoked": attempt["semanticAuditInvoked"],
            "attemptRole": attempt["attemptRole"],
            "contextFallbackApplied": attempt["contextFallbackApplied"],
            "candidateHash": attempt["candidateHash"],
            "candidateHashMatches": attempt["candidateHashMatches"],
            "feedback": {"deterministic": attempt["feedback"]["deterministic"]},
            "artifacts": {
                "builderInput": attempt["artifacts"]["builderInput"],
                "builderAnswer": attempt["artifacts"]["builderAnswer"],
            },
        } for attempt in rounds if (
            attempt["semanticStage"] == stage
            and attempt["round"] <= selected["round"]
        )]
    return rounds


def _trace_events(row: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(str(row.get("trace_path") or ""))
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    events.append(payload)
    except (OSError, ValueError):
        return []
    return events


def _validation(row: dict[str, Any]) -> dict[str, Any]:
    validation = row.get("generation_validation")
    if isinstance(validation, dict):
        return validation
    return {}


def build_review_artifact(experiment_root: Path, row: dict[str, Any]) -> dict[str, Any]:
    experiment_root = experiment_root.resolve()
    blueprint_dir = Path(str(row.get("blueprint_dir") or ""))
    cot_steps = _cot_steps(row)
    candidates: list[dict[str, Any]] = []
    for path in _candidate_files(blueprint_dir):
        text = path.read_text(encoding="utf-8")
        match = _CANDIDATE_RE.match(path.name)
        candidates.append({
            "candidateId": _candidate_id(path), "kind": match.group("kind") if match else "snapshot",
            "round": int(match.group("round")) if match and match.group("round") else None,
            "variant": str(match.group("variant") or "") if match else "",
            "availability": "available", "leanPath": _safe_relative(path, experiment_root),
            "leanSha256": _sha256(text), "lean": text,
            "nodes": _declaration_nodes(text, cot_steps),
        })
    events = _trace_events(row)
    generation_rounds = _generation_rounds(row, candidates, events)
    last_round = generation_rounds[-1]["round"] if generation_rounds else None
    for candidate in candidates:
        candidate["feedbackRound"] = (
            candidate.get("round")
            if candidate.get("kind") == "generation_round"
            else last_round
        )
    validation = _validation(row)
    audit = validation.get("semanticAudit") if isinstance(validation, dict) else {}
    return {
        "schemaVersion": REVIEW_SCHEMA_VERSION,
        "source": {key: row.get(key, "") for key in ("id", "record_id", "source_id", "subset", "split", "theorem_name", "claimed_answer")},
        "result": {key: row.get(key, "") for key in ("status", "phase", "success", "root_proved", "error", "semantic_status")},
        "cotSteps": cot_steps,
        "candidates": candidates,
        "generationRounds": generation_rounds,
        "validation": validation,
        "semanticAudit": audit if isinstance(audit, dict) else {},
        "traceSummary": {"eventCount": len(events), "tracePath": _safe_relative(Path(str(row.get("trace_path") or "")), experiment_root)},
        "readOnly": True,
    }


def write_review_artifact(experiment_root: Path, row: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    artifact = build_review_artifact(experiment_root, row)
    raw_blueprint_dir = str(row.get("blueprint_dir") or "")
    if not raw_blueprint_dir:
        raise ValueError("Cannot write review artifact without blueprint_dir")
    blueprint_dir = Path(raw_blueprint_dir)
    path = blueprint_dir / "review.json"
    _atomic_json(path, artifact)
    return path, artifact


def index_entry(artifact_path: Path, experiment_root: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    source = artifact.get("source", {})
    result = artifact.get("result", {})
    return {"schemaVersion": REVIEW_SCHEMA_VERSION, "artifactPath": _safe_relative(artifact_path, experiment_root),
            "id": source.get("id", ""), "sourceId": source.get("source_id", ""),
            "subset": source.get("subset", ""), "status": result.get("status", ""),
            "candidateCount": len(artifact.get("candidates") or []),
            "updatedAtNs": os.stat(artifact_path).st_mtime_ns}

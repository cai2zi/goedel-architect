"""Build an evidence-first report for the minimal Claim/Scope experiment."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_blueprint_refine.claim_scope_manifest import (
    decode_claim_scope_manifest,
    unassigned_spans,
)
from cot_blueprint_refine.common import (
    extract_boxed_contents,
    latest_rows,
    write_json,
    write_jsonl,
)
from cot_blueprint_refine.llm_cot_splitter import atomize_cot
from blueprint import _parse_blueprint
from semantic_fidelity import validate_blueprint_fidelity


_SEMANTIC_CODE_RE = re.compile(r"- ([A-Z][A-Z0-9_]+)(?: | \[)")
_RELATION_RE = re.compile(
    r"(?:\$\$|\\boxed|(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?![A-Za-z])|"
    r"[=<>≤≥≠]|\\(?:frac|angle|mid|in)\b)", re.IGNORECASE,
)
_ASSERTION_WORD_RE = re.compile(
    r"\b(?:is|are|equals?|therefore|hence|implies?|exactly|only|prime|"
    r"divisible|maximum|minimum|largest|smallest|must|cannot|satisf(?:y|ies))\b",
    re.IGNORECASE,
)
_PRESENTATION_PREFIX_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:"
    r"to solve|we (?:now )?(?:analy[sz]e|compute|consider|begin)|"
    r"let['’]?s (?:reconsider|compute|simplify)|now (?:compute|expand|combine)|"
    r"using [^:]+:|from (?:this|these|our result)|"
    r"the derivation involves|final answer|understanding the problem|"
    r"modeling the problem|computing the values|strategy\b"
    r")",
    re.IGNORECASE,
)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def _distribution(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "p50": None, "p90": None, "max": None}
    floats = [float(value) for value in values]
    return {
        "count": len(values), "min": min(values), "mean": statistics.mean(floats),
        "p50": _percentile(floats, .5), "p90": _percentile(floats, .9), "max": max(values),
    }


def _compact(text: str, limit: int = 300) -> str:
    value = re.sub(r"\s+", " ", text).strip()
    return value if len(value) <= limit else value[:limit - 1] + "…"


def _refinement_attempt_stats(refinement: dict[str, Any]) -> list[dict[str, Any]]:
    path_text = str(refinement.get("conversation_path") or "")
    if path_text and Path(path_text).exists():
        conversation = json.loads(Path(path_text).read_text(encoding="utf-8"))
        attempts = []
        for event in conversation.get("events") or []:
            response = event.get("response") or {}
            usage = response.get("usage") or {}
            request = event.get("request") or {}
            attempts.append({
                "attempt": event.get("attempt"),
                "status": event.get("status"),
                "finish_reason": event.get("finish_reason"),
                "max_tokens": int(request.get("max_tokens") or 0),
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
            })
        if attempts:
            return attempts
    usage = (refinement.get("raw_response") or {}).get("usage") or {}
    return [{
        "attempt": refinement.get("attempts"),
        "status": refinement.get("status"),
        "finish_reason": refinement.get("finish_reason"),
        "max_tokens": int(refinement.get("request_max_tokens") or 0),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
    }] if refinement else []


def _result_class(row: dict[str, Any]) -> str:
    if row.get("status") == "phase1_accepted":
        return "phase1_accepted"
    if row.get("status") == "exhausted":
        return "phase2_3_exhausted"
    if row.get("root_proved"):
        return "root_proved"
    error = str(row.get("error") or "")
    if "local semantic-fidelity gate" in error:
        return "local_semantic_gate"
    if "semantic-fidelity audit" in error:
        return "llm_semantic_audit"
    if row.get("phase") == "phase3":
        return "phase3_error"
    if "Lean compilation" in error or "did not compile" in error:
        return "phase1_lean_compile"
    if "finish_reason=length" in error or "output limit" in error:
        return "phase1_length"
    return "phase1_other"


def _trace_stats(path: Path) -> dict[str, Any]:
    request_count = 0
    usage = Counter()
    usage_by_phase: dict[str, Counter[str]] = {}
    phases = Counter()
    finish_reasons = Counter()
    durations: list[float] = []
    lean_durations: list[float] = []
    lean_batch_sizes: list[int] = []
    if not path.exists():
        return {
            "llm_request_count": 0, "usage": {}, "usage_by_phase": {}, "llm_phases": {},
            "finish_reasons": {}, "llm_duration_ms": [], "lean_duration_ms": [],
            "lean_batch_sizes": [],
        }
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            args = event.get("args") or {}
            if event.get("kind") == "llm_request_end":
                request_count += 1
                phase = str(args.get("phase") or event.get("phase") or "unknown")
                phases[phase] += 1
                finish_reasons[str(args.get("finish_reason") or "unknown")] += 1
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    if args.get(key) is not None:
                        usage[key] += int(args[key])
                        usage_by_phase.setdefault(phase, Counter())[key] += int(args[key])
                if event.get("duration_ms") is not None:
                    durations.append(float(event["duration_ms"]))
            elif event.get("kind") == "lean_check_result":
                timings = args.get("timings") or {}
                if timings.get("client_http_ms") is not None:
                    lean_durations.append(float(timings["client_http_ms"]))
                if timings.get("batch_size") is not None:
                    lean_batch_sizes.append(int(timings["batch_size"]))
    return {
        "llm_request_count": request_count,
        "usage": dict(usage),
        "usage_by_phase": {
            phase: dict(values) for phase, values in sorted(usage_by_phase.items())
        },
        "llm_phases": dict(phases),
        "finish_reasons": dict(finish_reasons),
        "llm_duration_ms": durations,
        "lean_duration_ms": lean_durations,
        "lean_batch_sizes": lean_batch_sizes,
    }


def _claim_risks(claim: dict[str, Any]) -> list[str]:
    text = str(claim["source_text"])
    stripped = text.strip()
    compact = _compact(text, 1000)
    risks: list[str] = []
    atoms = atomize_cot(text)
    if len(text) > 700 or len(atoms) > 5:
        risks.append("coarse_or_compound")
    if (
        _PRESENTATION_PREFIX_RE.match(compact)
        and not _RELATION_RE.search(compact)
        and not _ASSERTION_WORD_RE.search(compact)
    ):
        risks.append("non_propositional_presentation")
    if compact.endswith(":") and not _RELATION_RE.search(compact):
        risks.append("dangling_lead_in")
    if re.fullmatch(r"#{1,6}[^\n]{1,120}", stripped):
        risks.append("heading_only")
    return risks


def _scope_risks(scope: dict[str, Any]) -> list[str]:
    text = _compact(str(scope["source_text"]), 1000)
    risks: list[str] = []
    target_count = len(scope["applies_to_claim_ids"])
    if target_count >= 20:
        risks.append("overbroad_target_range")
    if (
        str(scope["scope_type"]) == "case_condition"
        and not _RELATION_RE.search(text)
        and not re.search(r"\b(?:if|when|assume|real|complex|grade|all|none)\b", text, re.I)
    ):
        risks.append("layout_as_scope")
    if re.search(
        r"\b(?:therefore|we (?:only need|conclude|obtain|find)|this (?:shows|proves)|"
        r"is uniformly distributed|equals exactly)\b",
        text,
        re.IGNORECASE,
    ):
        risks.append("complete_assertion_as_scope")
    return risks


def _persisted_semantic_issues(
    result: dict[str, Any],
    manifest: dict[str, Any],
    *,
    claimed_answer: str,
) -> list[dict[str, str]]:
    """Recompute every static issue from the persisted rejected candidate.

    The human-facing exception intentionally formats at most 20 issues.  It is
    therefore not a lossless analysis source; using it previously undercounted
    failures in candidates with many vacuous nodes.
    """
    if _result_class(result) != "local_semantic_gate":
        return []
    candidate_path = Path(str(result.get("blueprint_dir") or "")) / "phase1_failed_last.lean"
    if not candidate_path.exists():
        return []
    try:
        blueprint = _parse_blueprint(
            candidate_path.read_text(encoding="utf-8"),
            str(result.get("root_theorem") or result.get("theorem_name") or ""),
        )
        return [
            issue.to_dict()
            for issue in validate_blueprint_fidelity(
                blueprint,
                manifest,
                claimed_answer=claimed_answer,
                require_step_bindings=True,
            )
        ]
    except (OSError, TypeError, ValueError):
        # Preserve report generation for malformed/legacy artifacts.  The
        # truncated error codes below remain a best-effort fallback.
        return []


def analyze(root: Path, baseline_root: Path | None = None) -> dict[str, Any]:
    prepared = latest_rows(root / "prepared" / "generation_inputs.jsonl", "name")
    results = {
        str(row.get("source_id") or ""): row
        for row in latest_rows(root / "robustpa" / "blueprint" / "results.jsonl", "source_id")
    }
    annotation_audits = {
        str(row.get("ID") or ""): row
        for row in latest_rows(root / "claim_scope_quality_audit" / "results.jsonl", "ID")
    }
    refinements = {
        str(row.get("ID") or ""): row
        for row in latest_rows(
            root / "refinement" / "blueprint" / "refined_predictions.jsonl", "ID"
        )
    }
    evaluations = {
        str(row.get("ID") or ""): row
        for row in latest_rows(root / "evaluation" / "blueprint" / "comparison.jsonl", "ID")
    }
    baseline_evaluations: dict[str, dict[str, dict[str, Any]]] = {}
    baseline_prepared: dict[str, dict[str, Any]] = {}
    baseline_annotation_audits: dict[str, dict[str, Any]] = {}
    if baseline_root is not None:
        baseline_prepared = {
            str(row.get("name") or ""): row
            for row in latest_rows(
                baseline_root / "prepared" / "generation_inputs.jsonl", "name"
            )
        }
        baseline_annotation_audits = {
            str(row.get("ID") or ""): row
            for row in latest_rows(
                baseline_root / "claim_scope_quality_audit" / "results.jsonl", "ID"
            )
        }
        for arm in ("blueprint", "cot_only"):
            baseline_evaluations[arm] = {
                str(row.get("ID") or ""): row
                for row in latest_rows(
                    baseline_root / "evaluation" / arm / "comparison.jsonl", "ID"
                )
            }
    per_sample: list[dict[str, Any]] = []
    all_scope_types = Counter()
    all_error_codes = Counter()
    all_result_classes = Counter()
    llm_usage = Counter()
    llm_usage_by_phase: dict[str, Counter[str]] = {}
    finish_reasons = Counter()
    llm_durations: list[float] = []
    lean_durations: list[float] = []
    lean_batch_sizes: list[int] = []
    audit_codes = Counter()
    audit_verdicts = Counter()
    refine_statuses = Counter()
    refine_finish_reasons = Counter()
    context_qualities = Counter()
    for row in sorted(prepared, key=lambda item: str(item.get("name") or "")):
        source_id = str(row["name"])
        manifest = decode_claim_scope_manifest(row["cot_manifest_json"], source=str(row["post_think_cot"]))
        claims = list(manifest["claims"])
        scopes = list(manifest["scopes"])
        claim_risks = [
            {"claim_id": claim["claim_id"], "risks": risks, "text": _compact(claim["source_text"])}
            for claim in claims if (risks := _claim_risks(claim))
        ]
        scope_risks = [
            {"scope_id": scope["scope_id"], "risks": risks, "text": _compact(scope["source_text"])}
            for scope in scopes if (risks := _scope_risks(scope))
        ]
        context_risks = []
        for start, end, text in unassigned_spans(manifest):
            semantic_text = re.sub(r"(?m)^\s*#{1,6}[^\n]*\n?", "", text)
            semantic_text = re.sub(r"(?m)^\s*-{3,}\s*$", "", semantic_text)
            compact = _compact(semantic_text)
            if not compact:
                continue
            if _RELATION_RE.search(compact) or _ASSERTION_WORD_RE.search(compact):
                context_risks.append({
                    "source_start": start, "source_end": end, "text": compact,
                })
        result = results.get(source_id, {})
        annotation_audit = annotation_audits.get(source_id, {})
        audit_issues = list(annotation_audit.get("issues") or [])
        audit_codes.update(str(issue.get("code") or "") for issue in audit_issues)
        audit_verdicts[str(annotation_audit.get("verdict") or "MISSING")] += 1
        refinement = refinements.get(source_id, {})
        evaluation = evaluations.get(source_id, {})
        old_blueprint = baseline_evaluations.get("blueprint", {}).get(source_id, {})
        old_cot_only = baseline_evaluations.get("cot_only", {}).get(source_id, {})
        baseline_row = baseline_prepared.get(source_id, {})
        baseline_claims: list[dict[str, Any]] = []
        baseline_scopes: list[dict[str, Any]] = []
        if baseline_row.get("cot_manifest_json"):
            baseline_manifest = decode_claim_scope_manifest(
                baseline_row["cot_manifest_json"],
                source=str(baseline_row.get("post_think_cot") or ""),
            )
            baseline_claims = list(baseline_manifest["claims"])
            baseline_scopes = list(baseline_manifest["scopes"])
        baseline_audit = baseline_annotation_audits.get(source_id, {})
        refined_boxes = extract_boxed_contents(str(refinement.get("refined_cot") or ""))
        refined_answer = refined_boxes[-1] if refined_boxes else ""
        refinement_attempts = _refinement_attempt_stats(refinement)
        completion_tokens = int(
            refinement_attempts[-1]["completion_tokens"] if refinement_attempts else 0
        )
        request_max_tokens = int(refinement.get("request_max_tokens") or 0)
        if refinement:
            refine_statuses[str(refinement.get("status") or "unknown")] += 1
            refine_finish_reasons[str(refinement.get("finish_reason") or "unknown")] += 1
            context_qualities[str(refinement.get("context_quality") or "unknown")] += 1
        result_class = _result_class(result) if result else "missing_result"
        semantic_issues = _persisted_semantic_issues(
            result,
            manifest,
            claimed_answer=str(row.get("claimed_answer") or result.get("claimed_answer") or ""),
        )
        error_codes = (
            [str(issue["code"]) for issue in semantic_issues]
            if semantic_issues
            else _SEMANTIC_CODE_RE.findall(str(result.get("error") or ""))
        )
        trace_path = Path(str(result.get("trace_path") or ""))
        trace = _trace_stats(trace_path)
        all_scope_types.update(str(scope["scope_type"]) for scope in scopes)
        all_error_codes.update(error_codes)
        all_result_classes[result_class] += 1
        llm_usage.update(trace["usage"])
        for phase, usage in trace["usage_by_phase"].items():
            llm_usage_by_phase.setdefault(phase, Counter()).update(usage)
        finish_reasons.update(trace["finish_reasons"])
        llm_durations.extend(trace["llm_duration_ms"])
        lean_durations.extend(trace["lean_duration_ms"])
        lean_batch_sizes.extend(trace["lean_batch_sizes"])
        per_sample.append({
            "source_id": source_id,
            "source": row.get("source"),
            "claim_count": len(claims),
            "scope_count": len(scopes),
            "scope_types": [str(scope["scope_type"]) for scope in scopes],
            "scoped_claim_count": sum(bool(claim.get("scope_ids")) for claim in claims),
            "unassigned_context_chars": sum(end - start for start, end, _text in unassigned_spans(manifest)),
            "claim_risks": claim_risks,
            "scope_risks": scope_risks,
            "context_math_risks": context_risks,
            "baseline": {
                "claim_count": len(baseline_claims) if baseline_row else None,
                "scope_count": len(baseline_scopes) if baseline_row else None,
                "claim_risk_count": sum(bool(_claim_risks(claim)) for claim in baseline_claims),
                "audit_verdict": baseline_audit.get("verdict"),
                "audit_issue_count": len(baseline_audit.get("issues") or []),
            },
            "result_class": result_class,
            "status": result.get("status"),
            "phase": result.get("phase"),
            "root_proved": bool(result.get("root_proved")),
            "total_nodes": int(result.get("total_nodes") or 0),
            "proved_nodes": int(result.get("proved_node_count") or 0),
            "semantic_error_codes": error_codes,
            "semantic_issues": semantic_issues,
            "error_excerpt": _compact(str(result.get("error") or ""), 800),
            "annotation_audit": {
                "status": annotation_audit.get("status"),
                "verdict": annotation_audit.get("verdict"),
                "issues": audit_issues,
            },
            "cot_refine": {
                "status": refinement.get("status"),
                "finish_reason": refinement.get("finish_reason"),
                "context_quality": refinement.get("context_quality"),
                "blueprint_used": refinement.get("blueprint_used"),
                "blueprint_truncated": refinement.get("blueprint_truncated"),
                "input_tokens": refinement.get("input_tokens"),
                "completion_tokens": completion_tokens,
                "request_max_tokens": request_max_tokens,
                "attempt_stats": refinement_attempts,
                "attempt_completion_tokens_total": sum(
                    int(attempt["completion_tokens"]) for attempt in refinement_attempts
                ),
                "near_output_limit": bool(
                    any(
                        int(attempt["max_tokens"])
                        and int(attempt["completion_tokens"]) >= .95 * int(attempt["max_tokens"])
                        for attempt in refinement_attempts
                    )
                ),
                "latency_s": refinement.get("latency_s"),
                "refined_answer": refined_answer,
                "answer_changed_exactly": bool(
                    refined_answer and refined_answer != str(row.get("claimed_answer") or "")
                ),
                "error": refinement.get("error"),
                "final_envelope_warning": refinement.get("final_envelope_warning"),
            },
            "evaluation": {
                "current_blueprint_correct": (
                    bool(evaluation.get("after_math_verify_correct")) if evaluation else None
                ),
                "old_blueprint_correct": (
                    bool(old_blueprint.get("after_math_verify_correct"))
                    if old_blueprint else None
                ),
                "old_cot_only_correct": (
                    bool(old_cot_only.get("after_math_verify_correct"))
                    if old_cot_only else None
                ),
            },
            "trace": {
                "llm_request_count": trace["llm_request_count"],
                "usage": trace["usage"],
                "usage_by_phase": trace["usage_by_phase"],
                "llm_phases": trace["llm_phases"],
                "finish_reasons": trace["finish_reasons"],
            },
        })
    summary = {
        "experiment_root": str(root),
        "rows": len(per_sample),
        "claims": {
            "total": sum(row["claim_count"] for row in per_sample),
            "per_row": _distribution([row["claim_count"] for row in per_sample]),
            "rows_with_risk": sum(bool(row["claim_risks"]) for row in per_sample),
            "risk_count": sum(len(row["claim_risks"]) for row in per_sample),
            "risk_types": dict(sorted(Counter(
                risk
                for row in per_sample
                for item in row["claim_risks"]
                for risk in item["risks"]
            ).items())),
        },
        "scopes": {
            "total": sum(row["scope_count"] for row in per_sample),
            "per_row": _distribution([row["scope_count"] for row in per_sample]),
            "types": dict(sorted(all_scope_types.items())),
            "rows_with_risk": sum(bool(row["scope_risks"]) for row in per_sample),
            "risk_count": sum(len(row["scope_risks"]) for row in per_sample),
            "risk_types": dict(sorted(Counter(
                risk
                for row in per_sample
                for item in row["scope_risks"]
                for risk in item["risks"]
            ).items())),
        },
        "context": {
            "unassigned_chars": sum(row["unassigned_context_chars"] for row in per_sample),
            "rows_with_math_risk": sum(bool(row["context_math_risks"]) for row in per_sample),
            "math_risk_count": sum(len(row["context_math_risks"]) for row in per_sample),
        },
        "baseline_claim_scope": {
            "root": str(baseline_root) if baseline_root is not None else None,
            "available": len(baseline_prepared),
            "claims_total": sum(
                int(row["baseline"]["claim_count"] or 0) for row in per_sample
            ),
            "scopes_total": sum(
                int(row["baseline"]["scope_count"] or 0) for row in per_sample
            ),
            "claim_risk_count": sum(
                int(row["baseline"]["claim_risk_count"] or 0) for row in per_sample
            ),
            "audit_fail": sum(
                row["baseline"]["audit_verdict"] == "FAIL" for row in per_sample
            ),
        },
        "blueprint": {
            "result_classes": dict(sorted(all_result_classes.items())),
            "root_proved": sum(row["root_proved"] for row in per_sample),
            "phase1_accepted": sum(
                row["result_class"] == "phase1_accepted" for row in per_sample
            ),
            "semantic_error_codes": dict(all_error_codes.most_common()),
        },
        "annotation_audit": {
            "available": len(annotation_audits),
            "verdicts": dict(sorted(audit_verdicts.items())),
            "issue_codes": dict(audit_codes.most_common()),
        },
        "cot_refine": {
            "available": len(refinements),
            "statuses": dict(sorted(refine_statuses.items())),
            "finish_reasons": dict(sorted(refine_finish_reasons.items())),
            "context_qualities": dict(sorted(context_qualities.items())),
            "blueprint_truncated": sum(
                bool(row["cot_refine"]["blueprint_truncated"]) for row in per_sample
            ),
            "answer_changed_exactly": sum(
                bool(row["cot_refine"]["answer_changed_exactly"]) for row in per_sample
            ),
            "near_output_limit": sum(
                bool(row["cot_refine"]["near_output_limit"]) for row in per_sample
            ),
            "completion_tokens": _distribution([
                int(row["cot_refine"]["completion_tokens"])
                for row in per_sample if row["cot_refine"]["status"] is not None
            ]),
            "completion_tokens_total": sum(
                int(row["cot_refine"]["attempt_completion_tokens_total"] or 0)
                for row in per_sample
            ),
            "envelope_warnings": dict(Counter(
                str(row["cot_refine"]["final_envelope_warning"])
                for row in per_sample if row["cot_refine"]["final_envelope_warning"]
            )),
            "latency_s": _distribution([
                float(row["cot_refine"]["latency_s"])
                for row in per_sample if row["cot_refine"]["latency_s"] is not None
            ]),
        },
        "evaluation": {
            "current_blueprint_correct": sum(
                row["evaluation"]["current_blueprint_correct"] is True for row in per_sample
            ),
            "current_valid_output_accuracy": (
                sum(row["evaluation"]["current_blueprint_correct"] is True for row in per_sample)
                / len(evaluations)
            ) if evaluations else None,
            "current_available": len(evaluations),
            "old_blueprint_correct": sum(
                row["evaluation"]["old_blueprint_correct"] is True for row in per_sample
            ) if baseline_root is not None else None,
            "old_cot_only_correct": sum(
                row["evaluation"]["old_cot_only_correct"] is True for row in per_sample
            ) if baseline_root is not None else None,
            "baseline_root": str(baseline_root) if baseline_root is not None else None,
        },
        "runtime": {
            "llm_usage": dict(llm_usage),
            "llm_usage_by_phase": {
                phase: dict(values) for phase, values in sorted(llm_usage_by_phase.items())
            },
            "finish_reasons": dict(finish_reasons),
            "length_finish_samples": [
                row["source_id"] for row in per_sample
                if int(row["trace"]["finish_reasons"].get("length", 0)) > 0
            ],
            "llm_request_duration_ms": _distribution(llm_durations),
            "lean_http_duration_ms": _distribution(lean_durations),
            "lean_batch_sizes": dict(sorted(Counter(lean_batch_sizes).items())),
        },
    }
    analysis_dir = root / "analysis"
    write_json(analysis_dir / "claim_scope_analysis.json", {"summary": summary, "samples": per_sample})
    write_jsonl(analysis_dir / "claim_scope_per_sample.jsonl", per_sample)
    lines = [
        "# Claim/Scope Phase-1-only semantic-fidelity analysis", "",
        f"- Rows: {summary['rows']}",
        f"- Claims: {summary['claims']['total']} (mean {summary['claims']['per_row']['mean']:.2f}, "
        f"P90 {summary['claims']['per_row']['p90']})",
        f"- Scopes: {summary['scopes']['total']}",
        f"- Baseline Claims/Scopes: {summary['baseline_claim_scope']['claims_total']}/"
        f"{summary['baseline_claim_scope']['scopes_total']}",
        f"- Phase-1 accepted: {summary['blueprint']['phase1_accepted']}/{summary['rows']}",
        f"- Root proved: {summary['blueprint']['root_proved']}/{summary['rows']}",
        f"- Result classes: `{json.dumps(summary['blueprint']['result_classes'], sort_keys=True)}`", "",
        f"- Claim/Scope audit: `{json.dumps(summary['annotation_audit']['verdicts'], sort_keys=True)}`",
        f"- COT refine: `{json.dumps(summary['cot_refine']['statuses'], sort_keys=True)}`; "
        f"length={summary['cot_refine']['finish_reasons'].get('length', 0)}", "",
        f"- Math-Verify evaluation rows: {summary['evaluation']['current_available']} (Phase-1-only intentionally skips COT refine/Judge)",
        f"- Historical Math-Verify correct: current Blueprint {summary['evaluation']['current_blueprint_correct']}; "
        f"old Blueprint {summary['evaluation']['old_blueprint_correct']}; "
        f"old COT-only {summary['evaluation']['old_cot_only_correct']}", "",
        "## Interpretation", "",
        "- The persisted representation contains only Claim and shared Scope. Deterministic "
        "boundary normalization removes high-confidence headings, bare lead-ins, and pure "
        "method narration without rewriting mathematical assertions.",
        "- The LLM organization audit is diagnostic rather than ground truth. Its NONCLAIM count "
        "is directionally corroborated by deterministic heading/lead-in risks, but individual "
        "judgments require inspection.",
        "- Phase-1-only deliberately stops before node proving, Blueprint refine, COT refine, and "
        "Judge. A rejection therefore measures translation/compilation/audit quality rather than "
        "proof-search success.", "",
        "## Per sample", "",
        "| source_id | claims new/old | scopes new/old | audit | result | nodes/proved | semantic codes | refine | correct new/oldBP/oldCOT |",
        "|---|---:|---:|---|---|---:|---|---|---|",
    ]
    for row in per_sample:
        codes = ",".join(dict.fromkeys(row["semantic_error_codes"][:4])) or "-"
        audit = row["annotation_audit"]
        audit_label = ",".join(dict.fromkeys(
            str(issue["code"]) for issue in audit["issues"][:3]
        )) or str(audit["verdict"] or "-")
        refine = row["cot_refine"]
        evaluation = row["evaluation"]
        score = "/".join(
            "-" if value is None else ("1" if value else "0")
            for value in (
                evaluation["current_blueprint_correct"],
                evaluation["old_blueprint_correct"],
                evaluation["old_cot_only_correct"],
            )
        )
        lines.append(
            f"| {row['source_id']} | {row['claim_count']}/{row['baseline']['claim_count'] or '-'} | "
            f"{row['scope_count']}/{row['baseline']['scope_count'] or '-'} | "
            f"{audit_label} | {row['result_class']} | {row['total_nodes']}/{row['proved_nodes']} | "
            f"{codes} | {refine['status'] or '-'}:{refine['finish_reason'] or '-'} | {score} |"
        )
    (analysis_dir / "claim_scope_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    args = parser.parse_args()
    baseline_root = args.baseline_root.resolve() if args.baseline_root else None
    print(json.dumps(analyze(args.root.resolve(), baseline_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


STEP_ID_RE = re.compile(r"\bS\d{3}(?:\.[A-Za-z0-9]+)?\b")
NODE_BLOCK_RE = re.compile(
    r"@\[blueprint\s*(.*?)\]\s*\n\s*"
    r"(noncomputable\s+def|def|abbrev|lemma|theorem)\s+(\w+)(.*?)"
    r"(?=@\[blueprint|\Z)",
    re.DOTALL,
)
PROP_TRUE_RE = re.compile(
    r"\b(?:def|abbrev)\s+\w+[^\n:]*:\s*Prop\s*:=\s*True\b", re.DOTALL,
)
ROOT_TRUE_TEMPLATE = r"\btheorem\s+{name}\b.*?:\s*True\s*:=\s*by\b"
TERMINAL_STATUSES = {"solved", "exhausted", "error"}


def _selection_sha256(selected_ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for row_id in sorted(set(str(value) for value in selected_ids)):
        digest.update(row_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def latest(rows: Iterable[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for row in rows:
        item = str(row.get(key) or "")
        if item:
            values[item] = row
    return list(values.values())


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]


def quantiles(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "p50": percentile(values, 0.5),
        "p90": percentile(values, 0.9),
        "max": max(values) if values else None,
        "sum": sum(values),
    }


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _decode_manifest(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("cot_manifest") or row.get("cot_steps") or row.get("cot_manifest_json")
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if value:
        try:
            decoded = json.loads(str(value))
            if isinstance(decoded, list):
                return [item for item in decoded if isinstance(item, dict)]
        except json.JSONDecodeError:
            pass
    source = str(row.get("post_think_cot") or row.get("informal_proof") or "")
    if not source:
        return []
    try:
        from cot_blueprint_refine.cot_steps import split_cot_steps

        return split_cot_steps(source)
    except (ImportError, ValueError, TypeError):
        return []


def _checkpoint(result: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(result.get("checkpoint_path") or ""))
    value = read_json(path, {}) if path.is_file() else {}
    return value if isinstance(value, dict) else {}


def _node_mappings(lean_text: str) -> list[dict[str, Any]]:
    mappings: list[dict[str, Any]] = []
    for match in NODE_BLOCK_RE.finditer(lean_text):
        kind = match.group(2).replace("noncomputable ", "")
        ids = sorted(set(STEP_ID_RE.findall(match.group(1) + match.group(4))))
        mappings.append({"name": match.group(3), "kind": kind, "step_ids": ids})
    return mappings


def _root_claim(lean_text: str, root_name: str) -> str:
    if not root_name:
        return ""
    pattern = re.compile(
        rf"\btheorem\s+{re.escape(root_name)}\b(.*?)(?::=\s*by|:=\s*by|\Z)",
        re.DOTALL,
    )
    match = pattern.search(lean_text)
    if not match:
        return ""
    value = match.group(1)
    colon = value.rfind(":")
    return value[colon + 1 :].strip() if colon >= 0 else value.strip()


def _obvious_violations(lean_text: str, root_name: str, claimed_answer: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not lean_text:
        return counts
    counts["prop_true"] = len(PROP_TRUE_RE.findall(lean_text))
    if root_name and re.search(
        ROOT_TRUE_TEMPLATE.format(name=re.escape(root_name)), lean_text, re.DOTALL,
    ):
        counts["root_true"] += 1
    claim = _root_claim(lean_text, root_name)
    compact = re.sub(r"[\s()]", "", claim)
    if compact.count("=") == 1:
        left, right = compact.split("=", 1)
        if left and left == right:
            counts["root_self_equality"] += 1
    if claim and re.search(r"∃\s+\w+[^,]*,\s*\w+\s*=", claim):
        counts["root_unconstrained_exists_candidate"] += 1
    simple_answer = re.fullmatch(r"[-+]?\d+(?:\s*/\s*\d+)?", claimed_answer.strip())
    if simple_answer:
        answer = re.escape(re.sub(r"\s+", "", claimed_answer))
        for match in re.finditer(r"\b(?:def|abbrev)\s+\w+.*?:=\s*([^\n]+)", lean_text):
            rhs = re.sub(r"\s+", "", match.group(1))
            if re.fullmatch(rf"(?:\([^)]*\))?{answer}", rhs):
                counts["answer_literal_definition_candidate"] += 1
    return +counts


def _semantic_metadata(checkpoint: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "semantic_fidelity", "semantic_summary", "semantic_validation",
        "semantic_gate_history", "semantic_audit_history", "semantic_drift_history",
        "semantic_repair_history", "semantic_contract_snapshot", "semantic_gate_results",
        "semantic_status", "semantic_fidelity_enabled", "semantic_require_step_ids",
        "semantic_static_gate", "semantic_minimal_ir", "semantic_freeze_refinement",
        "semantic_audit_mode", "proof_policy",
    )
    payload: dict[str, Any] = {}
    for key in keys:
        if key in checkpoint:
            payload[key] = checkpoint[key]
        elif key in result:
            payload[key] = result[key]
    return payload


def trace_cost(results: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, int]]:
    request_counts: Counter[str] = Counter()
    prompt_tokens: Counter[str] = Counter()
    completion_tokens: Counter[str] = Counter()
    durations: defaultdict[str, list[float]] = defaultdict(list)
    event_counts: Counter[str] = Counter()
    audit_flags: Counter[str] = Counter()
    retry_count = 0
    length_count = 0
    lean_timings: defaultdict[str, list[float]] = defaultdict(list)
    per_sample: dict[str, int] = {}
    for result in results:
        row_id = str(result.get("source_id") or result.get("ID") or "")
        total_for_row = 0
        path = Path(str(result.get("trace_path") or ""))
        for event in read_jsonl(path):
            kind = str(event.get("kind") or "")
            event_counts[kind] += 1
            args = event.get("args") if isinstance(event.get("args"), dict) else {}
            if kind == "llm_request_end":
                operation = str(
                    args.get("operation") or args.get("stage") or args.get("phase") or "unknown"
                )
                request_counts[operation] += 1
                prompt = _safe_int(args.get("prompt_tokens"))
                completion = _safe_int(args.get("completion_tokens"))
                prompt_tokens[operation] += prompt
                completion_tokens[operation] += completion
                total_for_row += _safe_int(args.get("total_tokens")) or prompt + completion
                if isinstance(event.get("duration_ms"), (int, float)):
                    durations[operation].append(float(event["duration_ms"]))
                if args.get("finish_reason") == "length":
                    length_count += 1
                if _safe_int(args.get("retry_index")) > 0:
                    retry_count += 1
            if kind == "llm_response" and args.get("operation") == "semantic_audit":
                match = re.search(
                    r"\[\[SEMANTIC_AUDIT=(PASS|FAIL)\]\]", str(event.get("result") or ""),
                )
                audit_flags[match.group(1) if match else "FORMAT_ERROR"] += 1
            if kind in {"lean_check_result", "tool_result"}:
                timings = args.get("timings") if isinstance(args.get("timings"), dict) else {}
                for key in (
                    "micro_batch_wait_ms", "client_inflight_wait_ms", "client_http_ms",
                    "repl_wait_ms", "lean_exec_wall_ms", "server_total_ms",
                ):
                    if isinstance(timings.get(key), (int, float)):
                        lean_timings[key].append(float(timings[key]))
        if row_id:
            per_sample[row_id] = total_for_row
    operations = {}
    for operation in sorted(request_counts):
        operations[operation] = {
            "requests": request_counts[operation],
            "prompt_tokens": prompt_tokens[operation],
            "completion_tokens": completion_tokens[operation],
            "total_tokens": prompt_tokens[operation] + completion_tokens[operation],
            # This is aggregate concurrent request work, never wall time.
            "request_work_duration_ms": quantiles(durations[operation]),
        }
    return ({
        "operations": operations,
        "requests": sum(request_counts.values()),
        "prompt_tokens": sum(prompt_tokens.values()),
        "completion_tokens": sum(completion_tokens.values()),
        "total_tokens": sum(prompt_tokens.values()) + sum(completion_tokens.values()),
        "retry_requests": retry_count,
        "finish_reason_length": length_count,
        "event_counts": dict(sorted(event_counts.items())),
        "semantic_event_counts": {
            key: value for key, value in sorted(event_counts.items())
            if "semantic" in key or "fidelity" in key or "step_manifest" in key
        },
        "semantic_audit_flags": dict(sorted(audit_flags.items())),
        "lean_timings_ms": {
            key: quantiles(values) for key, values in sorted(lean_timings.items())
        },
    }, per_sample)


def _runtime(root: Path) -> dict[str, Any]:
    value = read_json(root / "robustpa/blueprint/runtime_history.json", {})
    if not isinstance(value, dict):
        return {"wall_time_s": None, "wall_time_human": None}
    return {
        "wall_time_s": value.get("total_elapsed_s"),
        "wall_time_human": value.get("total_elapsed_time"),
    }


def _refine_summary(root: Path, selected_ids: set[str]) -> tuple[dict[str, Any], dict[str, bool]]:
    comparisons = [
        row for row in latest(
            read_jsonl(root / "evaluation/blueprint/comparison.jsonl"), "ID",
        )
        if str(row.get("ID") or "") in selected_ids
    ]
    per_sample = {
        str(row.get("ID") or ""): bool(row.get("after_math_verify_correct"))
        for row in comparisons
    }
    refined = [
        row for row in latest(
            read_jsonl(root / "refinement/blueprint/refined_predictions.jsonl"), "ID",
        )
        if str(row.get("ID") or "") in selected_ids
    ]
    usage = Counter()
    for row in refined:
        response = row.get("raw_response") if isinstance(row.get("raw_response"), dict) else {}
        tokens = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage[key] += _safe_int(tokens.get(key))
    return ({
        "available": bool(refined or comparisons),
        "rows": len(refined),
        "status_counts": dict(Counter(str(row.get("status") or "missing") for row in refined)),
        "strict_math_verify_correct": sum(per_sample.values()),
        "comparison_rows": len(comparisons),
        "blueprint_truncated": sum(bool(row.get("blueprint_truncated")) for row in refined),
        "usage": dict(usage),
    }, per_sample)


def arm_summary(
    label: str,
    root: Path,
    selected_ids: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_results = latest(read_jsonl(root / "robustpa/blueprint/results.jsonl"), "source_id")
    results = [row for row in all_results if str(row.get("source_id") or "") in selected_ids]
    if not results:
        return ({"label": label, "root": str(root), "available": False}, [])
    generation = {
        str(row.get("name") or ""): row
        for row in latest(read_jsonl(root / "prepared/generation_inputs.jsonl"), "name")
    }
    contexts = {
        str(row.get("ID") or ""): row
        for row in latest(
            read_jsonl(root / "blueprint_contexts/blueprint_contexts.jsonl"), "ID",
        )
    }
    refine, refine_per_sample = _refine_summary(root, selected_ids)
    cost, token_per_sample = trace_cost(results)
    per_sample: list[dict[str, Any]] = []
    statuses: Counter[str] = Counter()
    failure_stages: Counter[str] = Counter()
    context_quality: Counter[str] = Counter()
    node_signals: Counter[str] = Counter()
    violations: Counter[str] = Counter()
    semantic_metadata_rows = 0
    total_steps = 0
    mapped_steps = 0
    assertion_nodes = 0
    mapped_assertion_nodes = 0
    multi_source_assertion_nodes = 0
    orphan_step_refs = 0
    formally_negated = 0
    infra_error_nodes = 0
    phase1_success = 0
    iter1 = 0
    for result in sorted(results, key=lambda row: str(row.get("source_id") or "")):
        row_id = str(result.get("source_id") or "")
        statuses[str(result.get("status") or "missing")] += 1
        failure_stage = str(
            result.get("failed_blueprint_failure_stage")
            or (result.get("phase") if result.get("status") == "error" else "")
            or ""
        )
        if failure_stage:
            failure_stages[failure_stage] += 1
        checkpoint = _checkpoint(result)
        lean_text = str(checkpoint.get("blueprint_lean_file") or "")
        if lean_text:
            phase1_success += 1
        manifest = _decode_manifest(generation.get(row_id, checkpoint))
        source_ids = {
            str(step.get("step_id") or "") for step in manifest if step.get("step_id")
        }
        mappings = _node_mappings(lean_text)
        referenced = {step_id.split(".", 1)[0] for node in mappings for step_id in node["step_ids"]}
        mapped = source_ids & referenced
        total_steps += len(source_ids)
        mapped_steps += len(mapped)
        assertions = [node for node in mappings if node["kind"] in {"lemma", "theorem"}]
        assertion_nodes += len(assertions)
        mapped_assertion_nodes += sum(len(node["step_ids"]) == 1 for node in assertions)
        multi_source_assertion_nodes += sum(len(node["step_ids"]) > 1 for node in assertions)
        orphan_step_refs += len(referenced - source_ids)
        claimed = str(
            result.get("claimed_answer")
            or generation.get(row_id, {}).get("claimed_answer")
            or checkpoint.get("claimed_answer")
            or ""
        )
        obvious = _obvious_violations(
            lean_text, str(checkpoint.get("blueprint_target") or result.get("root_theorem") or ""),
            claimed,
        )
        violations.update(obvious)
        metadata = _semantic_metadata(checkpoint, result)
        semantic_metadata_rows += bool(metadata)
        signals = Counter(
            str(value.get("signal") or "")
            for value in (checkpoint.get("node_results") or {}).values()
            if isinstance(value, dict)
        )
        node_signals.update(signals)
        formally_negated += signals["formally_negated"]
        infra_error_nodes += _safe_int(result.get("infra_error_node_count"))
        context = contexts.get(row_id, {})
        context_quality[str(context.get("context_quality") or "missing")] += 1
        iter1 += _safe_int(result.get("iterations")) >= 1
        per_sample.append({
            "arm": label,
            "ID": row_id,
            "status": str(result.get("status") or "missing"),
            "terminal": str(result.get("status") or "") in TERMINAL_STATUSES,
            "root_proved": bool(result.get("root_proved")),
            "iterations": _safe_int(result.get("iterations")),
            "context_quality": str(context.get("context_quality") or "missing"),
            "step_count": len(source_ids),
            "mapped_step_count": len(mapped),
            "step_coverage": len(mapped) / len(source_ids) if source_ids else None,
            "assertion_nodes": len(assertions),
            "mapped_assertion_nodes": sum(len(node["step_ids"]) == 1 for node in assertions),
            "multi_source_assertion_nodes": sum(len(node["step_ids"]) > 1 for node in assertions),
            "orphan_step_refs": len(referenced - source_ids),
            "obvious_violation_count": sum(obvious.values()),
            "obvious_violations": dict(obvious),
            "formally_negated_nodes": signals["formally_negated"],
            "semantic_metadata_available": bool(metadata),
            "semantic_metadata": metadata,
            "robustpa_llm_tokens": token_per_sample.get(row_id, 0),
            "refine_math_verify_correct": refine_per_sample.get(row_id),
            "trace_path": str(result.get("trace_path") or ""),
            "checkpoint_path": str(result.get("checkpoint_path") or ""),
        })
    summary = {
        "label": label,
        "root": str(root),
        "available": True,
        "rows": len(results),
        "terminal": sum(str(row.get("status") or "") in TERMINAL_STATUSES for row in results),
        "status_counts": dict(sorted(statuses.items())),
        "failure_stage_counts": dict(sorted(failure_stages.items())),
        "phase1_success": phase1_success,
        "iter1_rows": iter1,
        "root_proved": sum(bool(row.get("root_proved")) for row in results),
        "context_quality": dict(sorted(context_quality.items())),
        "node_signals": dict(sorted(node_signals.items())),
        "formally_negated_nodes": formally_negated,
        "infra_error_nodes": infra_error_nodes,
        "mapping": {
            "manifest_rows": sum(row["step_count"] > 0 for row in per_sample),
            "total_steps": total_steps,
            "mapped_steps": mapped_steps,
            "step_coverage": mapped_steps / total_steps if total_steps else None,
            "assertion_nodes": assertion_nodes,
            "mapped_assertion_nodes": mapped_assertion_nodes,
            "assertion_single_source_rate": (
                mapped_assertion_nodes / assertion_nodes if assertion_nodes else None
            ),
            "multi_source_assertion_nodes": multi_source_assertion_nodes,
            "orphan_step_refs": orphan_step_refs,
        },
        "obvious_violations_heuristic": dict(sorted(violations.items())),
        "semantic_metadata_rows": semantic_metadata_rows,
        "cost": cost,
        "runtime": _runtime(root),
        "refine": refine,
    }
    return summary, per_sample


def _gate(value: bool | None) -> str:
    return "UNAVAILABLE" if value is None else "PASS" if value else "FAIL"


def quality_gates(summary: dict[str, Any], expected_rows: int) -> dict[str, str]:
    manifest = summary.get("manifest") if isinstance(summary.get("manifest"), dict) else {}
    record_validation = (
        manifest.get("blueprint_validation")
        if isinstance(manifest.get("blueprint_validation"), dict)
        else None
    )
    record_validation_gate = _gate(
        bool(record_validation.get("passed")) if record_validation is not None else None
    )
    refine_validation = (
        manifest.get("refine_validation")
        if isinstance(manifest.get("refine_validation"), dict) else None
    )
    evaluate_validation = (
        manifest.get("evaluate_validation")
        if isinstance(manifest.get("evaluate_validation"), dict) else None
    )
    refine_validation_gate = _gate(
        bool(refine_validation.get("passed")) if refine_validation is not None else None
    )
    evaluate_validation_gate = _gate(
        bool(evaluate_validation.get("passed")) if evaluate_validation is not None else None
    )
    if not summary.get("available"):
        return {
            "available": "UNAVAILABLE",
            "record_completion_validation": record_validation_gate,
            "refine_completion_validation": refine_validation_gate,
            "evaluate_completion_validation": evaluate_validation_gate,
        }
    mapping = summary["mapping"]
    semantic_available = _safe_int(summary.get("semantic_metadata_rows")) > 0
    violations = sum(_safe_int(value) for value in summary["obvious_violations_heuristic"].values())
    return {
        "record_completion_validation": record_validation_gate,
        "refine_completion_validation": refine_validation_gate,
        "evaluate_completion_validation": evaluate_validation_gate,
        "all_rows_terminal": _gate(summary.get("terminal") == expected_rows),
        "no_infra_error": _gate(_safe_int(summary.get("infra_error_nodes")) == 0),
        "all_steps_mapped_or_explicit": _gate(
            mapping.get("step_coverage") == 1.0 if mapping.get("total_steps") else None
        ),
        "assertion_nodes_have_single_source": _gate(
            mapping.get("assertion_single_source_rate") == 1.0
            if mapping.get("assertion_nodes") else None
        ),
        "no_obvious_accepted_degeneration": _gate(violations == 0),
        # Exact accepted/attempted drift requires the new checkpoint metadata.
        "semantic_metadata_present": _gate(semantic_available),
    }


def _pairwise_rows(
    summaries: dict[str, dict[str, Any]],
    rows_by_arm: dict[str, list[dict[str, Any]]],
    parents: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm, parent in parents.items():
        if arm not in rows_by_arm or parent not in rows_by_arm:
            continue
        child_rows = rows_by_arm[arm]
        parent_rows = rows_by_arm[parent]
        if not child_rows or not parent_rows:
            continue
        child_index = {str(row["ID"]): row for row in child_rows}
        parent_index = {str(row["ID"]): row for row in parent_rows}
        common = sorted(set(child_index) & set(parent_index))
        if not common:
            continue
        paired_refine = [
            row_id for row_id in common
            if child_index[row_id].get("refine_math_verify_correct") is not None
            and parent_index[row_id].get("refine_math_verify_correct") is not None
        ]
        rows.append({
            "arm": arm,
            "parent": parent,
            "paired_rows": len(common),
            "root_gained": sum(
                child_index[row_id]["root_proved"] and not parent_index[row_id]["root_proved"]
                for row_id in common
            ),
            "root_lost": sum(
                parent_index[row_id]["root_proved"] and not child_index[row_id]["root_proved"]
                for row_id in common
            ),
            "violation_rows_improved": sum(
                child_index[row_id]["obvious_violation_count"]
                < parent_index[row_id]["obvious_violation_count"]
                for row_id in common
            ),
            "violation_rows_regressed": sum(
                child_index[row_id]["obvious_violation_count"]
                > parent_index[row_id]["obvious_violation_count"]
                for row_id in common
            ),
            "refine_paired_rows": len(paired_refine),
            "refine_gained": sum(
                bool(child_index[row_id]["refine_math_verify_correct"])
                and not bool(parent_index[row_id]["refine_math_verify_correct"])
                for row_id in paired_refine
            ),
            "refine_lost": sum(
                bool(parent_index[row_id]["refine_math_verify_correct"])
                and not bool(child_index[row_id]["refine_math_verify_correct"])
                for row_id in paired_refine
            ),
            "robustpa_token_delta": (
                summaries[arm].get("cost", {}).get("total_tokens", 0)
                - summaries[parent].get("cost", {}).get("total_tokens", 0)
            ),
            "wall_time_delta_s": (
                _safe_float(summaries[arm].get("runtime", {}).get("wall_time_s"))
                - _safe_float(summaries[parent].get("runtime", {}).get("wall_time_s"))
            ),
        })
    return rows


def build_matrix_report(
    *,
    matrix_root: Path,
    arms: Iterable[str],
    legacy_root: Path,
    subset_metrics_path: Path,
    selected_ids: Iterable[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    arm_list = list(arms)
    subset = read_json(subset_metrics_path, {})
    subset_ids = {str(value) for value in (subset or {}).get("selected_ids", [])}
    if not subset_ids:
        raise ValueError(f"no selected_ids in {subset_metrics_path}")
    requested_ids = (
        {str(value) for value in selected_ids if str(value)}
        if selected_ids is not None else set()
    )
    if not requested_ids:
        suite = read_json(matrix_root / "matrix_manifest.json", {}) or {}
        persisted_ids = suite.get("selected_ids") if isinstance(suite, dict) else None
        if isinstance(persisted_ids, list):
            requested_ids = {str(value) for value in persisted_ids if str(value)}
    selected_id_set = requested_ids or subset_ids
    unknown_ids = sorted(selected_id_set - subset_ids)
    if unknown_ids:
        raise ValueError(
            f"report selected_ids are outside the frozen subset: {unknown_ids}"
        )
    summaries: dict[str, dict[str, Any]] = {}
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    legacy, legacy_rows = arm_summary("legacy_E0", legacy_root, selected_id_set)
    legacy["quality_gates"] = quality_gates(legacy, len(selected_id_set))
    summaries["legacy_E0"] = legacy
    rows_by_arm["legacy_E0"] = legacy_rows
    parents: dict[str, str] = {}
    for arm in arm_list:
        arm_root = matrix_root / arm
        manifest = read_json(arm_root / "semantic_run_manifest.json", {}) or {}
        summary, rows = arm_summary(arm, arm_root, selected_id_set)
        summary["manifest"] = manifest
        summary["quality_gates"] = quality_gates(summary, len(selected_id_set))
        summaries[arm] = summary
        rows_by_arm[arm] = rows
        parents[arm] = str(manifest.get("parent") or "")
    pairwise = _pairwise_rows(summaries, rows_by_arm, parents)
    all_rows = [
        row for label in ["legacy_E0", *arm_list] for row in rows_by_arm.get(label, [])
    ]
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "matrix_root": str(matrix_root),
        "legacy_root": str(legacy_root),
        "subset_metrics": str(subset_metrics_path),
        "selected_rows": len(selected_id_set),
        "subset_rows": len(subset_ids),
        "selected_ids": sorted(selected_id_set),
        "selection_sha256": _selection_sha256(selected_id_set),
        "arms": summaries,
        "pairwise": pairwise,
        "notes": [
            "LLM request duration sums are concurrent request-work, not wall time.",
            "obvious_violations_heuristic is diagnostic; exact gate/drift verdicts come from semantic metadata.",
            "UNAVAILABLE quality gates are never treated as passes.",
        ],
    }
    return report, all_rows, pairwise


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Wrong-answer-76 semantic fidelity matrix", "",
        f"Selected rows: **{report['selected_rows']}** of **{report.get('subset_rows', report['selected_rows'])}**.", "",
        "| Arm | Rows | Terminal | Phase1 | Root proved | Step coverage | Single-source assertions | Heuristic violations | Negated | Refine correct | RobustPA tokens | Wall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, item in report["arms"].items():
        if not item.get("available"):
            lines.append(f"| {arm} | unavailable | — | — | — | — | — | — | — | — | — | — |")
            continue
        mapping = item["mapping"]
        violations = sum(_safe_int(v) for v in item["obvious_violations_heuristic"].values())
        lines.append(
            f"| {arm} | {item['rows']} | {item['terminal']} | {item['phase1_success']} | "
            f"{item['root_proved']} | {_fmt(mapping['step_coverage'])} | "
            f"{_fmt(mapping['assertion_single_source_rate'])} | {violations} | "
            f"{item['formally_negated_nodes']} | "
            f"{item['refine']['strict_math_verify_correct'] if item['refine']['available'] else '—'} | "
            f"{item['cost']['total_tokens']} | {_fmt(item['runtime']['wall_time_s'])} s |"
        )
    lines.extend(["", "## Quality gates", ""])
    for arm, item in report["arms"].items():
        gates = item.get("quality_gates", {})
        lines.append(f"- **{arm}**: " + ", ".join(f"`{key}={value}`" for key, value in gates.items()))
    lines.extend(["", "## Pairwise deltas", ""])
    if report["pairwise"]:
        lines.extend([
            "| Arm | Parent | Paired | Root +/− | Violations improved/regressed | Refine +/− | Token Δ | Wall Δ |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in report["pairwise"]:
            lines.append(
                f"| {row['arm']} | {row['parent']} | {row['paired_rows']} | "
                f"{row['root_gained']}/{row['root_lost']} | "
                f"{row['violation_rows_improved']}/{row['violation_rows_regressed']} | "
                f"{row['refine_gained']}/{row['refine_lost']} | "
                f"{row['robustpa_token_delta']} | {_fmt(row['wall_time_delta_s'])} s |"
            )
    else:
        lines.append("No comparable completed parent/child arms yet.")
    lines.extend(["", "## Interpretation notes", ""])
    lines.extend(f"- {note}" for note in report["notes"])
    return "\n".join(lines) + "\n"


def write_matrix_report(
    *,
    matrix_root: Path,
    arms: Iterable[str],
    legacy_root: Path,
    subset_metrics_path: Path,
    selected_ids: Iterable[str] | None = None,
) -> Path:
    arm_list = list(arms)
    report, per_sample, pairwise = build_matrix_report(
        matrix_root=matrix_root,
        arms=arm_list,
        legacy_root=legacy_root,
        subset_metrics_path=subset_metrics_path,
        selected_ids=selected_ids,
    )
    matrix_root.mkdir(parents=True, exist_ok=True)
    json_path = matrix_root / "semantic_quality_report.json"
    md_path = matrix_root / "semantic_quality_report.md"
    rows_path = matrix_root / "semantic_quality_per_sample.jsonl"
    pairwise_path = matrix_root / "semantic_quality_pairwise.csv"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(markdown(report), encoding="utf-8")
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in per_sample:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    fields = list(pairwise[0]) if pairwise else ["arm", "parent", "paired_rows"]
    with pairwise_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pairwise)
    return md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the semantic fidelity matrix report.")
    parser.add_argument("--matrix-root", required=True, type=Path)
    parser.add_argument("--arms", default="E0,E1,E2,E3,E4,E5,E6,R1,R2,R3,R4,R5,R6")
    parser.add_argument("--legacy-root", required=True, type=Path)
    parser.add_argument("--subset-metrics", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = write_matrix_report(
        matrix_root=args.matrix_root.resolve(),
        arms=[item.strip().upper() for item in args.arms.split(",") if item.strip()],
        legacy_root=args.legacy_root.resolve(),
        subset_metrics_path=args.subset_metrics.resolve(),
    )
    print(output)


if __name__ == "__main__":
    main()

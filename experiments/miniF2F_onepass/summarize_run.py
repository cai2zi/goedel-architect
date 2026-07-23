from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

try:
    from blueprint import _parse_blueprint, phase2_contract_error_counts, phase2_contract_errors
except Exception:  # pragma: no cover - keeps trace-only summarization usable.
    _parse_blueprint = None
    phase2_contract_error_counts = None
    phase2_contract_errors = None


EVENT_COLUMNS = [
    "experiment_id",
    "sample_id",
    "thm_name",
    "kind",
    "phase",
    "turn",
    "attempt",
    "model",
    "operation",
    "tool_name",
    "call_id",
    "success",
    "ok",
    "error_type",
    "error_message",
    "status_code",
    "retry_index",
    "max_retries",
    "retryable",
    "exhausted",
    "request_id",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "timestamp",
    "timestamp_unix_s",
    "wall_time_s",
    "proof_body",
    "raw_result",
    "target",
    "validated",
    "error_count",
    "warning_count",
    "goal_count",
    "query",
    "k",
    "args_json",
    "extra_json",
]

SAMPLE_COLUMNS = [
    "experiment_id",
    "sample_id",
    "source_id",
    "split",
    "final_success",
    "phase0_success",
    "phase1_success",
    "phase2_success",
    "blueprint_success",
    "blueprint_reused",
    "phase1_skipped",
    "root_theorem",
    "num_theorems",
    "proved_node_count",
    "proved_ratio",
    "num_llm_calls",
    "num_llm_errors",
    "llm_exhausted_count",
    "num_tool_calls",
    "num_compile_attempts",
    "phase1_compile_attempts",
    "phase2_compile_attempts",
    "proof_error_count",
    "proof_error_message_count",
    "infra_error_count",
    "timeout_count",
    "total_tokens",
    "phase1_tokens",
    "phase2_tokens",
    "wall_time",
    "trace_event_count",
    "first_success_attempt",
    "first_success_phase",
    "first_success_turn",
    "failure_category",
    "node_signal_counts",
    "failed_nodes",
    "proved_nodes",
    "statement_wrong_count",
    "formally_negated_count",
    "proof_too_hard_count",
    "infra_node_count",
    "sorry_error_count",
    "contract_error_counts",
    "blueprint_fully_validated",
    "checkpoint_done",
    "checkpoint_success",
    "result_error",
    "checkpoint_path",
    "trace_path",
    "blueprint_path",
]


INFRA_PATTERNS = (
    "api connection",
    "apiconnectionerror",
    "api status",
    "apistatuserror",
    "apitimeouterror",
    "bad gateway",
    "connection error",
    "connection reset",
    "internal server",
    "kimina request failed",
    "rate limit",
    "server did not return",
    "error code: 400",
    "error code: 408",
    "error code: 429",
    "error code: 500",
    "error code: 502",
    "error code: 503",
    "error code: 504",
    "error code: 524",
    "service unavailable",
    "status_code=400",
    "status_code=408",
    "status_code=429",
    "status_code=500",
    "status_code=502",
    "status_code=503",
    "status_code=504",
    "status_code=524",
    "transport_error",
    "upstream",
    "origin web server",
    "未接收到上游响应",
    "消息流出现异常",
)

TIMEOUT_PATTERNS = (
    "timeout",
    "timed out",
    "proxy read timeout",
    "status_code=524",
    " 524",
    "120-second",
    "超时",
)

SORRY_PATTERNS = (
    "proof contains `sorry`",
    "declaration uses `sorry`",
    "unexpected sorry",
)


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_filename(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._-")
    return safe or "experiment"


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _utc_timestamp(ts: Any) -> str:
    if not isinstance(ts, (int, float)):
        return ""
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"1", "true", "yes", "y"}:
            return True
        if lower in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _text_contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(pattern in lower for pattern in patterns)


def _is_infra_message(text: str) -> bool:
    return _text_contains_any(text, INFRA_PATTERNS)


def _is_timeout_message(text: str) -> bool:
    return _text_contains_any(text, TIMEOUT_PATTERNS)


def _is_sorry_message(text: str) -> bool:
    return _text_contains_any(text, SORRY_PATTERNS)


def _load_metrics(run_root: Path) -> dict[str, Any]:
    path = run_root / "metrics.json"
    if not path.exists():
        return {}
    try:
        data = _read_json(path)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def infer_experiment_id(run_root: Path, metrics: dict[str, Any] | None = None) -> str:
    metrics = metrics or {}
    parts: list[str] = []
    if run_root.parent.name:
        parts.append(run_root.parent.name)
    if run_root.name:
        parts.append(run_root.name)
    split = str(metrics.get("split") or "").strip()
    if split and split not in parts:
        parts.append(split)
    return "_".join(_safe_filename(part) for part in parts if part) or "experiment"


def _latest_results(run_root: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(run_root / "results.jsonl"):
        row_id = row.get("id")
        if row_id is not None:
            latest[str(row_id)] = row
    return latest


def _empty_trace_stats() -> dict[str, Any]:
    return {
        "event_count": 0,
        "kind_counts": Counter(),
        "num_llm_calls": 0,
        "num_llm_errors": 0,
        "num_tool_calls": 0,
        "phase1_compile_attempts": 0,
        "phase2_compile_attempts": 0,
        "proof_error_count": 0,
        "proof_error_message_count": 0,
        "infra_error_count": 0,
        "llm_exhausted_count": 0,
        "timeout_count": 0,
        "sorry_error_count": 0,
        "tokens": 0,
        "phase1_tokens": 0,
        "phase2_tokens": 0,
        "first_ts": None,
        "last_ts": None,
        "first_success_attempt": None,
        "first_success_phase": "",
        "first_success_turn": None,
        "compile_attempt_index": 0,
    }


def _infer_phase(kind: str, args: dict[str, Any]) -> str:
    phase = args.get("phase")
    if phase:
        return str(phase)
    if kind in {"lean_check_result", "llm_response"}:
        return "phase1"
    if kind in {"resume"}:
        return "phase1"
    if kind in {"theorem_start", "tool_call", "tool_result", "model_text", "final_verify"}:
        return "phase2"
    return ""


def _event_success(event: dict[str, Any], args: dict[str, Any]) -> bool | None:
    if "success" in args:
        return _as_bool(args.get("success"))
    if "validated" in args and event.get("kind") == "lean_check_result":
        return _as_bool(event.get("ok"))
    return _as_bool(event.get("ok"))


def _event_error_message(args: dict[str, Any]) -> str:
    if args.get("message"):
        return str(args.get("message"))
    if args.get("error"):
        return str(args.get("error"))
    errors = args.get("errors")
    if isinstance(errors, list):
        return "\n".join(str(item) for item in errors)
    if errors:
        return str(errors)
    return ""


def _event_proof_body(kind: str, args: dict[str, Any]) -> str:
    if args.get("proof_body"):
        return str(args.get("proof_body"))
    if args.get("proof"):
        return str(args.get("proof"))
    return ""


def _flatten_event(
    *,
    experiment_id: str,
    sample_id: str,
    event: dict[str, Any],
    trace_line: int,
    trace_path: Path,
) -> dict[str, Any]:
    args = event.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    kind = str(event.get("kind") or "")
    phase = _infer_phase(kind, args)
    result = event.get("result")
    raw_result = result if isinstance(result, str) or result is None else _json_compact(result)
    errors = args.get("errors") if isinstance(args.get("errors"), list) else []
    warnings = args.get("warnings") if isinstance(args.get("warnings"), list) else []
    goals = args.get("goals") if isinstance(args.get("goals"), list) else []
    row = {
        "experiment_id": experiment_id,
        "sample_id": sample_id,
        "thm_name": str(event.get("thm_name") or ""),
        "kind": kind,
        "phase": phase,
        "turn": _as_int(args.get("turn")) if args.get("turn") is not None else _as_int(event.get("turn")),
        "attempt": _as_int(args.get("attempt")),
        "model": str(args.get("model") or ""),
        "operation": str(args.get("operation") or ""),
        "tool_name": str(event.get("tool_name") or ""),
        "call_id": str(event.get("call_id") or ""),
        "success": _event_success(event, args),
        "ok": _as_bool(event.get("ok")),
        "error_type": str(args.get("error_type") or ""),
        "error_message": _event_error_message(args),
        "status_code": _as_int(args.get("status_code")),
        "retry_index": _as_int(args.get("retry_index")),
        "max_retries": _as_int(args.get("max_retries")),
        "retryable": _as_bool(args.get("retryable")),
        "exhausted": _as_bool(args.get("exhausted")),
        "request_id": str(args.get("request_id") or ""),
        "finish_reason": str(args.get("finish_reason") or ""),
        "prompt_tokens": _as_int(args.get("prompt_tokens")),
        "completion_tokens": _as_int(args.get("completion_tokens")),
        "total_tokens": _as_int(args.get("total_tokens")),
        "timestamp": _utc_timestamp(event.get("ts")),
        "timestamp_unix_s": _as_float(event.get("ts")),
        "wall_time_s": _as_float(args.get("wall_time_s")),
        "proof_body": _event_proof_body(kind, args),
        "raw_result": raw_result,
        "target": str(args.get("target") or ""),
        "validated": _as_bool(args.get("validated")),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "goal_count": len(goals),
        "query": str(args.get("query") or ""),
        "k": _as_int(args.get("k")),
        "args_json": _json_compact(args),
        "extra_json": _json_compact({
            "trace_path": str(trace_path),
            "trace_line": trace_line,
            "event": event,
        }),
    }
    return row


def _update_trace_stats(stats: dict[str, Any], row: dict[str, Any], event: dict[str, Any]) -> None:
    kind = row["kind"]
    phase = row["phase"]
    stats["event_count"] += 1
    stats["kind_counts"][kind] += 1

    ts = row.get("timestamp_unix_s")
    if isinstance(ts, (int, float)):
        stats["first_ts"] = ts if stats["first_ts"] is None else min(stats["first_ts"], ts)
        stats["last_ts"] = ts if stats["last_ts"] is None else max(stats["last_ts"], ts)

    if kind == "llm_usage":
        stats["num_llm_calls"] += 1
        tokens = row.get("total_tokens") or 0
        stats["tokens"] += tokens
        if phase == "phase1":
            stats["phase1_tokens"] += tokens
        elif phase == "phase2":
            stats["phase2_tokens"] += tokens
    elif kind == "llm_error":
        stats["num_llm_errors"] += 1
        stats["infra_error_count"] += 1
        if row.get("exhausted"):
            stats["llm_exhausted_count"] += 1
        message = " ".join([
            str(row.get("error_type") or ""),
            str(row.get("error_message") or ""),
            str(row.get("status_code") or ""),
        ])
        if _is_timeout_message(message):
            stats["timeout_count"] += 1
    elif kind == "tool_call":
        stats["num_tool_calls"] += 1
        if row.get("tool_name") == "lean_compile":
            stats["phase2_compile_attempts"] += 1
            stats["compile_attempt_index"] += 1
    elif kind == "lean_check_result":
        stats["phase1_compile_attempts"] += 1
        stats["compile_attempt_index"] += 1

    if kind in {"tool_result", "lean_check_result"} and (
        row.get("tool_name") == "lean_compile" or kind == "lean_check_result"
    ):
        success = bool(row.get("success"))
        if success and stats["first_success_attempt"] is None:
            stats["first_success_attempt"] = stats["compile_attempt_index"] or None
            stats["first_success_phase"] = phase
            stats["first_success_turn"] = row.get("turn")
        if not success and row.get("error_count", 0) > 0:
            args = event.get("args") or {}
            errors = args.get("errors") if isinstance(args.get("errors"), list) else []
            text = "\n".join(str(error) for error in errors)
            if _is_sorry_message(text):
                stats["sorry_error_count"] += 1
            if _is_timeout_message(text):
                stats["timeout_count"] += 1
            if _is_infra_message(text):
                stats["infra_error_count"] += 1
            else:
                stats["proof_error_count"] += 1
                stats["proof_error_message_count"] += len(errors)


def collect_events(run_root: Path, experiment_id: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    stats_by_sample: dict[str, dict[str, Any]] = {}
    traces_dir = run_root / "traces"
    if not traces_dir.exists():
        return rows, stats_by_sample
    for trace_path in sorted(traces_dir.glob("*.jsonl")):
        sample_id = trace_path.stem
        stats = stats_by_sample.setdefault(sample_id, _empty_trace_stats())
        with trace_path.open("r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    event = {
                        "kind": "trace_parse_error",
                        "thm_name": sample_id,
                        "args": {"error_type": type(exc).__name__, "message": str(exc)},
                        "result": line,
                        "ok": False,
                    }
                row = _flatten_event(
                    experiment_id=experiment_id,
                    sample_id=sample_id,
                    event=event,
                    trace_line=line_no,
                    trace_path=trace_path,
                )
                rows.append(row)
                _update_trace_stats(stats, row, event)
    return rows, stats_by_sample


def _load_checkpoint_info(run_root: Path, sample_id: str, result: dict[str, Any] | None) -> dict[str, Any]:
    path = run_root / "checkpoints" / f"{sample_id}.json"
    info = {
        "path": str(path) if path.exists() else "",
        "exists": path.exists(),
        "done": None,
        "success": None,
        "blueprint_fully_validated": None,
        "blueprint_node_count": None,
        "contract_error_counts": Counter(),
        "contract_errors": [],
        "blueprint_parse_error": "",
        "node_signal_counts": Counter(),
        "node_lean_errors": [],
        "node_analysis_text": "",
    }
    if not path.exists():
        return info
    try:
        data = _read_json(path)
    except Exception as exc:
        info["blueprint_parse_error"] = f"checkpoint_json_error: {exc}"
        return info

    info["done"] = _as_bool(data.get("done"))
    info["success"] = _as_bool(data.get("success"))
    info["blueprint_fully_validated"] = _as_bool(data.get("blueprint_fully_validated"))

    node_results = data.get("node_results") or {}
    if isinstance(node_results, dict):
        analysis_parts: list[str] = []
        for node_result in node_results.values():
            if not isinstance(node_result, dict):
                continue
            signal = str(node_result.get("signal") or "")
            if signal:
                info["node_signal_counts"][signal] += 1
            errors = node_result.get("lean_errors") or []
            if isinstance(errors, list):
                info["node_lean_errors"].extend(str(error) for error in errors)
            analysis = str(node_result.get("analysis") or "")
            if analysis:
                analysis_parts.append(analysis)
        info["node_analysis_text"] = "\n".join(analysis_parts)

    lean_file = str(data.get("blueprint_lean_file") or "")
    target = str(data.get("blueprint_target") or (result or {}).get("root_theorem") or sample_id)
    if lean_file and _parse_blueprint is not None:
        try:
            blueprint = _parse_blueprint(lean_file, target)
            info["blueprint_node_count"] = len(blueprint.nodes)
            if phase2_contract_errors is not None and phase2_contract_error_counts is not None:
                errors = phase2_contract_errors(blueprint)
                info["contract_errors"] = errors
                info["contract_error_counts"] = Counter(phase2_contract_error_counts(errors))
        except Exception as exc:
            info["blueprint_parse_error"] = str(exc)
    return info


def _result_bool(result: dict[str, Any] | None, key: str) -> bool:
    if not result:
        return False
    return bool(result.get(key))


def _classify_failure(
    *,
    result: dict[str, Any] | None,
    trace_stats: dict[str, Any],
    checkpoint: dict[str, Any],
) -> str:
    if _result_bool(result, "success") or _result_bool(result, "root_proved"):
        return "success"

    contract_counts: Counter[str] = checkpoint["contract_error_counts"]
    if contract_counts.get("missing_sorry_using_placeholder"):
        return "code_missing_sorry_using_placeholder"
    if contract_counts.get("multiple_sorry_using_placeholders"):
        return "code_multiple_sorry_using_placeholders"
    if contract_counts.get("dependency_cycle"):
        return "code_dependency_cycle"

    result_error = str((result or {}).get("error") or "")
    all_node_errors = "\n".join(checkpoint["node_lean_errors"])
    all_analysis = str(checkpoint.get("node_analysis_text") or "")
    combined_text = "\n".join([result_error, all_node_errors, all_analysis])
    signal_counts: Counter[str] = checkpoint["node_signal_counts"]
    final_timeout = _is_timeout_message(result_error) or (
        signal_counts.get("infra_error", 0) > 0 and _is_timeout_message(combined_text)
    )
    final_infra = (
        _is_infra_message(result_error)
        or signal_counts.get("infra_error", 0) > 0
        or (
            trace_stats.get("llm_exhausted_count", 0) > 0
            and not signal_counts
            and not trace_stats.get("proof_error_count", 0)
        )
    )

    if "no attempt produced any @[blueprint]-annotated nodes" in result_error:
        return "model_blueprint_generation_failed_no_nodes"
    if "zero blueprint nodes" in result_error or "not fully validated" in result_error:
        return "code_empty_or_unvalidated_blueprint"
    if checkpoint.get("blueprint_parse_error"):
        return "code_blueprint_parse_error"
    if final_timeout:
        return "infra_timeout"
    if final_infra:
        if signal_counts.get("statement_wrong") or signal_counts.get("formally_negated"):
            return "mixed_infra_and_blueprint_statement_wrong"
        return "infra_api_or_tool_error"
    if signal_counts.get("statement_wrong") or signal_counts.get("formally_negated"):
        return "model_blueprint_statement_wrong_or_formally_negated"
    if _is_sorry_message(combined_text) or trace_stats.get("sorry_error_count", 0) > 0:
        return "model_outputs_sorry_or_incomplete"
    if signal_counts.get("proof_too_hard") or trace_stats.get("proof_error_count", 0) > 0:
        return "model_proof_search_failed"
    if result and result.get("blueprint_success") and not result.get("root_proved"):
        return "model_no_verified_compile"
    if result_error:
        return "model_or_protocol_failure"
    return "unknown_failure"


def build_sample_rows(
    *,
    run_root: Path,
    experiment_id: str,
    results: dict[str, dict[str, Any]],
    trace_stats_by_sample: dict[str, dict[str, Any]],
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    sample_ids = set(results)
    sample_ids.update(trace_stats_by_sample)
    checkpoint_dir = run_root / "checkpoints"
    if checkpoint_dir.exists():
        sample_ids.update(path.stem for path in checkpoint_dir.glob("*.json"))

    rows: list[dict[str, Any]] = []
    for sample_id in sorted(sample_ids):
        result = results.get(sample_id)
        stats = trace_stats_by_sample.get(sample_id, _empty_trace_stats())
        checkpoint = _load_checkpoint_info(run_root, sample_id, result)
        signal_counts: Counter[str] = checkpoint["node_signal_counts"]
        contract_counts: Counter[str] = checkpoint["contract_error_counts"]
        first_ts = stats.get("first_ts")
        last_ts = stats.get("last_ts")
        wall_time = None
        if isinstance(first_ts, (int, float)) and isinstance(last_ts, (int, float)):
            wall_time = round(max(0.0, last_ts - first_ts), 3)

        num_theorems = (result or {}).get("total_nodes")
        if num_theorems in (None, ""):
            num_theorems = checkpoint.get("blueprint_node_count")
        if num_theorems in (None, ""):
            num_theorems = sum(signal_counts.values()) or 0

        phase1_success = bool((result or {}).get("blueprint_success"))
        phase2_success = bool((result or {}).get("root_proved"))
        final_success = bool((result or {}).get("success") or phase2_success)
        row = {
            "experiment_id": experiment_id,
            "sample_id": sample_id,
            "source_id": str((result or {}).get("source_id") or sample_id),
            "split": str((result or {}).get("split") or metrics.get("split") or ""),
            "final_success": final_success,
            "phase0_success": _as_bool((result or {}).get("phase0_success")),
            "phase1_success": phase1_success,
            "phase2_success": phase2_success,
            "blueprint_success": phase1_success,
            "blueprint_reused": bool((result or {}).get("blueprint_reused")),
            "phase1_skipped": bool((result or {}).get("phase1_skipped")),
            "root_theorem": str((result or {}).get("root_theorem") or ""),
            "num_theorems": _as_int(num_theorems) or 0,
            "proved_node_count": _as_int((result or {}).get("proved_node_count")) or 0,
            "proved_ratio": _as_float((result or {}).get("proved_ratio")) or 0.0,
            "num_llm_calls": stats["num_llm_calls"],
            "num_llm_errors": stats["num_llm_errors"],
            "llm_exhausted_count": stats["llm_exhausted_count"],
            "num_tool_calls": stats["num_tool_calls"],
            "num_compile_attempts": stats["phase1_compile_attempts"] + stats["phase2_compile_attempts"],
            "phase1_compile_attempts": stats["phase1_compile_attempts"],
            "phase2_compile_attempts": stats["phase2_compile_attempts"],
            "proof_error_count": stats["proof_error_count"],
            "proof_error_message_count": stats["proof_error_message_count"],
            "infra_error_count": stats["infra_error_count"] + signal_counts.get("infra_error", 0),
            "timeout_count": stats["timeout_count"],
            "total_tokens": stats["tokens"],
            "phase1_tokens": stats["phase1_tokens"],
            "phase2_tokens": stats["phase2_tokens"],
            "wall_time": wall_time if wall_time is not None else "",
            "trace_event_count": stats["event_count"],
            "first_success_attempt": stats["first_success_attempt"] or "",
            "first_success_phase": stats["first_success_phase"],
            "first_success_turn": stats["first_success_turn"] if stats["first_success_turn"] is not None else "",
            "failure_category": _classify_failure(result=result, trace_stats=stats, checkpoint=checkpoint),
            "node_signal_counts": _json_compact(dict(signal_counts)),
            "failed_nodes": _json_compact((result or {}).get("failed_nodes") or []),
            "proved_nodes": _json_compact((result or {}).get("proved_nodes") or []),
            "statement_wrong_count": signal_counts.get("statement_wrong", 0),
            "formally_negated_count": signal_counts.get("formally_negated", 0),
            "proof_too_hard_count": signal_counts.get("proof_too_hard", 0),
            "infra_node_count": signal_counts.get("infra_error", 0),
            "sorry_error_count": stats["sorry_error_count"],
            "contract_error_counts": _json_compact(dict(contract_counts)),
            "blueprint_fully_validated": checkpoint.get("blueprint_fully_validated"),
            "checkpoint_done": checkpoint.get("done"),
            "checkpoint_success": checkpoint.get("success"),
            "result_error": str((result or {}).get("error") or ""),
            "checkpoint_path": str(run_root / "checkpoints" / f"{sample_id}.json"),
            "trace_path": str(run_root / "traces" / f"{sample_id}.jsonl"),
            "blueprint_path": str(run_root / "blueprints" / f"{sample_id}.lean"),
        }
        rows.append(row)
    return rows


def write_events_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to write events.parquet. Install pyarrow or run in "
            "the lean4-czx environment."
        ) from exc

    schema = pa.schema([
        ("experiment_id", pa.string()),
        ("sample_id", pa.string()),
        ("thm_name", pa.string()),
        ("kind", pa.string()),
        ("phase", pa.string()),
        ("turn", pa.int64()),
        ("attempt", pa.int64()),
        ("model", pa.string()),
        ("operation", pa.string()),
        ("tool_name", pa.string()),
        ("call_id", pa.string()),
        ("success", pa.bool_()),
        ("ok", pa.bool_()),
        ("error_type", pa.string()),
        ("error_message", pa.string()),
        ("status_code", pa.int64()),
        ("retry_index", pa.int64()),
        ("max_retries", pa.int64()),
        ("retryable", pa.bool_()),
        ("exhausted", pa.bool_()),
        ("request_id", pa.string()),
        ("finish_reason", pa.string()),
        ("prompt_tokens", pa.int64()),
        ("completion_tokens", pa.int64()),
        ("total_tokens", pa.int64()),
        ("timestamp", pa.string()),
        ("timestamp_unix_s", pa.float64()),
        ("wall_time_s", pa.float64()),
        ("proof_body", pa.string()),
        ("raw_result", pa.string()),
        ("target", pa.string()),
        ("validated", pa.bool_()),
        ("error_count", pa.int64()),
        ("warning_count", pa.int64()),
        ("goal_count", pa.int64()),
        ("query", pa.string()),
        ("k", pa.int64()),
        ("args_json", pa.string()),
        ("extra_json", pa.string()),
    ])
    table = pa.Table.from_pylist(rows, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def write_samples_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SAMPLE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_run(
    run_root: Path,
    *,
    experiment_id: str | None = None,
    timestamp: str | None = None,
    events_path: Path | None = None,
    samples_path: Path | None = None,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    metrics = _load_metrics(run_root)
    experiment_id = experiment_id or infer_experiment_id(run_root, metrics)
    timestamp = timestamp or _now_stamp()
    safe_id = _safe_filename(experiment_id)
    if events_path is None:
        events_path = run_root / f"events_{safe_id}_{timestamp}.parquet"
    if samples_path is None:
        samples_path = run_root / f"samples_{safe_id}_{timestamp}.csv"

    results = _latest_results(run_root)
    event_rows, trace_stats = collect_events(run_root, experiment_id)
    sample_rows = build_sample_rows(
        run_root=run_root,
        experiment_id=experiment_id,
        results=results,
        trace_stats_by_sample=trace_stats,
        metrics=metrics,
    )

    write_events_parquet(events_path, event_rows)
    write_samples_csv(samples_path, sample_rows)
    category_counts = Counter(row["failure_category"] for row in sample_rows)
    return {
        "experiment_id": experiment_id,
        "timestamp": timestamp,
        "events_path": str(events_path),
        "samples_path": str(samples_path),
        "event_count": len(event_rows),
        "sample_count": len(sample_rows),
        "failure_category_counts": dict(sorted(category_counts.items())),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a miniF2F one-pass experiment directory.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--events-path", type=Path, default=None)
    parser.add_argument("--samples-path", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = summarize_run(
        args.run_root,
        experiment_id=args.experiment_id,
        timestamp=args.timestamp,
        events_path=args.events_path,
        samples_path=args.samples_path,
    )
    print(f"[summary] experiment_id={summary['experiment_id']} timestamp={summary['timestamp']}")
    print(f"[summary] events={summary['events_path']} rows={summary['event_count']}")
    print(f"[summary] samples={summary['samples_path']} rows={summary['sample_count']}")
    print(f"[summary] failure_category_counts={summary['failure_category_counts']}")


if __name__ == "__main__":
    main()
# python goedel-architect/experiments/miniF2F_onepass/summarize_run.py --run-root czx_work/goedel-architect/miniF2F_onepass/gpt-5.4-mini
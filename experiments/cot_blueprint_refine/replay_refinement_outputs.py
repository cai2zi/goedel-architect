from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from cot_blueprint_refine.common import (  # noqa: E402
    latest_rows,
    write_json,
    write_jsonl,
)
from cot_blueprint_refine.run_cot_refinement import (  # noqa: E402
    _final_marker_contract_error,
    normalize_refined_output,
)


REPLAY_SCHEMA_VERSION = 1
ALLOWED_ENVELOPE_WARNINGS = {
    "content_outside_final_refined_solution",
    "multiple_final_refined_solution_markers",
}
SAFE_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def _safe_component(value: str, *, name: str) -> str:
    if not SAFE_COMPONENT_RE.fullmatch(value) or value in {".", ".."}:
        raise ValueError(
            f"{name} must be one path component matching "
            "[A-Za-z0-9][A-Za-z0-9._-]*"
        )
    return value


def replay_artifact_dir(experiment_root: Path, variant: str, output_label: str) -> Path:
    variant = _safe_component(variant, name="variant")
    output_label = _safe_component(output_label, name="output label")
    return experiment_root.resolve() / "refinement" / variant / "replays" / output_label


def _response_parts(event: dict[str, Any]) -> tuple[bool, str, str | None, Any]:
    """Return completed, content, finish reason, and serialized response."""
    response = event.get("response")
    completed = isinstance(response, dict)
    content = event.get("assistant_content")
    finish_reason = event.get("finish_reason")
    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            choice = choices[0]
            if finish_reason is None:
                finish_reason = choice.get("finish_reason")
            message = choice.get("message")
            if content is None and isinstance(message, dict):
                content = message.get("content")
    # Synthesized legacy conversations can have the response fields copied to
    # the event without retaining the original response object.
    if bool(event.get("reconstructed_from_refined_predictions")):
        completed = "assistant_content" in event
    return completed, str(content or ""), finish_reason, response


def _attempt_number(event: dict[str, Any], event_index: int) -> int:
    try:
        return int(event.get("attempt"))
    except (TypeError, ValueError):
        return event_index + 1


def _usage(event: dict[str, Any]) -> dict[str, int | None]:
    response = event.get("response")
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}

    def integer(key: str) -> int | None:
        value = usage.get(key)
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return int(value)
        return None

    prompt_tokens = integer("prompt_tokens")
    completion_tokens = integer("completion_tokens")
    total_tokens = integer("total_tokens")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _request_was_sent(event: dict[str, Any]) -> bool:
    # run_cot_refinement persists request only immediately before making the
    # call. Exceptions after that point still represent a real avoided request.
    return isinstance(event.get("request"), dict)


def _sum_latencies(events: Iterable[dict[str, Any]]) -> float | None:
    values = [
        float(event["latency_s"])
        for event in events
        if isinstance(event.get("latency_s"), (int, float))
    ]
    return sum(values) if values else None


def _diagnose_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for event_index, event in enumerate(events):
        completed, content, finish_reason, _response = _response_parts(event)
        if not completed:
            continue
        normalized, error, _think_stripped, _box_count = normalize_refined_output(
            content,
            finish_reason,
        )
        envelope_error = _final_marker_contract_error(content)
        allowed_warning = envelope_error if envelope_error in ALLOWED_ENVELOPE_WARNINGS else ""
        if not error and envelope_error and not allowed_warning:
            error = envelope_error
            normalized = ""
        diagnostics.append({
            "attempt": _attempt_number(event, event_index),
            "event_index": event_index,
            "mode": str(event.get("request_mode") or "unknown"),
            "event_status": str(event.get("status") or ""),
            "finish_reason": finish_reason,
            "normalization_error": error or None,
            "final_envelope_warning": allowed_warning or None,
            "valid": not bool(error),
        })
    return diagnostics


def _valid_candidates(
    events: list[dict[str, Any]],
) -> list[tuple[int, int, dict[str, Any], dict[str, Any]]]:
    candidates: list[tuple[int, int, dict[str, Any], dict[str, Any]]] = []
    for event_index, event in enumerate(events):
        completed, content, finish_reason, response = _response_parts(event)
        if not completed:
            continue
        normalized, error, think_stripped, box_count = normalize_refined_output(
            content,
            finish_reason,
        )
        envelope_error = _final_marker_contract_error(content)
        if envelope_error and envelope_error not in ALLOWED_ENVELOPE_WARNINGS:
            error = error or envelope_error
            normalized = ""
        # Be explicit about the non-negotiable failures even if the legacy
        # normalizer changes later.
        if str(finish_reason or "").strip().lower() == "length":
            error = "finish_reason_length"
            normalized = ""
        if error:
            continue
        candidates.append((
            _attempt_number(event, event_index),
            event_index,
            event,
            {
                "refined_cot": normalized,
                "think_stripped": think_stripped,
                "boxed_answer_count": box_count,
                "finish_reason": finish_reason,
                "raw_content": content,
                "raw_response": response,
                "final_envelope_warning": envelope_error or None,
            },
        ))
    return sorted(candidates, key=lambda item: (item[0], item[1]))


def _avoided_cost(
    events: list[dict[str, Any]],
    selected_event_index: int,
) -> dict[str, int]:
    avoided = [
        event
        for event in events[selected_event_index + 1:]
        if _request_was_sent(event)
    ]
    result = {
        "request_count": len(avoided),
        "requests_with_unknown_total_tokens": 0,
        "prompt_tokens_known": 0,
        "completion_tokens_known": 0,
        "total_tokens_known": 0,
    }
    for event in avoided:
        usage = _usage(event)
        if usage["prompt_tokens"] is not None:
            result["prompt_tokens_known"] += int(usage["prompt_tokens"])
        if usage["completion_tokens"] is not None:
            result["completion_tokens_known"] += int(usage["completion_tokens"])
        if usage["total_tokens"] is None:
            result["requests_with_unknown_total_tokens"] += 1
        else:
            result["total_tokens_known"] += int(usage["total_tokens"])
    return result


def _base_row(
    row_id: str,
    context: dict[str, Any] | None,
    current: dict[str, Any] | None,
    conversation: dict[str, Any] | None,
) -> dict[str, Any]:
    if current is not None:
        return deepcopy(current)
    source = {**(conversation or {}), **(context or {})}
    keys = (
        "source",
        "problem",
        "claimed_answer",
        "root_proved",
        "blueprint_context_status",
        "context_quality",
        "refine_variant",
        "prompt_mode",
        "blueprint_used",
        "source_solution_model_label",
    )
    result = {key: deepcopy(source.get(key)) for key in keys if key in source}
    result["ID"] = row_id
    return result


def _selected_row(
    *,
    base: dict[str, Any],
    conversation_path: Path,
    events: list[dict[str, Any]],
    selected: tuple[int, int, dict[str, Any], dict[str, Any]],
    output_label: str,
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    attempt, event_index, event, normalized = selected
    request = event.get("request") if isinstance(event.get("request"), dict) else {}
    avoided = _avoided_cost(events, event_index)
    original = {
        "status": base.get("status"),
        "error": base.get("error"),
        "attempts": base.get("attempts"),
    }
    result = deepcopy(base)
    result.update({
        "status": "ok",
        "error": None,
        "refined_cot": normalized["refined_cot"],
        "raw_content": normalized["raw_content"],
        "raw_response": normalized["raw_response"],
        "reasoning_content": event.get("assistant_reasoning_content"),
        "finish_reason": normalized["finish_reason"],
        "think_stripped": normalized["think_stripped"],
        "boxed_answer_count": normalized["boxed_answer_count"],
        "final_envelope_warning": normalized["final_envelope_warning"],
        "attempts": attempt,
        "concise_recovery_used": str(event.get("request_mode") or "") == "concise_recovery",
        "latency_s": _sum_latencies(events[:event_index + 1]),
        "conversation_path": str(conversation_path),
        "replay_schema_version": REPLAY_SCHEMA_VERSION,
        "replay_label": output_label,
        "replay_strategy": "earliest_valid_attempt",
        "replay_selected_attempt": attempt,
        "replay_selected_event_index": event_index,
        "replay_selected_mode": str(event.get("request_mode") or "unknown"),
        "replay_original_terminal": original,
        "replay_avoided_request_count": avoided["request_count"],
        "replay_avoided_requests_with_unknown_total_tokens": avoided[
            "requests_with_unknown_total_tokens"
        ],
        "replay_avoided_prompt_tokens_known": avoided["prompt_tokens_known"],
        "replay_avoided_completion_tokens_known": avoided["completion_tokens_known"],
        "replay_avoided_total_tokens_known": avoided["total_tokens_known"],
        "replay_attempt_diagnostics": diagnostics,
    })
    request_fields = {
        "messages": "prompt",
        "max_tokens": "request_max_tokens",
        "extra_body": "request_extra_body",
        "model": "model",
        "base_url": "openai_base_url",
        "temperature": "temperature",
        "timeout_s": "timeout_s",
    }
    for request_key, row_key in request_fields.items():
        if request_key in request:
            result[row_key] = deepcopy(request[request_key])
    return result


def _retained_row(
    *,
    base: dict[str, Any],
    conversation_path: Path | None,
    output_label: str,
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    result = deepcopy(base)
    if "status" not in result:
        result.update({
            "status": "pipeline_error",
            "error": "replay_missing_terminal_result",
            "refined_cot": "",
        })
    if conversation_path is not None:
        result["conversation_path"] = str(conversation_path)
    result.update({
        "replay_schema_version": REPLAY_SCHEMA_VERSION,
        "replay_label": output_label,
        "replay_strategy": "retained_latest_terminal",
        "replay_selected_attempt": None,
        "replay_selected_event_index": None,
        "replay_selected_mode": None,
        "replay_avoided_request_count": 0,
        "replay_avoided_requests_with_unknown_total_tokens": 0,
        "replay_avoided_prompt_tokens_known": 0,
        "replay_avoided_completion_tokens_known": 0,
        "replay_avoided_total_tokens_known": 0,
        "replay_attempt_diagnostics": diagnostics,
    })
    return result


def _load_conversations(
    directory: Path,
) -> tuple[dict[str, tuple[dict[str, Any], Path]], list[dict[str, str]]]:
    conversations: dict[str, tuple[dict[str, Any], Path]] = {}
    errors: list[dict[str, str]] = []
    if not directory.exists():
        return conversations, errors
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("conversation root is not an object")
            row_id = str(payload.get("ID") or "")
            if not row_id:
                raise ValueError("conversation has no ID")
            conversations[row_id] = (payload, path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    return conversations, errors


def replay_refinement_outputs(
    experiment_root: Path,
    variant: str,
    output_label: str,
) -> dict[str, Any]:
    """Derive lenient-envelope outputs from persisted responses, with no live calls."""
    experiment_root = experiment_root.resolve()
    variant = _safe_component(variant, name="variant")
    output_label = _safe_component(output_label, name="output label")
    variant_root = experiment_root / "refinement" / variant
    current_path = variant_root / "refined_predictions.jsonl"
    contexts_path = experiment_root / "blueprint_contexts" / "blueprint_contexts.jsonl"
    conversations_dir = variant_root / "conversations"
    if not current_path.exists() and not contexts_path.exists():
        raise FileNotFoundError(
            f"neither current rows nor contexts exist under {experiment_root}"
        )

    output_dir = replay_artifact_dir(experiment_root, variant, output_label)
    output_path = output_dir / "refined_predictions.jsonl"
    metrics_path = output_dir / "refinement_metrics.json"
    # A label identifies an immutable analysis artifact. Refusing reuse avoids
    # both accidental source overwrite and silent mutation of a prior replay.
    if output_dir.exists():
        raise FileExistsError(f"replay output label already exists: {output_dir}")

    contexts = latest_rows(contexts_path, "ID")
    current_rows = latest_rows(current_path, "ID")
    contexts_by_id = {str(row.get("ID") or ""): row for row in contexts}
    current_by_id = {str(row.get("ID") or ""): row for row in current_rows}
    conversations, conversation_load_errors = _load_conversations(conversations_dir)

    ordered_ids: list[str] = []
    seen: set[str] = set()
    for rows in (contexts, current_rows):
        for row in rows:
            row_id = str(row.get("ID") or "")
            if row_id and row_id not in seen:
                ordered_ids.append(row_id)
                seen.add(row_id)
    for row_id in sorted(conversations):
        if row_id not in seen:
            ordered_ids.append(row_id)
            seen.add(row_id)

    replayed: list[dict[str, Any]] = []
    for row_id in ordered_ids:
        conversation_entry = conversations.get(row_id)
        conversation = conversation_entry[0] if conversation_entry else None
        path = conversation_entry[1] if conversation_entry else None
        events_value = conversation.get("events") if conversation else []
        events = (
            [event for event in events_value if isinstance(event, dict)]
            if isinstance(events_value, list)
            else []
        )
        diagnostics = _diagnose_events(events)
        base = _base_row(
            row_id,
            contexts_by_id.get(row_id),
            current_by_id.get(row_id),
            conversation,
        )
        candidates = _valid_candidates(events)
        if candidates and path is not None:
            row = _selected_row(
                base=base,
                conversation_path=path,
                events=events,
                selected=candidates[0],
                output_label=output_label,
                diagnostics=diagnostics,
            )
        else:
            row = _retained_row(
                base=base,
                conversation_path=path,
                output_label=output_label,
                diagnostics=diagnostics,
            )
        replayed.append(row)

    counts: dict[str, int] = {}
    strategy_counts: dict[str, int] = {}
    mode_counts: dict[str, int] = {}
    warning_counts: dict[str, int] = {}
    for row in replayed:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
        strategy = str(row.get("replay_strategy") or "unknown")
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        mode = row.get("replay_selected_mode")
        if mode:
            mode_counts[str(mode)] = mode_counts.get(str(mode), 0) + 1
        warning = row.get("final_envelope_warning")
        if warning:
            warning_counts[str(warning)] = warning_counts.get(str(warning), 0) + 1

    metrics: dict[str, Any] = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "replay_label": output_label,
        "refine_variant": variant,
        "policy": {
            "selection": "earliest_valid_attempt",
            "allowed_final_envelope_warnings": sorted(ALLOWED_ENVELOPE_WARNINGS),
            "finish_reason_length_accepted": False,
            "unclosed_final_marker_accepted": False,
            "conflicting_boxed_answers_accepted": False,
            "live_calls": False,
        },
        "rows": len(replayed),
        "counts": dict(sorted(counts.items())),
        "strategy_counts": dict(sorted(strategy_counts.items())),
        "selected_mode_counts": dict(sorted(mode_counts.items())),
        "final_envelope_warning_counts": dict(sorted(warning_counts.items())),
        "avoided_request_count": sum(
            int(row["replay_avoided_request_count"]) for row in replayed
        ),
        "avoided_requests_with_unknown_total_tokens": sum(
            int(row["replay_avoided_requests_with_unknown_total_tokens"])
            for row in replayed
        ),
        "avoided_prompt_tokens_known": sum(
            int(row["replay_avoided_prompt_tokens_known"]) for row in replayed
        ),
        "avoided_completion_tokens_known": sum(
            int(row["replay_avoided_completion_tokens_known"]) for row in replayed
        ),
        "avoided_total_tokens_known": sum(
            int(row["replay_avoided_total_tokens_known"]) for row in replayed
        ),
        "conversation_files_loaded": len(conversations),
        "conversation_load_errors": conversation_load_errors,
        "source": {
            "experiment_root": str(experiment_root),
            "current_rows": str(current_path),
            "contexts": str(contexts_path),
            "conversations": str(conversations_dir),
        },
        "output": str(output_path),
        "metrics": str(metrics_path),
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    write_jsonl(output_path, replayed)
    write_json(metrics_path, metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline replay of persisted refinement responses with only outside/multiple "
            "final-envelope violations treated as warnings."
        )
    )
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output-label", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = replay_refinement_outputs(
        args.experiment_root,
        args.variant,
        args.output_label,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

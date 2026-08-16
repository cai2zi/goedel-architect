"""Lossless, secret-free capture of one OpenAI-compatible chat request."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    for method_name in ("model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return to_jsonable(method(mode="json"))
            except TypeError:
                return to_jsonable(method())
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return to_jsonable(attributes)
    return str(value)


def _message_value(message: Any, name: str) -> Any:
    value = getattr(message, name, None)
    if value is not None:
        return value
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict) and extra.get(name) is not None:
        return extra[name]
    if name == "reasoning_content":
        value = getattr(message, "reasoning", None)
        if value is not None:
            return value
        if isinstance(extra, dict):
            return extra.get("reasoning")
    return None


def capture_chat_once(client: Any, stage: str, request: dict[str, Any]) -> tuple[Any | None, dict[str, Any]]:
    """Make exactly one SDK call and return its full replay record."""
    started_at = utc_now()
    started = time.monotonic()
    conversation: dict[str, Any] = {
        "stage": stage,
        "started_at": started_at,
        "finished_at": None,
        "latency_seconds": None,
        "request": to_jsonable(request),
        "raw_response": None,
        "assistant_reasoning_content": None,
        "assistant_content": None,
        "finish_reason": None,
        "usage": None,
        "exception": None,
        "parsed_artifact": {"status": "pending", "path": None, "error": None},
    }
    response = None
    try:
        response = client.chat.completions.create(**request)
        choice = response.choices[0]
        message = choice.message
        conversation.update({
            "raw_response": to_jsonable(response),
            "assistant_reasoning_content": to_jsonable(
                _message_value(message, "reasoning_content")
            ),
            "assistant_content": to_jsonable(_message_value(message, "content")),
            "finish_reason": to_jsonable(getattr(choice, "finish_reason", None)),
            "usage": to_jsonable(getattr(response, "usage", None)),
        })
    except Exception as exc:  # The exception itself is part of the experiment artifact.
        conversation["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        conversation["finished_at"] = utc_now()
        conversation["latency_seconds"] = time.monotonic() - started
    return response, conversation


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def persist_conversation(path: Path, journal_path: Path, conversation: dict[str, Any]) -> None:
    write_json(path, conversation)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with journal_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(conversation), ensure_ascii=False) + "\n")

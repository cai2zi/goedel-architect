from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from omegaconf import DictConfig

from cot_blueprint_refine.common import EXPERIMENT_DIR, append_jsonl, read_jsonl, response_to_json


PROMPT_VERSION = "answer-equivalence-flag-v2"
PROMPTS_DIR = EXPERIMENT_DIR / "prompts"
JUDGE_FLAG_RE = re.compile(r"\[\[JUDGE=([01])\]\]")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def judge_cache_key(
    *,
    problem: str,
    gold: str,
    candidate: str,
    model: str,
) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "problem": problem,
        "gold": gold,
        "candidate": candidate,
        "model": model,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_judge_messages(request: dict[str, Any]) -> list[dict[str, str]]:
    system = (PROMPTS_DIR / "judge_system.md").read_text(encoding="utf-8").strip()
    template = (PROMPTS_DIR / "judge_user.md").read_text(encoding="utf-8")
    user = (
        template.replace("{{problem}}", str(request.get("problem") or ""))
        .replace("{{gold}}", str(request.get("gold") or ""))
        .replace("{{candidate}}", str(request.get("candidate") or ""))
        .strip()
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _message_parts(response: Any) -> tuple[str, str | None, str | None]:
    choice = response.choices[0]
    message = choice.message
    content = str(message.content or "")
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning is None and getattr(message, "model_extra", None):
        reasoning = message.model_extra.get("reasoning_content")
    return content, None if reasoning is None else str(reasoning), getattr(choice, "finish_reason", None)


def parse_judge_flag(content: str) -> tuple[bool, str]:
    matches = JUDGE_FLAG_RE.findall(content)
    if not matches:
        raise ValueError("judge flag missing")
    if len(matches) != 1:
        raise ValueError("judge flag conflict or duplicate")
    return matches[0] == "1", matches[0]


def _error_text(exc: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


def _request_id(value: Any) -> str | None:
    for name in ("request_id", "_request_id", "id"):
        request_id = getattr(value, name, None)
        if request_id:
            return str(request_id)
    headers = getattr(value, "headers", None)
    if headers is not None:
        for name in ("x-request-id", "X-Request-ID"):
            try:
                request_id = headers.get(name)
            except Exception:  # noqa: BLE001
                request_id = None
            if request_id:
                return str(request_id)
    return None


def _http_status(value: Any) -> int | None:
    status = getattr(value, "status_code", None)
    if status is None:
        response = getattr(value, "response", None)
        status = getattr(response, "status_code", None)
    try:
        return None if status is None else int(status)
    except (TypeError, ValueError):
        return None


def _exception_raw_body(exc: BaseException) -> Any:
    # A JSONDecodeError raised while decoding the HTTP response contains the
    # literal response document in `doc`; status/API exceptions usually expose
    # the underlying httpx response instead.
    if isinstance(exc, json.JSONDecodeError):
        return exc.doc
    body = getattr(exc, "body", None)
    if body is not None:
        if isinstance(body, bytes):
            return body.decode("utf-8", errors="replace")
        try:
            json.dumps(body)
            return body
        except (TypeError, ValueError):
            return repr(body)
    response = getattr(exc, "response", None)
    if response is None:
        return None
    try:
        return response.text
    except Exception:  # noqa: BLE001
        content = getattr(response, "content", None)
        if isinstance(content, bytes):
            return content.decode("utf-8", errors="replace")
        return None


def _response_snapshot(response: Any) -> tuple[Any, dict[str, Any] | None]:
    try:
        payload = response_to_json(response)
    except Exception as exc:  # noqa: BLE001
        return repr(response), {"serialization_error": _error_text(exc)}
    try:
        body = response.model_dump_json() if hasattr(response, "model_dump_json") else json.dumps(
            payload, ensure_ascii=False, default=str,
        )
    except Exception:  # noqa: BLE001
        body = json.dumps(payload, ensure_ascii=False, default=str)
    return body, payload


async def _create_judge_response(
    client: AsyncOpenAI,
    kwargs: dict[str, Any],
) -> tuple[Any, Any, str | None, int | None]:
    """Return parsed response plus raw HTTP observability when supported.

    OpenAI's `with_raw_response` path lets us save the response body and request
    id *before* SDK decoding. Test doubles and older clients fall back to the
    ordinary parsed-response API.
    """
    completions = client.chat.completions
    raw_api = getattr(completions, "with_raw_response", None)
    raw_create = getattr(raw_api, "create", None)
    if raw_create is not None:
        http_response = await raw_create(**kwargs)
        raw_body = http_response.text
        request_id = _request_id(http_response)
        status = _http_status(http_response)
        try:
            response = http_response.parse()
        except Exception as exc:  # noqa: BLE001
            # Preserve transport metadata even when SDK response decoding is
            # exactly the operation that failed.
            setattr(exc, "raw_body", raw_body)
            setattr(exc, "request_id", request_id)
            setattr(exc, "status_code", status)
            raise
        return response, raw_body, request_id, status
    response = await completions.create(**kwargs)
    raw_body, _snapshot = _response_snapshot(response)
    return response, raw_body, _request_id(response), _http_status(response)


async def _judge_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    request: dict[str, Any],
    config: DictConfig,
) -> dict[str, Any]:
    messages = build_judge_messages(request)
    start = time.time()
    last_error = ""
    attempt_log: list[dict[str, Any]] = []
    max_retries = int(config.judge.max_retries)
    async with semaphore:
        for attempt in range(1, max_retries + 1):
            attempt_start = time.time()
            attempt_row: dict[str, Any] = {
                "attempt": attempt,
                "started_at": _utc_now(),
                "finished_at": None,
                "latency_s": None,
                "status": "started",
                "error_layer": None,
                "error_type": None,
                "error": None,
                "http_status": None,
                "request_id": None,
                "raw_body": None,
                "raw_content": None,
                "reasoning_content": None,
                "finish_reason": None,
                "raw_response": None,
                "judge_flag": None,
            }
            response = None
            try:
                response, raw_body, request_id, http_status = await _create_judge_response(
                    client,
                    {
                        "model": str(config.judge.model),
                        "messages": messages,
                        "temperature": float(config.judge.temperature),
                        "max_tokens": int(config.judge.max_tokens),
                        "timeout": float(config.judge.timeout_s),
                        # Qwen3/3.5 otherwise spends this deliberately tiny
                        # output budget on its hidden thinking preamble and is
                        # truncated before emitting the machine-readable flag.
                        # This is a chat-template control, not a structured
                        # response format, and is supported by vLLM's OpenAI
                        # compatible endpoint through ``extra_body``.
                        "extra_body": {
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    },
                )
                attempt_row.update({
                    "raw_body": raw_body,
                    "request_id": request_id,
                    "http_status": http_status,
                })
            except Exception as exc:  # noqa: BLE001
                layer = (
                    "api_response_decoding"
                    if isinstance(exc, json.JSONDecodeError)
                    else "api_request"
                )
                response_for_metadata = getattr(exc, "response", None)
                attempt_row.update({
                    "status": "error",
                    "error_layer": layer,
                    "error_type": type(exc).__name__,
                    "error": _error_text(exc),
                    "http_status": _http_status(exc),
                    "request_id": _request_id(exc) or _request_id(response_for_metadata),
                    "raw_body": _exception_raw_body(exc),
                })

            if response is not None:
                snapshot_body, raw_response = _response_snapshot(response)
                if attempt_row["raw_body"] is None:
                    attempt_row["raw_body"] = snapshot_body
                attempt_row["raw_response"] = raw_response
                try:
                    content, reasoning, finish_reason = _message_parts(response)
                    attempt_row.update({
                        "raw_content": content,
                        "reasoning_content": reasoning,
                        "finish_reason": finish_reason,
                    })
                except Exception as exc:  # noqa: BLE001
                    attempt_row.update({
                        "status": "error",
                        "error_layer": "api_response_shape",
                        "error_type": type(exc).__name__,
                        "error": _error_text(exc),
                    })
                else:
                    try:
                        equivalent, flag = parse_judge_flag(content)
                    except Exception as exc:  # noqa: BLE001
                        layer = (
                            "judge_flag_missing"
                            if "missing" in str(exc)
                            else "judge_flag_conflict"
                        )
                        attempt_row.update({
                            "status": "error",
                            "error_layer": layer,
                            "error_type": type(exc).__name__,
                            "error": _error_text(exc),
                        })
                        if attempt < max_retries:
                            messages.extend([
                                {"role": "assistant", "content": content},
                                {"role": "user", "content": (
                                    "Your previous response did not contain exactly one valid decision flag. "
                                    "Reply with exactly [[JUDGE=1]] or [[JUDGE=0]] and nothing else."
                                )},
                            ])
                    else:
                        attempt_row.update({
                            "status": "ok",
                            "error_layer": None,
                            "error_type": None,
                            "error": None,
                            "judge_flag": flag,
                        })

            attempt_row["finished_at"] = _utc_now()
            attempt_row["latency_s"] = time.time() - attempt_start
            attempt_log.append(attempt_row)
            if attempt_row["status"] == "ok":
                content, reasoning, finish_reason = _message_parts(response)
                return {
                    **request,
                    "status": "ok",
                    "equivalent": equivalent,
                    "reason": "",
                    "judge_flag": flag,
                    "error": "",
                    "attempts": attempt,
                    "latency_s": time.time() - start,
                    "cache_hit": False,
                    "model": str(config.judge.model),
                    "openai_base_url": str(config.judge.openai_base_url),
                    "prompt_version": PROMPT_VERSION,
                    "prompt": messages,
                    "raw_content": content,
                    "reasoning_content": reasoning,
                    "finish_reason": finish_reason,
                    "request_id": attempt_row["request_id"],
                    "http_status": attempt_row["http_status"],
                    "raw_body": attempt_row["raw_body"],
                    "raw_response": attempt_row["raw_response"],
                    "attempt_log": attempt_log,
                    "created_at": _utc_now(),
                }
            last_error = str(attempt_row.get("error") or "unknown judge error")
            if attempt >= max_retries:
                break
            delay = min(
                float(config.judge.retry_max_delay_s),
                float(config.judge.retry_base_delay_s) * (2 ** (attempt - 1)),
            )
            await asyncio.sleep(delay)
    last_attempt = attempt_log[-1] if attempt_log else {}
    return {
        **request,
        "status": "error",
        "equivalent": None,
        "reason": "",
        "judge_flag": last_attempt.get("judge_flag"),
        "error": last_error,
        "attempts": len(attempt_log),
        "latency_s": time.time() - start,
        "cache_hit": False,
        "model": str(config.judge.model),
        "openai_base_url": str(config.judge.openai_base_url),
        "prompt_version": PROMPT_VERSION,
        "prompt": messages,
        "error_layer": last_attempt.get("error_layer"),
        "request_id": last_attempt.get("request_id"),
        "http_status": last_attempt.get("http_status"),
        "raw_body": last_attempt.get("raw_body"),
        "raw_content": last_attempt.get("raw_content"),
        "reasoning_content": last_attempt.get("reasoning_content"),
        "finish_reason": last_attempt.get("finish_reason"),
        "raw_response": last_attempt.get("raw_response"),
        "attempt_log": attempt_log,
        "created_at": _utc_now(),
    }


async def judge_equivalences(
    requests: list[dict[str, Any]],
    config: DictConfig,
    output_path: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.touch(exist_ok=True)
    prepared: list[dict[str, Any]] = []
    for request in requests:
        prepared.append({
            **request,
            "cache_key": judge_cache_key(
                problem=str(request.get("problem") or ""),
                gold=str(request.get("gold") or ""),
                candidate=str(request.get("candidate") or ""),
                model=str(config.judge.model),
            ),
        })

    existing_rows = read_jsonl(output_path) if bool(config.resume) else []
    successful_cache = {
        str(row.get("cache_key") or ""): row
        for row in existing_rows
        if row.get("status") == "ok" and row.get("cache_key")
    }
    if not bool(config.resume):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

    results: dict[tuple[str, str], dict[str, Any]] = {}
    pending_by_key: dict[str, dict[str, Any]] = {}
    request_keys_by_cache: dict[str, list[tuple[str, str]]] = {}
    request_metadata: dict[tuple[str, str], dict[str, Any]] = {}
    for request in prepared:
        result_key = (str(request.get("ID") or ""), str(request.get("side") or ""))
        request_metadata[result_key] = request
        cache_key = str(request["cache_key"])
        request_keys_by_cache.setdefault(cache_key, []).append(result_key)
        cached = successful_cache.get(cache_key)
        if cached is not None:
            results[result_key] = {
                **cached,
                "ID": result_key[0],
                "side": result_key[1],
                "variant": str(request.get("variant") or cached.get("variant") or ""),
                "cache_hit": True,
            }
        else:
            pending_by_key.setdefault(cache_key, request)

    if not pending_by_key:
        return results

    client = AsyncOpenAI(
        api_key=str(config.judge.api_key),
        base_url=str(config.judge.openai_base_url).rstrip("/"),
    )
    semaphore = asyncio.Semaphore(int(config.judge.concurrency))
    tasks = [
        asyncio.create_task(_judge_one(client, semaphore, request, config))
        for request in pending_by_key.values()
    ]
    try:
        for task in asyncio.as_completed(tasks):
            result = await task
            append_jsonl(output_path, result)
            cache_key = str(result["cache_key"])
            for result_key in request_keys_by_cache[cache_key]:
                results[result_key] = {
                    **result,
                    "ID": result_key[0],
                    "side": result_key[1],
                    "variant": str(
                        request_metadata[result_key].get("variant")
                        or result.get("variant")
                        or ""
                    ),
                }
    finally:
        await client.close()
    return results

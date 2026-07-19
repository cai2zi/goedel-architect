"""Shared OpenAI-client construction, routing Fireworks- and Mistral-hosted
models to their own base_url instead of OpenAI's.

Fireworks addresses its models as "accounts/<org>/models/<name>" (e.g.
"accounts/fireworks/models/deepseek-v4-flash"), and its API is
OpenAI-compatible for both chat.completions and the Responses API, so a
model_id in that shape is routed there.

Mistral's "Labs" models (e.g. "labs-leanstral-1-5") are addressed with a
"labs-" prefix; Mistral's API is OpenAI-compatible for chat.completions
only — it has no Responses API equivalent, so callers must use
chat.completions.create with this client, not client.responses.create.

Everything else uses the normal OpenAI route unless GOEDEL_OPENAI_BASE_URL or
OPENAI_BASE_URL points it at another OpenAI-compatible endpoint.
"""
from __future__ import annotations

import os
import random
import time
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from tracer import TraceEvent


def make_client(model_id: str, timeout: float | None = None) -> OpenAI:
    if model_id.startswith("accounts/"):
        return OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=os.environ["FIREWORKS_API_KEY"],
            timeout=timeout,
        )
    if model_id.startswith("labs-"):
        return OpenAI(
            base_url="https://api.mistral.ai/v1",
            api_key=os.environ["MISTRAL_API_KEY"],
            timeout=timeout,
        )
    kwargs = {"timeout": timeout}
    base_url = os.environ.get("GOEDEL_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("GOEDEL_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if base_url:
        kwargs["base_url"] = base_url.rstrip("/")
    if api_key:
        kwargs["api_key"] = api_key
    return OpenAI(**kwargs)


def _retry_config() -> tuple[int, float, float, float]:
    return (
        int(os.environ.get("GOEDEL_LLM_MAX_RETRIES", "6")),
        float(os.environ.get("GOEDEL_LLM_RETRY_BASE_DELAY_S", "2")),
        float(os.environ.get("GOEDEL_LLM_RETRY_MAX_DELAY_S", "60")),
        float(os.environ.get("GOEDEL_LLM_RETRY_JITTER_S", "1")),
    )


def _status_code(exc: Exception) -> int | None:
    return getattr(exc, "status_code", None)


def _request_id(exc: Exception) -> str:
    return str(getattr(exc, "request_id", "") or "")


def _is_retryable_llm_error(exc: Exception) -> bool:
    if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        return _status_code(exc) in {408, 409, 429, 500, 502, 503, 504}
    return False


def _sleep_s(*, retry_index: int, base_delay_s: float, max_delay_s: float, jitter_s: float) -> float:
    delay = min(max_delay_s, base_delay_s * (2 ** max(0, retry_index - 1)))
    if jitter_s > 0:
        delay += random.uniform(0, jitter_s)
    return delay


def _emit_llm_error(
    tracer,
    *,
    thm_name: str,
    phase: str,
    model_id: str,
    operation: str,
    retry_index: int,
    max_retries: int,
    sleep_s: float,
    exc: Exception,
    retryable: bool,
    exhausted: bool,
    trace_args: dict[str, Any] | None,
) -> None:
    if tracer is None:
        return
    args = dict(trace_args or {})
    args.update({
        "phase": phase,
        "model": model_id,
        "operation": operation,
        "retry_index": retry_index,
        "max_retries": max_retries,
        "sleep_s": sleep_s,
        "error_type": type(exc).__name__,
        "status_code": _status_code(exc),
        "message": str(exc),
        "request_id": _request_id(exc),
        "retryable": retryable,
        "exhausted": exhausted,
    })
    tracer.emit(TraceEvent(
        kind="llm_error",
        thm_name=thm_name,
        args=args,
        ok=False,
    ))


def chat_completion_with_retry(
    client: OpenAI,
    *,
    tracer=None,
    thm_name: str = "",
    phase: str = "",
    model_id: str,
    operation: str,
    trace_args: dict[str, Any] | None = None,
    **create_kwargs,
):
    max_retries, base_delay_s, max_delay_s, jitter_s = _retry_config()
    for retry_index in range(max_retries + 1):
        try:
            return client.chat.completions.create(**create_kwargs)
        except Exception as exc:
            retryable = _is_retryable_llm_error(exc)
            exhausted = retry_index >= max_retries
            if not retryable or exhausted:
                _emit_llm_error(
                    tracer,
                    thm_name=thm_name,
                    phase=phase,
                    model_id=model_id,
                    operation=operation,
                    retry_index=retry_index,
                    max_retries=max_retries,
                    sleep_s=0.0,
                    exc=exc,
                    retryable=retryable,
                    exhausted=exhausted,
                    trace_args=trace_args,
                )
                raise
            wait_s = _sleep_s(
                retry_index=retry_index + 1,
                base_delay_s=base_delay_s,
                max_delay_s=max_delay_s,
                jitter_s=jitter_s,
            )
            _emit_llm_error(
                tracer,
                thm_name=thm_name,
                phase=phase,
                model_id=model_id,
                operation=operation,
                retry_index=retry_index + 1,
                max_retries=max_retries,
                sleep_s=wait_s,
                exc=exc,
                retryable=True,
                exhausted=False,
                trace_args=trace_args,
            )
            time.sleep(wait_s)

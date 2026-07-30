"""Kimina server backend for the existing Lean compiler contract."""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

import httpx

from lean_compiler import CompilerResult, LeanCompiler


_RETRY_DELAYS_S = (0.5, 1.0, 2.0)


def _compiler_message_text(message: dict[str, Any]) -> str:
    text = str(message.get("data") or message.get("message") or "")
    if "pos" not in message and "endPos" not in message:
        return text
    return json.dumps(message, ensure_ascii=False)


class KiminaLeanCompiler(LeanCompiler):
    """Run Lean snippets through a manually managed Kimina server."""

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        api_key_env: str = "KIMINA_API_KEY",
        timeout_s: int = 300,
        reuse: bool = True,
        debug: bool = False,
        check_concurrency: int = 8,
        *,
        _client: httpx.Client | None = None,
        _sleep: Callable[[float], None] = time.sleep,
    ):
        super().__init__()
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if check_concurrency <= 0:
            raise ValueError("check_concurrency must be positive")

        self.api_url = api_url.rstrip("/")
        self.check_url = self._check_url(self.api_url)
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        self.reuse = reuse
        self.debug = debug
        self.check_concurrency = check_concurrency
        self._semaphore = threading.BoundedSemaphore(check_concurrency)
        self._sleep = _sleep
        self._closed = False

        api_key = os.environ.get(api_key_env, "").strip()
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._client = _client or httpx.Client(
            headers=headers,
            timeout=httpx.Timeout(float(timeout_s)),
        )

    @staticmethod
    def _check_url(api_url: str) -> str:
        if api_url.endswith("/api/check") or api_url.endswith("/check"):
            return api_url
        if api_url.endswith("/api"):
            return f"{api_url}/check"
        return f"{api_url}/api/check"

    def close(self) -> None:
        """Close only this process's HTTP connection pool."""
        if not self._closed:
            self._client.close()
            self._closed = True

    def _run_lean(self, code: str) -> CompilerResult:
        snippet_id = f"goedel-{uuid.uuid4().hex}"
        payload = {
            "snippets": [{"id": snippet_id, "code": code}],
            "timeout": self.timeout_s,
            "debug": self.debug,
            "reuse": self.reuse,
        }
        with self._semaphore:
            return self._request_with_retries(payload)

    def _request_with_retries(self, payload: dict[str, Any]) -> CompilerResult:
        for attempt in range(len(_RETRY_DELAYS_S) + 1):
            try:
                response = self._client.post(self.check_url, json=payload)
            except httpx.HTTPError as exc:
                return self._transport_failure(payload, f"Kimina request failed: {exc}")

            response_data = self._response_json(response)
            raw_output = self._raw_output(payload, response.status_code, response_data)

            if response.status_code == 429:
                if attempt < len(_RETRY_DELAYS_S):
                    self._sleep(_RETRY_DELAYS_S[attempt])
                    continue
                return CompilerResult(
                    success=False,
                    errors=[self._http_error(response.status_code, response_data)],
                    raw_output=raw_output,
                    validated=False,
                )

            if response.status_code >= 500:
                return CompilerResult(
                    success=False,
                    errors=[self._http_error(response.status_code, response_data)],
                    raw_output=raw_output,
                    validated=False,
                )
            if response.status_code >= 400:
                return CompilerResult(
                    success=False,
                    errors=[self._http_error(response.status_code, response_data)],
                    raw_output=raw_output,
                    validated=False,
                )

            result = self._parse_check_response(response_data, raw_output)
            if self._is_no_available_repl(result.errors) and attempt < len(_RETRY_DELAYS_S):
                self._sleep(_RETRY_DELAYS_S[attempt])
                continue
            return result

        raise AssertionError("unreachable")

    def _parse_check_response(self, data: Any, raw_output: str) -> CompilerResult:
        if not isinstance(data, dict):
            return CompilerResult(
                success=False,
                errors=["Kimina returned a non-object JSON response"],
                raw_output=raw_output,
                validated=False,
            )
        results = data.get("results")
        if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
            return CompilerResult(
                success=False,
                errors=["Kimina response must contain exactly one result"],
                raw_output=raw_output,
                validated=False,
            )

        item = results[0]
        top_error = item.get("error")
        if top_error:
            return CompilerResult(
                success=False,
                errors=[str(top_error)],
                raw_output=raw_output,
                validated=False,
            )

        command = item.get("response")
        if not isinstance(command, dict):
            return CompilerResult(
                success=False,
                errors=["Kimina result has neither an error nor a Lean response"],
                raw_output=raw_output,
                validated=False,
            )
        if command.get("message"):
            return CompilerResult(
                success=False,
                errors=[str(command["message"])],
                raw_output=raw_output,
                validated=False,
            )

        goals: list[str] = []
        errors: list[str] = []
        warnings: list[str] = []
        messages = command.get("messages") or []
        if not isinstance(messages, list):
            messages = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            severity = str(message.get("severity") or "").lower()
            text = _compiler_message_text(message)
            if severity == "error":
                errors.append(text)
            elif severity == "warning":
                warnings.append(text)
            elif severity in {"info", "information", "trace"}:
                if "⊢" in text or "goal" in text.lower():
                    goals.append(text)

        sorries = command.get("sorries") or []
        if isinstance(sorries, list) and sorries:
            if not any("declaration uses" in warning and "sorry" in warning for warning in warnings):
                warnings.append("declaration uses `sorry`")
            for sorry in sorries:
                if isinstance(sorry, dict) and sorry.get("goal"):
                    goals.append(str(sorry["goal"]))

        return CompilerResult(
            success=not errors,
            goals=goals,
            errors=errors,
            warnings=warnings,
            raw_output=raw_output,
            validated=True,
        )

    @staticmethod
    def _is_no_available_repl(errors: list[str]) -> bool:
        return any("no available repl" in error.lower() for error in errors)

    @staticmethod
    def _response_json(response: httpx.Response) -> Any:
        try:
            return response.json()
        except (json.JSONDecodeError, ValueError):
            return {"non_json_body": response.text}

    @staticmethod
    def _http_error(status_code: int, data: Any) -> str:
        detail = data.get("detail") if isinstance(data, dict) else data
        return f"Kimina HTTP {status_code}: {detail}"

    @staticmethod
    def _raw_output(payload: dict[str, Any], status_code: int, response: Any) -> str:
        return json.dumps(
            {"request": payload, "http_status": status_code, "response": response},
            ensure_ascii=False,
        )

    @staticmethod
    def _transport_failure(payload: dict[str, Any], error: str) -> CompilerResult:
        raw_output = json.dumps(
            {"request": payload, "response": None, "transport_error": error},
            ensure_ascii=False,
        )
        return CompilerResult(
            success=False,
            errors=[error],
            raw_output=raw_output,
            validated=False,
        )

"""Kimina-only Lean compiler used by the RobustPA pipeline."""
from __future__ import annotations

import json
import os
import random
import re
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from blueprint_text import BLUEPRINT_PROOF_RE, extract_current_node_decl


MATHLIB_HEADER = "import Mathlib\nimport Architect\n\n"
GOEDEL_HEADER = "import Mathlib\nimport Architect\nimport GoedelArch\n\n"
FailureKind = Literal["lean", "infra", "assembly"]
_RETRY_DELAYS_S = (5.0, 10.0, 20.0, 40.0, 60.0)
_FORBIDDEN_COMMANDS = {
    "axiom", "class", "inductive", "instance", "macro", "namespace",
    "notation", "section", "structure", "syntax", "variable",
}


class KiminaInfrastructureError(RuntimeError):
    """Raised by pipeline stages that cannot continue without Kimina."""


class WeightedSemaphore:
    """Atomically reserves multiple slots without partial-acquire deadlocks."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._available = capacity
        self._condition = threading.Condition()

    def acquire(self, weight: int) -> None:
        if not 0 < weight <= self.capacity:
            raise ValueError("weight must be between 1 and capacity")
        with self._condition:
            self._condition.wait_for(lambda: self._available >= weight)
            self._available -= weight

    def release(self, weight: int) -> None:
        with self._condition:
            if self._available + weight > self.capacity:
                raise ValueError("weighted semaphore released too many slots")
            self._available += weight
            self._condition.notify_all()


@dataclass(frozen=True)
class CompileRequest:
    lean_code: str
    allow_sorry: bool = False
    request_id: str = ""


@dataclass
class CompilerResult:
    success: bool
    goals: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_output: str = ""
    failure_kind: FailureKind | None = None

    @property
    def diagnostics(self) -> list[str]:
        return [*self.errors, *self.goals]

    @property
    def has_sorry(self) -> bool:
        return any(
            "declaration uses" in warning and ("sorry" in warning or "admit" in warning)
            for warning in self.warnings
        )


def _mask_comments_and_strings(code: str) -> str:
    out: list[str] = []
    i = 0
    block_depth = 0
    in_line_comment = False
    in_string = False
    while i < len(code):
        ch = code[i]
        nxt = code[i + 1] if i + 1 < len(code) else ""
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
                out.append(ch)
            else:
                out.append(" ")
            i += 1
            continue
        if block_depth:
            if ch == "/" and nxt == "-":
                block_depth += 1
                out.extend("  ")
                i += 2
                continue
            if ch == "-" and nxt == "/":
                block_depth -= 1
                out.extend("  ")
                i += 2
                continue
            out.append("\n" if ch == "\n" else " ")
            i += 1
            continue
        if in_string:
            if ch == "\\" and nxt:
                out.extend("  ")
                i += 2
                continue
            if ch == '"':
                in_string = False
            out.append("\n" if ch == "\n" else " ")
            i += 1
            continue
        if ch == "-" and nxt == "-":
            in_line_comment = True
            out.extend("  ")
            i += 2
            continue
        if ch == "/" and nxt == "-":
            block_depth = 1
            out.extend("  ")
            i += 2
            continue
        if ch == '"':
            in_string = True
            out.append(" ")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _command_words(line: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_']*", line)


def _find_forbidden_construct(code: str) -> str | None:
    masked = _mask_comments_and_strings(code)
    for line in masked.splitlines():
        words = _command_words(line)
        if not words:
            continue
        if words[:2] == ["noncomputable", "section"]:
            return "noncomputable section"
        if words[:2] == ["partial", "def"]:
            return "partial def"
        if words[:2] == ["local", "notation"]:
            return "local notation"
        if words[0] in _FORBIDDEN_COMMANDS:
            return words[0]
    if re.search(r"\bnative_decide\b", masked):
        return "native_decide"
    return None


def assemble_node_attempt(
    node_decl: str,
    parent_lemma_decls: str,
    proof_body: str,
    header: str,
) -> str:
    if not header.strip():
        raise ValueError("node compilation requires an explicit blueprint header")
    decl_text = extract_current_node_decl(node_decl)
    decl_text, replacements = BLUEPRINT_PROOF_RE.subn(
        f":= {proof_body}", decl_text, count=1,
    )
    if replacements != 1:
        raise ValueError(
            "proof node declaration must contain exactly one "
            "`:= by sorry_using [...]` placeholder"
        )
    parts = [header.rstrip()]
    if parent_lemma_decls.strip():
        parts.append(parent_lemma_decls.strip())
    parts.append(decl_text.strip())
    return "\n\n".join(parts) + "\n"


def _message_text(message: dict[str, Any]) -> str:
    """Keep source positions and every other diagnostic field intact."""
    return json.dumps(message, ensure_ascii=False, sort_keys=True)


class KiminaLeanCompiler:
    """Compile complete files and assembled node attempts through Kimina."""

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        api_key_env: str = "KIMINA_API_KEY",
        timeout_s: int = 300,
        reuse: bool = True,
        debug: bool = False,
        max_inflight_snippets: int = 128,
        batch_size: int = 8,
        retry_delays_s: Sequence[float] = _RETRY_DELAYS_S,
        retry_jitter_s: float = 1.0,
        *,
        _client: httpx.Client | None = None,
        _sleep: Callable[[float], None] = time.sleep,
        _random: Callable[[], float] = random.random,
    ):
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if max_inflight_snippets <= 0:
            raise ValueError("max_inflight_snippets must be positive")
        if not 0 < batch_size <= max_inflight_snippets:
            raise ValueError("batch_size must be between 1 and max_inflight_snippets")
        if retry_jitter_s < 0 or any(delay < 0 for delay in retry_delays_s):
            raise ValueError("retry delays and jitter must be non-negative")
        self.api_url = api_url.rstrip("/")
        self.check_url = self._check_url(self.api_url)
        self.timeout_s = timeout_s
        self.reuse = reuse
        self.debug = debug
        self.max_inflight_snippets = max_inflight_snippets
        self.batch_size = batch_size
        self.retry_delays_s = tuple(float(delay) for delay in retry_delays_s)
        self.retry_jitter_s = float(retry_jitter_s)
        self._snippet_slots = WeightedSemaphore(max_inflight_snippets)
        self._batch_executor = ThreadPoolExecutor(
            max_workers=max_inflight_snippets,
            thread_name_prefix="kimina-batch",
        )
        self._sleep = _sleep
        self._random = _random
        self._stats_lock = threading.Lock()
        self._stats: Counter[str] = Counter()
        self._batch_sizes: Counter[int] = Counter()
        self._inflight_snippets = 0
        self._closed = False
        api_key = os.environ.get(api_key_env, "").strip()
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._client = _client or httpx.Client(
            headers=headers, timeout=httpx.Timeout(float(timeout_s)),
        )

    @staticmethod
    def _check_url(api_url: str) -> str:
        if api_url.endswith("/api/check") or api_url.endswith("/check"):
            return api_url
        if api_url.endswith("/api"):
            return f"{api_url}/check"
        return f"{api_url}/api/check"

    def close(self) -> None:
        if not self._closed:
            self._batch_executor.shutdown(wait=True)
            self._client.close()
            self._closed = True

    def stats(self) -> dict[str, Any]:
        with self._stats_lock:
            return {
                "submitted_batches": self._stats["submitted_batches"],
                "submitted_snippets": self._stats["submitted_snippets"],
                "http_requests": self._stats["http_requests"],
                "http_429": self._stats["http_429"],
                "retries": self._stats["retries"],
                "no_available_repl": self._stats["no_available_repl"],
                "max_inflight_snippets": self.max_inflight_snippets,
                "peak_inflight_snippets": self._stats["peak_inflight_snippets"],
                "current_inflight_snippets": self._inflight_snippets,
                "batch_size": self.batch_size,
                "batch_size_distribution": {
                    str(size): count for size, count in sorted(self._batch_sizes.items())
                },
            }

    def check(self, lean_code: str, allow_sorry: bool = False) -> CompilerResult:
        return self.check_many([CompileRequest(lean_code, allow_sorry)])[0]

    def check_node(
        self,
        proof_body: str,
        *,
        node_decl: str,
        parent_lemma_decls: str,
        header: str,
    ) -> CompilerResult:
        try:
            code = assemble_node_attempt(
                node_decl, parent_lemma_decls, proof_body, header,
            )
        except ValueError as exc:
            message = f"Node assembly rejected: {exc}"
            return CompilerResult(False, errors=[message], raw_output=message, failure_kind="assembly")
        return self.check(code)

    def check_blueprint(self, lean_code: str, target_name: str) -> CompilerResult:
        if "import GoedelArch" not in lean_code:
            code = lean_code.replace(
                "import Architect", "import Architect\nimport GoedelArch", 1,
            )
            if "import GoedelArch" not in code:
                code = GOEDEL_HEADER + lean_code
        else:
            code = lean_code
        code = code.rstrip() + f"\n\n#validate_blueprint {target_name}\n"
        return self.check(code, allow_sorry=True)

    def check_many(
        self,
        requests: Sequence[CompileRequest],
        *,
        batch_concurrency: int = 1,
    ) -> list[CompilerResult]:
        if not requests:
            return []
        if batch_concurrency <= 0:
            raise ValueError("batch_concurrency must be positive")
        if self._closed:
            raise RuntimeError("KiminaLeanCompiler is closed")
        results: list[CompilerResult | None] = [None] * len(requests)
        pending: list[tuple[int, CompileRequest, str]] = []
        seen_ids: set[str] = set()
        for index, request in enumerate(requests):
            forbidden = _find_forbidden_construct(request.lean_code)
            if forbidden:
                message = f"Safeguard rejected: forbidden construct `{forbidden}` is not allowed."
                results[index] = CompilerResult(
                    False, errors=[message], raw_output=message, failure_kind="assembly",
                )
                continue
            request_id = request.request_id or f"goedel-{uuid.uuid4().hex}"
            while request_id in seen_ids:
                request_id = f"{request_id}-{uuid.uuid4().hex[:8]}"
            seen_ids.add(request_id)
            pending.append((index, request, request_id))
        chunks = [
            pending[start:start + self.batch_size]
            for start in range(0, len(pending), self.batch_size)
        ]
        for start in range(0, len(chunks), batch_concurrency):
            group = chunks[start:start + batch_concurrency]
            futures = [
                self._batch_executor.submit(self._send_chunk, chunk)
                for chunk in group
            ]
            for chunk, future in zip(group, futures, strict=True):
                batch = future.result()
                for index, request, request_id in chunk:
                    result = self._finalize_result(batch, request, request_id)
                    results[index] = result
        return [result for result in results if result is not None]

    def _send_chunk(
        self,
        chunk: Sequence[tuple[int, CompileRequest, str]],
    ) -> dict[str, CompilerResult]:
        payload = {
            "snippets": [
                {"id": request_id, "code": request.lean_code}
                for _, request, request_id in chunk
            ],
            "timeout": self.timeout_s,
            "debug": self.debug,
            "reuse": self.reuse,
        }
        weight = len(chunk)
        self._snippet_slots.acquire(weight)
        with self._stats_lock:
            self._stats["submitted_batches"] += 1
            self._stats["submitted_snippets"] += weight
            self._batch_sizes[weight] += 1
            self._inflight_snippets += weight
            self._stats["peak_inflight_snippets"] = max(
                self._stats["peak_inflight_snippets"], self._inflight_snippets,
            )
        try:
            return self._request_with_retries(payload)
        finally:
            with self._stats_lock:
                self._inflight_snippets -= weight
            self._snippet_slots.release(weight)

    @staticmethod
    def _finalize_result(
        batch: dict[str, CompilerResult],
        request: CompileRequest,
        request_id: str,
    ) -> CompilerResult:
        result = batch.get(request_id)
        if result is None:
            return CompilerResult(
                False,
                errors=[f"Kimina response omitted snippet `{request_id}`"],
                failure_kind="infra",
            )
        if result.success and result.has_sorry and not request.allow_sorry:
            return CompilerResult(
                False,
                goals=result.goals,
                errors=result.errors + [
                    "Proof contains `sorry` - not a complete proof.",
                    *result.warnings,
                ],
                warnings=result.warnings,
                raw_output=result.raw_output,
                failure_kind="lean",
            )
        return result

    def _request_with_retries(self, payload: dict[str, Any]) -> dict[str, CompilerResult]:
        ids = [str(item["id"]) for item in payload["snippets"]]
        for attempt in range(len(self.retry_delays_s) + 1):
            with self._stats_lock:
                self._stats["http_requests"] += 1
            try:
                response = self._client.post(self.check_url, json=payload)
            except httpx.HTTPError as exc:
                return self._batch_failure(ids, payload, f"Kimina request failed: {exc}")
            data = self._response_json(response)
            raw_output = self._raw_output(payload, response.status_code, data)
            if response.status_code == 429:
                with self._stats_lock:
                    self._stats["http_429"] += 1
                if attempt < len(self.retry_delays_s):
                    with self._stats_lock:
                        self._stats["retries"] += 1
                    self._sleep(self._retry_delay(response, attempt))
                    continue
            if response.status_code >= 400:
                return self._batch_failure(
                    ids, payload, self._http_error(response.status_code, data), raw_output,
                )
            parsed = self._parse_check_response(data, raw_output)
            no_available = any(
                self._is_no_available_repl(result.errors) for result in parsed.values()
            )
            if no_available:
                with self._stats_lock:
                    self._stats["no_available_repl"] += 1
                if attempt < len(self.retry_delays_s):
                    with self._stats_lock:
                        self._stats["retries"] += 1
                    self._sleep(self._retry_delay(response, attempt))
                    continue
            return parsed
        raise AssertionError("unreachable")

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        retry_after = 0.0
        value = response.headers.get("Retry-After", "").strip()
        try:
            retry_after = max(float(value), 0.0) if value else 0.0
        except ValueError:
            retry_after = 0.0
        return max(self.retry_delays_s[attempt], retry_after) + (
            self._random() * self.retry_jitter_s
        )

    def _parse_check_response(self, data: Any, raw_output: str) -> dict[str, CompilerResult]:
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            return {}
        parsed: dict[str, CompilerResult] = {}
        for item in data["results"]:
            if not isinstance(item, dict) or "id" not in item:
                continue
            request_id = str(item["id"])
            top_error = item.get("error")
            if top_error:
                parsed[request_id] = CompilerResult(
                    False, errors=[str(top_error)], raw_output=raw_output, failure_kind="infra",
                )
                continue
            command = item.get("response")
            if not isinstance(command, dict) or command.get("message"):
                message = command.get("message") if isinstance(command, dict) else "missing Lean response"
                parsed[request_id] = CompilerResult(
                    False, errors=[str(message)], raw_output=raw_output, failure_kind="infra",
                )
                continue
            goals: list[str] = []
            errors: list[str] = []
            warnings: list[str] = []
            messages = command.get("messages") or []
            if isinstance(messages, list):
                for message in messages:
                    if not isinstance(message, dict):
                        continue
                    severity = str(message.get("severity") or "").lower()
                    text = _message_text(message)
                    if severity == "error":
                        errors.append(text)
                    elif severity == "warning":
                        warnings.append(text)
                    elif severity in {"info", "information", "trace"}:
                        data_text = str(message.get("data") or message.get("message") or "")
                        if "⊢" in data_text or "goal" in data_text.lower():
                            goals.append(text)
            sorries = command.get("sorries") or []
            if isinstance(sorries, list) and sorries:
                warnings.append("declaration uses `sorry`")
                for sorry in sorries:
                    if isinstance(sorry, dict) and sorry.get("goal"):
                        goals.append(str(sorry["goal"]))
            parsed[request_id] = CompilerResult(
                not errors, goals, errors, warnings, raw_output,
                "lean" if errors else None,
            )
        return parsed

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
    def _batch_failure(
        ids: list[str],
        payload: dict[str, Any],
        error: str,
        raw_output: str | None = None,
    ) -> dict[str, CompilerResult]:
        raw = raw_output or json.dumps(
            {"request": payload, "response": None, "transport_error": error},
            ensure_ascii=False,
        )
        return {
            request_id: CompilerResult(
                False, errors=[error], raw_output=raw, failure_kind="infra",
            )
            for request_id in ids
        }

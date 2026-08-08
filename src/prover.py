"""Phase 2 per-node prover with a bounded, concurrent tool protocol."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from blueprint_text import BLUEPRINT_PROOF_RE, extract_current_node_decl, lemma_to_theorem
from goedel_prompts import load, render
from kimina_lean_compiler import (
    CompileRequest,
    CompilerResult,
    KiminaLeanCompiler,
    assemble_node_attempt,
)
from llm_client import chat_completion_with_retry, make_client
from mathlib_retrieval import MathlibRetrieval
from tracer import NullTracer, TraceEvent


PROVER_SYSTEM_PROMPT = load("prover_system")
PROVER_USER_TEMPLATE = load("prover_user")
DEFAULT_NODE_MAX_PROVE_TURNS = 8
DEFAULT_MAX_TOOL_CALLS_PER_TURN = 3
DEFAULT_TOOL_FEEDBACK_MAX_CHARS = 8192


def _max_tokens() -> int:
    return int(os.environ.get("GOEDEL_PROVER_MAX_TOKENS", "64000"))


def _length_retry_max_tokens() -> int:
    return int(os.environ.get("GOEDEL_PROVER_LENGTH_RETRY_MAX_TOKENS", str(_max_tokens())))


def _tool_feedback_max_chars() -> int:
    return max(
        512,
        int(os.environ.get(
            "GOEDEL_TOOL_FEEDBACK_MAX_CHARS",
            str(DEFAULT_TOOL_FEEDBACK_MAX_CHARS),
        )),
    )


def _stable_deduplicate_lines(text: str) -> tuple[str, int]:
    """Collapse exact repeated lines while preserving first-seen order."""
    lines = text.splitlines()
    counts = Counter(lines)
    seen: set[str] = set()
    compacted: list[str] = []
    removed = 0
    for line in lines:
        if line in seen:
            removed += 1
            continue
        seen.add(line)
        compacted.append(line)
        repeats = counts[line] - 1
        if repeats > 0 and line.strip():
            compacted.append(f"[previous line repeated {repeats} additional times]")
    return "\n".join(compacted), removed


def _head_tail_limit(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    digest = hashlib.sha256(text.encode()).hexdigest()
    marker = (
        "\n... [tool feedback truncated; "
        f"original_chars={len(text)} sha256={digest}] ...\n"
    )
    available = max_chars - len(marker)
    if available <= 0:
        return marker[:max_chars], True
    head_chars = available // 2
    tail_chars = available - head_chars
    return text[:head_chars] + marker + text[-tail_chars:], True


def _tool_feedback_for_model(
    output: str,
    *,
    max_chars: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Project full tool output into bounded model history.

    The caller keeps ``output`` unchanged for trace/cache.  Only the text sent
    back to the model is deduplicated and bounded.
    """
    compacted, duplicate_lines_removed = _stable_deduplicate_lines(output)
    sent, truncated = _head_tail_limit(
        compacted,
        _tool_feedback_max_chars() if max_chars is None else max(512, max_chars),
    )
    return sent, {
        "original_chars": len(output),
        "deduplicated_chars": len(compacted),
        "sent_chars": len(sent),
        "duplicate_lines_removed": duplicate_lines_removed,
        "truncated": truncated,
        "full_sha256": hashlib.sha256(output.encode()).hexdigest(),
    }


def _stable_deduplicate_diagnostics(diagnostics: list[str]) -> list[str]:
    counts = Counter(diagnostics)
    seen: set[str] = set()
    compacted: list[str] = []
    for diagnostic in diagnostics:
        if diagnostic in seen:
            continue
        seen.add(diagnostic)
        compacted.append(diagnostic)
        repeats = counts[diagnostic] - 1
        if repeats > 0:
            compacted.append(
                f"[previous diagnostic repeated {repeats} additional times]"
            )
    return compacted


def _retain_latest_assistant_turn(
    messages: list[dict[str, Any]],
    base_messages: tuple[dict[str, Any], ...],
) -> None:
    """Retain the immutable prompt and the most recent legal assistant turn."""
    base_len = len(base_messages)
    latest_assistant = next(
        (
            index
            for index in range(len(messages) - 1, base_len - 1, -1)
            if messages[index].get("role") == "assistant"
        ),
        None,
    )
    if latest_assistant is None:
        messages[:] = [dict(message) for message in base_messages]
        return
    latest_turn = messages[latest_assistant:]
    messages[:] = [dict(message) for message in base_messages] + latest_turn


LEAN_COMPILE_TOOL = {
    "type": "function",
    "function": {
        "name": "lean_compile",
        "description": "Compile a proof body in the current node declaration and dependency context.",
        "parameters": {
            "type": "object",
            "properties": {
                "proof_body": {
                    "type": "string",
                    "description": "A Lean proof beginning with `by`. Do not include the theorem declaration.",
                },
            },
            "required": ["proof_body"],
        },
    },
}

STEP_LEAN_COMPILE_TOOL = {
    "type": "function",
    "function": {
        "name": "step_lean_compile",
        "description": (
            "Exploratory compile of a complete standalone Lean file. Include all imports and "
            "the complete theorem yourself. Success does not solve the current node."
        ),
        "parameters": {
            "type": "object",
            "properties": {"lean_code": {"type": "string"}},
            "required": ["lean_code"],
        },
    },
}

MATHLIB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "mathlib_search",
        "description": "Semantic search over Mathlib for general library lemmas.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    },
}

NORMAL_TOOLS = [LEAN_COMPILE_TOOL, STEP_LEAN_COMPILE_TOOL, MATHLIB_SEARCH_TOOL]

SYSTEM_SUFFIX = """
## Tool-first workflow

Use lean_compile for the current node, step_lean_compile for isolated experiments,
and mathlib_search when you do not know a Mathlib lemma name. A successful
lean_compile call is the only way to solve the node. step_lean_compile must contain
a complete file including imports and never solves the node. Submit at most
{{max_tool_calls}} tool calls per turn; excess and duplicate calls are discarded.
The final turn exposes only lean_compile, so use earlier turns for exploration.
"""


class ProofSignal(str, Enum):
    SOLVED = "solved"
    PROOF_TOO_HARD = "proof_too_hard"
    BLOCKED_BY_DEPENDENCY = "blocked_by_dependency"
    FORMALLY_NEGATED = "formally_negated"
    INFRA_ERROR = "infra_error"
    PROTOCOL_ERROR = "protocol_error"


@dataclass
class ProverResult:
    signal: ProofSignal
    proof_body: str = ""
    lean_errors: list[str] = field(default_factory=list)

    def diagnosis_block(self, node_name: str = "") -> str:
        errors = "\n".join(self.lean_errors) or "(none)"
        proof = self.proof_body or "(none)"
        return (
            "/- Diagnosis\n"
            f"## Signal\n{self.signal.value}\n\n"
            "## Proof body\n```lean\n"
            f"{proof}\n```\n"
            "## Lean errors\n"
            f"{errors}\n-/"
        )


@dataclass(frozen=True)
class _AcceptedCall:
    index: int
    call_id: str
    name: str
    args: dict[str, Any]
    call_hash: str
    raw: Any

    @property
    def original_index(self) -> int:
        return self.index

    @property
    def call(self):
        return self.raw


@dataclass
class _ToolOutcome:
    output: str
    ok: bool = False
    proof_body: str = ""
    errors: list[str] = field(default_factory=list)
    failure_kind: str | None = None
    timings: dict[str, Any] = field(default_factory=dict)


@dataclass
class _TurnOutcome:
    had_calls: bool = False
    solved_proof: str = ""
    last_proof: str = ""
    last_errors: list[str] = field(default_factory=list)
    last_failure_kind: str | None = None


@dataclass(frozen=True)
class _ExecutedCall:
    call: _AcceptedCall
    outcome: _ToolOutcome
    cache_hit: bool


class GoedelProver:
    def __init__(
        self,
        model_id: str = "labs-leanstral-1-5",
        retrieval: MathlibRetrieval | None = None,
        tracer=None,
        api_timeout_s: float | None = 120.0,
        max_prove_turns: int | None = None,
        max_negation_probe_turns: int = 1,
        max_tool_calls_per_turn: int = DEFAULT_MAX_TOOL_CALLS_PER_TURN,
    ):
        if max_tool_calls_per_turn <= 0:
            raise ValueError("max_tool_calls_per_turn must be positive")
        self.model_id = model_id
        self.client = make_client(model_id, timeout=api_timeout_s)
        self.retrieval = retrieval or MathlibRetrieval()
        self.tracer = tracer or NullTracer()
        self.max_prove_turns = max_prove_turns or DEFAULT_NODE_MAX_PROVE_TURNS
        if max_negation_probe_turns < 0:
            raise ValueError("max_negation_probe_turns must be non-negative")
        self.max_negation_probe_turns = max_negation_probe_turns
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self._tool_cache: dict[str, _ToolOutcome] = {}
        self._completion_max_tokens = _max_tokens()
        self._length_retry_max_tokens = max(
            self._completion_max_tokens,
            _length_retry_max_tokens(),
        )

    def prove_node(
        self,
        compiler: KiminaLeanCompiler,
        node_name: str,
        node_stmt: str,
        user_prompt: str,
        parent_lemma_decls: str,
        header: str,
    ) -> ProverResult:
        started = time.monotonic()
        self.tracer.emit(TraceEvent(
            kind="theorem_start", thm_name=node_name, args={"thm_stmt": node_stmt},
        ))
        system_prompt = (
            PROVER_SYSTEM_PROMPT.strip()
            + "\n\n"
            + render(SYSTEM_SUFFIX, max_tool_calls=self.max_tool_calls_per_turn).strip()
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        base_messages = tuple(dict(message) for message in messages)
        last_proof = ""
        last_errors: list[str] = []
        last_failure_kind: str | None = None
        protocol_error = False
        final_protocol_error = False

        for turn in range(1, self.max_prove_turns + 1):
            final_turn = turn == self.max_prove_turns
            tools = [LEAN_COMPILE_TOOL] if final_turn else NORMAL_TOOLS
            response = self._chat(
                messages, node_name, turn, "prove", tools,
                operation="prove_node_final" if final_turn else "prove_node",
            )
            outcome = self._process_turn(
                response=response,
                messages=messages,
                compiler=compiler,
                node_name=node_name,
                node_decl=node_stmt,
                parent_lemma_decls=parent_lemma_decls,
                header=header,
                stage="prove",
                turn=turn,
                limit=1 if final_turn else self.max_tool_calls_per_turn,
                allowed_names={"lean_compile"} if final_turn else {
                    "lean_compile", "step_lean_compile", "mathlib_search",
                },
            )
            if outcome.solved_proof:
                result = ProverResult(ProofSignal.SOLVED, outcome.solved_proof)
                return self._finish(node_name, started, result)
            if outcome.last_proof:
                last_proof = outcome.last_proof
                last_errors = outcome.last_errors
                last_failure_kind = outcome.last_failure_kind
            _retain_latest_assistant_turn(messages, base_messages)
            if not outcome.had_calls:
                protocol_error = True
                if final_turn:
                    final_protocol_error = True
                    break

        if last_failure_kind == "infra":
            return self._finish(
                node_name, started,
                ProverResult(ProofSignal.INFRA_ERROR, last_proof, last_errors),
            )
        if last_failure_kind == "assembly":
            return self._finish(
                node_name, started,
                ProverResult(ProofSignal.PROTOCOL_ERROR, last_proof, last_errors),
            )

        negated = self._probe_negation(
            compiler, node_name, node_stmt, parent_lemma_decls, header,
        )
        if negated is not None:
            return self._finish(node_name, started, negated)

        signal = (
            ProofSignal.PROTOCOL_ERROR
            if final_protocol_error or (protocol_error and not last_proof)
            else ProofSignal.PROOF_TOO_HARD
        )
        return self._finish(node_name, started, ProverResult(signal, last_proof, last_errors))

    def _finish(self, node_name: str, started: float, result: ProverResult) -> ProverResult:
        self.tracer.emit(TraceEvent(
            kind="node_finished",
            thm_name=node_name,
            ok=result.signal == ProofSignal.SOLVED,
            args={
                "wall_time_s": time.monotonic() - started,
                "signal": result.signal.value,
                "proof": result.proof_body,
                "lean_errors": result.lean_errors,
            },
        ))
        return result

    def _chat(
        self,
        messages: list[dict[str, Any]],
        node_name: str,
        turn: int,
        stage: str,
        tools: list[dict[str, Any]],
        operation: str,
    ):
        def request(max_tokens: int, *, adaptive_retry: bool):
            return chat_completion_with_retry(
                self.client,
                tracer=self.tracer,
                thm_name=node_name,
                phase="phase2",
                model_id=self.model_id,
                operation=operation,
                trace_args={
                    "stage": stage,
                    "turn": turn,
                    "adaptive_length_retry": adaptive_retry,
                    "max_completion_tokens": max_tokens,
                },
                model=self.model_id,
                messages=messages,
                tools=tools,
                max_completion_tokens=max_tokens,
                parallel_tool_calls=len(tools) > 1,
                tool_choice="required",
            )

        requested_tokens = self._completion_max_tokens
        response = request(requested_tokens, adaptive_retry=False)
        choice = response.choices[0]
        if (
            getattr(choice, "finish_reason", None) == "length"
            and self._length_retry_max_tokens > requested_tokens
        ):
            upgraded_tokens = self._length_retry_max_tokens
            self.tracer.emit(TraceEvent(
                kind="llm_length_retry",
                thm_name=node_name,
                turn=turn,
                args={
                    "phase": "phase2",
                    "operation": operation,
                    "stage": stage,
                    "previous_max_completion_tokens": requested_tokens,
                    "max_completion_tokens": upgraded_tokens,
                },
            ))
            # A GoedelProver instance handles exactly one node. Once that node
            # demonstrates that 4K is insufficient, retain the upgraded limit
            # for its remaining turns without affecting any other node.
            self._completion_max_tokens = upgraded_tokens
            response = request(upgraded_tokens, adaptive_retry=True)
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.tracer.emit(TraceEvent(
                kind="llm_usage",
                thm_name=node_name,
                turn=turn,
                args={
                    "stage": stage,
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                },
            ))
        return response

    def _process_turn(
        self,
        *,
        response,
        messages: list[dict[str, Any]],
        compiler: KiminaLeanCompiler,
        node_name: str,
        node_decl: str,
        parent_lemma_decls: str,
        header: str,
        stage: str,
        turn: int,
        limit: int,
        allowed_names: set[str],
    ) -> _TurnOutcome:
        msg = response.choices[0].message
        accepted, dropped = self._select_calls(
            msg.tool_calls or [], stage, node_decl, parent_lemma_decls, header,
            limit, allowed_names,
        )
        if dropped:
            self.tracer.emit(TraceEvent(
                kind="tool_calls_dropped",
                thm_name=node_name,
                turn=turn,
                args={"stage": stage, "count": len(dropped), "calls": dropped},
            ))
        if msg.content:
            self.tracer.emit(TraceEvent(
                kind="model_text", thm_name=node_name, turn=turn,
                args={"stage": stage}, result=msg.content,
            ))
        if not accepted:
            if msg.content:
                messages.append({"role": "assistant", "content": msg.content})
            return _TurnOutcome()

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [call.raw.model_dump() for call in accepted],
        })
        executions = self._execute_calls(
            accepted, compiler, node_decl, parent_lemma_decls, header,
            node_name=node_name, turn=turn, stage=stage,
        )
        turn_outcome = _TurnOutcome(had_calls=True)
        successful: list[tuple[int, str]] = []
        for execution in executions:
            call = execution.call
            outcome = execution.outcome
            cache_hit = execution.cache_hit
            model_output, feedback_metadata = _tool_feedback_for_model(outcome.output)
            messages.append({
                "role": "tool", "tool_call_id": call.call_id, "content": model_output,
            })
            if (
                feedback_metadata["duplicate_lines_removed"] > 0
                or feedback_metadata["truncated"]
            ):
                self.tracer.emit(TraceEvent(
                    kind="tool_feedback_compacted",
                    thm_name=node_name,
                    turn=turn,
                    call_id=call.call_id,
                    tool_name=call.name,
                    args={"stage": stage, **feedback_metadata},
                ))
            if call.name == "lean_compile":
                turn_outcome.last_proof = outcome.proof_body
                turn_outcome.last_errors = list(outcome.errors)
                turn_outcome.last_failure_kind = outcome.failure_kind
                if outcome.ok:
                    successful.append((call.index, outcome.proof_body))
        if successful:
            turn_outcome.solved_proof = min(successful, key=lambda item: item[0])[1]
        return turn_outcome

    def _select_calls(
        self,
        tool_calls,
        stage: str,
        node_decl: str,
        parent_lemma_decls: str,
        header: str,
        limit: int,
        allowed_names: set[str],
    ) -> tuple[list[_AcceptedCall], list[dict[str, Any]]]:
        accepted: list[_AcceptedCall] = []
        dropped: list[dict[str, Any]] = []
        seen: set[str] = set()
        context_hash = hashlib.sha256(
            (stage + "\0" + header + "\0" + parent_lemma_decls + "\0" + node_decl).encode()
        ).hexdigest()
        for index, tc in enumerate(tool_calls):
            name = str(tc.function.name)
            raw_arguments = tc.function.arguments or "{}"
            try:
                args = json.loads(raw_arguments)
                if not isinstance(args, dict):
                    raise ValueError("arguments must be a JSON object")
            except (json.JSONDecodeError, ValueError) as exc:
                call_hash = hashlib.sha256(
                    (context_hash + "\0" + name + "\0" + raw_arguments).encode()
                ).hexdigest()
                dropped.append({
                    "index": index,
                    "reason": "invalid_arguments",
                    "detail": str(exc),
                    "hash": call_hash,
                })
                continue
            canonical = json.dumps(
                {"context": context_hash, "name": name, "arguments": args},
                sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            )
            call_hash = hashlib.sha256(canonical.encode()).hexdigest()
            argument_error = _tool_argument_error(name, args)
            if argument_error is not None:
                dropped.append({
                    "index": index,
                    "reason": "invalid_arguments",
                    "detail": argument_error,
                    "hash": call_hash,
                })
            elif name not in allowed_names:
                dropped.append({"index": index, "reason": "not_allowed", "hash": call_hash})
            elif call_hash in seen:
                dropped.append({"index": index, "reason": "duplicate", "hash": call_hash})
            elif len(accepted) >= limit:
                dropped.append({"index": index, "reason": "over_limit", "hash": call_hash})
            else:
                seen.add(call_hash)
                accepted.append(_AcceptedCall(index, tc.id, name, args, call_hash, tc))
        return accepted, dropped

    def _prepare_calls(
        self,
        tool_calls,
        allowed_names: set[str],
        limit: int,
        stage: str,
        node_decl: str,
        parent_lemma_decls: str,
        header: str,
    ) -> tuple[list[_AcceptedCall], list[dict[str, Any]]]:
        return self._select_calls(
            tool_calls, stage, node_decl, parent_lemma_decls, header,
            limit, allowed_names,
        )

    def _process_response(self, **kwargs) -> _TurnOutcome:
        return self._process_turn(**kwargs)

    def _execute_calls(
        self,
        calls: list[_AcceptedCall],
        compiler: KiminaLeanCompiler,
        node_decl: str,
        parent_lemma_decls: str,
        header: str,
        *,
        node_name: str = "",
        turn: int = 0,
        stage: str = "",
    ) -> list[_ExecutedCall]:
        outcomes: dict[int, tuple[_ToolOutcome, bool]] = {}
        uncached = [call for call in calls if call.call_hash not in self._tool_cache]

        def start_trace(call: _AcceptedCall) -> tuple[str, int]:
            span_id = uuid.uuid4().hex
            started_ns = time.monotonic_ns()
            self.tracer.emit(TraceEvent(
                kind="tool_call", thm_name=node_name, turn=turn,
                call_id=call.call_id, tool_name=call.name, span_id=span_id,
                args={"stage": stage, "arguments": call.args, "hash": call.call_hash},
            ))
            return span_id, started_ns

        def finish_trace(
            call: _AcceptedCall,
            outcome: _ToolOutcome,
            cache_hit: bool,
            span_id: str,
            started_ns: int,
        ) -> None:
            self.tracer.emit(TraceEvent(
                kind="tool_result", thm_name=node_name, turn=turn,
                call_id=call.call_id, tool_name=call.name, span_id=span_id,
                result=outcome.output, ok=outcome.ok,
                args={"stage": stage, "hash": call.call_hash, "cache_hit": cache_hit,
                      "timings": outcome.timings},
                duration_ms=(time.monotonic_ns() - started_ns) / 1_000_000,
            ))

        for call in calls:
            if call.call_hash in self._tool_cache:
                span_id, started_ns = start_trace(call)
                outcome = self._tool_cache[call.call_hash]
                outcomes[call.index] = (outcome, True)
                finish_trace(call, outcome, True, span_id, started_ns)

        compile_calls: list[_AcceptedCall] = []
        compile_requests: list[CompileRequest] = []
        search_calls: list[_AcceptedCall] = []
        for call in uncached:
            if call.name == "mathlib_search":
                search_calls.append(call)
                continue
            try:
                if call.name == "lean_compile":
                    proof = _normalize_node_proof_body(str(call.args["proof_body"]))
                    code = assemble_node_attempt(
                        node_decl, parent_lemma_decls, proof, header,
                    )
                else:
                    code = str(call.args["lean_code"])
                compile_calls.append(call)
                compile_requests.append(CompileRequest(code, request_id=call.call_hash))
            except (KeyError, ValueError) as exc:
                span_id, started_ns = start_trace(call)
                outcome = _ToolOutcome(
                    f"Tool protocol error: {exc}", errors=[str(exc)], failure_kind="assembly",
                )
                outcomes[call.index] = (outcome, False)
                self._tool_cache[call.call_hash] = outcome
                finish_trace(call, outcome, False, span_id, started_ns)

        def run_compiles() -> list[CompilerResult]:
            trace_starts = {call.index: start_trace(call) for call in compile_calls}
            try:
                results = compiler.check_many(compile_requests)
            except Exception as exc:  # noqa: BLE001
                results = [
                    CompilerResult(
                        False,
                        errors=[f"Lean batch execution failed: {exc}"],
                        failure_kind="infra",
                    )
                    for _call in compile_calls
                ]
            for call, result in zip(compile_calls, results, strict=True):
                proof = (
                    _normalize_node_proof_body(str(call.args["proof_body"]))
                    if call.name == "lean_compile" else ""
                )
                outcome = _compiler_outcome(result, proof)
                outcomes[call.index] = (outcome, False)
                self._tool_cache[call.call_hash] = outcome
                finish_trace(call, outcome, False, *trace_starts[call.index])
            return results

        def run_search(call: _AcceptedCall) -> _ToolOutcome:
            span_id, started_ns = start_trace(call)
            try:
                hits = self.retrieval.search(
                    str(call.args["query"]), int(call.args.get("k", 10)),
                )
                output = "\n\n".join(hit.format() for hit in hits) or "No Mathlib results."
                outcome = _ToolOutcome(output, ok=True)
            except Exception as exc:  # noqa: BLE001
                outcome = _ToolOutcome(f"Mathlib search failed: {exc}", failure_kind="infra")
            outcomes[call.index] = (outcome, False)
            self._tool_cache[call.call_hash] = outcome
            finish_trace(call, outcome, False, span_id, started_ns)
            return outcome

        with ThreadPoolExecutor(max_workers=max(1, 1 + len(search_calls))) as executor:
            compile_future = executor.submit(run_compiles) if compile_requests else None
            search_futures = {call.index: executor.submit(run_search, call) for call in search_calls}
            if compile_future is not None:
                compile_future.result()
            for call in search_calls:
                search_futures[call.index].result()
        return [
            _ExecutedCall(call, outcomes[call.index][0], outcomes[call.index][1])
            for call in calls
        ]

    def _probe_negation(
        self,
        compiler: KiminaLeanCompiler,
        node_name: str,
        node_decl: str,
        parent_lemma_decls: str,
        header: str,
    ) -> ProverResult | None:
        if self.max_negation_probe_turns == 0:
            return None
        try:
            negation_decl = _build_negation_node_decl(node_decl, node_name)
        except ValueError:
            return None
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": PROVER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Try to prove this formal negation. Call lean_compile with "
                    f"a proof body beginning with `by`.\n\n```lean\n{negation_decl}\n```"
                ),
            },
        ]
        base_messages = tuple(dict(message) for message in messages)
        for turn in range(1, self.max_negation_probe_turns + 1):
            response = self._chat(
                messages, node_name, turn, "negation_probe", [LEAN_COMPILE_TOOL],
                "negation_probe",
            )
            outcome = self._process_turn(
                response=response,
                messages=messages,
                compiler=compiler,
                node_name=node_name,
                node_decl=negation_decl,
                parent_lemma_decls=parent_lemma_decls,
                header=header,
                stage="negation_probe",
                turn=turn,
                limit=1,
                allowed_names={"lean_compile"},
            )
            if outcome.solved_proof:
                return ProverResult(ProofSignal.FORMALLY_NEGATED, outcome.solved_proof)
            _retain_latest_assistant_turn(messages, base_messages)
        return None

    def probe_negation_only(
        self,
        compiler: KiminaLeanCompiler,
        node_name: str,
        node_decl: str,
        parent_lemma_decls: str,
        header: str,
    ) -> ProverResult | None:
        """Run only the bounded formal-negation stage for a selected node."""
        started = time.monotonic()
        self.tracer.emit(TraceEvent(
            kind="critical_negation_start",
            thm_name=node_name,
            args={"max_turns": self.max_negation_probe_turns},
        ))
        result = self._probe_negation(
            compiler, node_name, node_decl, parent_lemma_decls, header,
        )
        self.tracer.emit(TraceEvent(
            kind="critical_negation_end",
            thm_name=node_name,
            ok=result is not None and result.signal == ProofSignal.FORMALLY_NEGATED,
            args={
                "max_turns": self.max_negation_probe_turns,
                "signal": result.signal.value if result is not None else "not_proved",
                "wall_time_s": time.monotonic() - started,
            },
        ))
        return result


def _compiler_outcome(result: CompilerResult, proof_body: str) -> _ToolOutcome:
    if result.success:
        return _ToolOutcome("Compilation SUCCESSFUL.", True, proof_body, timings=result.timings)
    lines = ["Compilation FAILED.", *result.errors]
    if result.goals:
        lines.extend(["Goals:", *result.goals])
    lines.append("Fix errors and call lean_compile again.")
    return _ToolOutcome(
        "\n".join(lines),
        False,
        proof_body,
        _stable_deduplicate_diagnostics(result.diagnostics),
        result.failure_kind,
        result.timings,
    )


def _build_negation_node_decl(node_decl: str, node_name: str) -> str:
    decl = lemma_to_theorem(extract_current_node_decl(node_decl)).strip()
    proof_match = BLUEPRINT_PROOF_RE.search(decl)
    signature = decl[:proof_match.start()].strip() if proof_match else decl.split(":=", 1)[0].strip()
    if not signature:
        raise ValueError("empty node declaration")
    head = re.match(r"^\s*(?:theorem|lemma)\s+\S+", signature)
    if not head:
        raise ValueError("node declaration is not a theorem/lemma")
    signature = f"theorem {_lean_safe_negation_name(node_name)}" + signature[head.end():]
    colon = _find_top_level_colon(signature)
    if colon is None:
        raise ValueError("could not find theorem conclusion separator")
    prefix = signature[:colon].rstrip()
    conclusion = signature[colon + 1:].strip()
    if not conclusion:
        raise ValueError("empty theorem conclusion")
    return f"{prefix} : ¬ ({conclusion}) := by sorry_using []"


def _lean_safe_negation_name(node_name: str) -> str:
    ident = re.sub(r"[^A-Za-z0-9_']", "_", f"neg_{node_name}")
    return ident if re.match(r"[A-Za-z_]", ident) else f"neg_{ident}"


def _find_top_level_colon(text: str) -> int | None:
    depth = 0
    i = 0
    in_string = False
    line_comment = False
    block_depth = 0
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if line_comment:
            line_comment = ch != "\n"
            i += 1
            continue
        if block_depth:
            if ch == "/" and nxt == "-":
                block_depth += 1
                i += 2
                continue
            if ch == "-" and nxt == "/":
                block_depth -= 1
                i += 2
                continue
            i += 1
            continue
        if in_string:
            if ch == "\\":
                i += 2
                continue
            in_string = ch != '"'
            i += 1
            continue
        if ch == "-" and nxt == "-":
            line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "-":
            block_depth = 1
            i += 2
            continue
        if ch == '"':
            in_string = True
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == ":" and depth == 0:
            return i
        i += 1
    return None


def _normalize_node_proof_body(proof_body: str) -> str:
    body = proof_body.strip()
    if body.startswith(":= by"):
        return "by" + body[len(":= by"):]
    if body.startswith(":="):
        return body[len(":="):].lstrip()
    return body


def _tool_argument_error(name: str, args: dict[str, Any]) -> str | None:
    required_string = {
        "lean_compile": "proof_body",
        "step_lean_compile": "lean_code",
        "mathlib_search": "query",
    }.get(name)
    if required_string is not None:
        value = args.get(required_string)
        if not isinstance(value, str) or not value.strip():
            return f"{required_string} must be a non-empty string"
    if name == "mathlib_search" and "k" in args:
        k = args["k"]
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            return "k must be a positive integer"
    return None


def prove_node(
    node_name: str,
    canonical_stmt: str,
    parent_signatures: list[str],
    definition_decls: list[str],
    parent_lemma_decls: str,
    header: str,
    compiler: KiminaLeanCompiler,
    retrieval: MathlibRetrieval,
    model: str = "labs-leanstral-1-5",
    node_statement_nl: str = "",
    node_proof_sketch_nl: str = "",
    tracer=None,
    api_timeout_s: float | None = 120.0,
    max_prove_turns: int | None = None,
    max_negation_probe_turns: int = 1,
    max_tool_calls_per_turn: int = DEFAULT_MAX_TOOL_CALLS_PER_TURN,
) -> ProverResult:
    context_parts: list[str] = []
    if definition_decls:
        context_parts.append(
            "Available definitions (complete declarations):\n```lean\n"
            + "\n\n".join(definition_decls)
            + "\n```"
        )
    if parent_signatures:
        context_parts.append(
            "Available proved parent signatures:\n```lean\n"
            + "\n".join(parent_signatures)
            + "\n```"
        )
    user_prompt = render(
        PROVER_USER_TEMPLATE,
        canonical_stmt=canonical_stmt,
        nl_statement=node_statement_nl,
        nl_proof_sketch=node_proof_sketch_nl,
        parent_proofs="\n\n".join(context_parts),
    )
    prover = GoedelProver(
        model_id=model,
        retrieval=retrieval,
        tracer=tracer,
        api_timeout_s=api_timeout_s,
        max_prove_turns=max_prove_turns,
        max_negation_probe_turns=max_negation_probe_turns,
        max_tool_calls_per_turn=max_tool_calls_per_turn,
    )
    return prover.prove_node(
        compiler, node_name, canonical_stmt, user_prompt, parent_lemma_decls, header,
    )


def probe_node_negation(
    *,
    node_name: str,
    canonical_stmt: str,
    parent_lemma_decls: str,
    header: str,
    compiler: KiminaLeanCompiler,
    retrieval: MathlibRetrieval,
    model: str = "labs-leanstral-1-5",
    tracer=None,
    api_timeout_s: float | None = 120.0,
    max_negation_probe_turns: int = 1,
    max_tool_calls_per_turn: int = DEFAULT_MAX_TOOL_CALLS_PER_TURN,
) -> ProverResult | None:
    """Probe one caller-selected failed node without rerunning normal proving."""
    prover = GoedelProver(
        model_id=model,
        retrieval=retrieval,
        tracer=tracer,
        api_timeout_s=api_timeout_s,
        max_prove_turns=1,
        max_negation_probe_turns=max_negation_probe_turns,
        max_tool_calls_per_turn=max_tool_calls_per_turn,
    )
    return prover.probe_negation_only(
        compiler, node_name, canonical_stmt, parent_lemma_decls, header,
    )

"""Phase 2: Per-node tool-equipped prover.

Uses the OpenAI-compatible chat.completions API (stateless — the running
`messages` list is threaded through explicitly), so it works against OpenAI,
Fireworks, or Mistral (see llm_client.make_client). Three tools: lean_compile,
repo_search, mathlib_search.

Compiler backend is injectable — pass a VSBLeanCompiler for VeriSoftBench or
LeanCompiler for standalone Lean projects.

Returns one of four structured signals per the paper:
    solved | statement_wrong | proof_too_hard | formally_negated
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from blueprint_text import BLUEPRINT_PROOF_RE, extract_current_node_decl, lemma_to_theorem
from lean_compiler import AbstractLeanCompiler, CompilerResult
from llm_client import chat_completion_with_retry, make_client
from mathlib_retrieval import MathlibRetrieval
from goedel_prompts import load, render
from tracer import NullTracer, TraceEvent


def _chat_reasoning_kwargs(model: str) -> dict:
    """Return reasoning_effort kwarg for models that support it (gpt-5.x/o-series).

    Only safe to pass on calls that omit `tools` — chat.completions rejects
    function tools + reasoning_effort together for these models ("Function
    tools with reasoning_effort are not supported ... Please use /v1/responses
    instead"), unlike the Responses API this loop used to call.
    """
    if model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3") or model.startswith("o4"):
        return {"reasoning_effort": "low"}
    return {}


def _tool_choice_kwargs(choice: str | dict) -> dict:
    """Return chat.completions tool_choice kwargs.

    Some OpenAI-compatible endpoints reject object/required tool_choice while a
    model is in thinking mode. Set GOEDEL_TOOL_CHOICE_MODE=auto to keep tools
    enabled while letting the model choose when to call them.
    """
    mode = os.environ.get("GOEDEL_TOOL_CHOICE_MODE", "").strip().lower()
    if mode == "auto" and choice != "none":
        return {"tool_choice": "auto"}
    if mode == "omit":
        return {}
    return {"tool_choice": choice}


def _parallel_tool_calls_kwargs(parallel_tool_calls: int | None) -> dict:
    if parallel_tool_calls is None:
        return {}
    return {"parallel_tool_calls": parallel_tool_calls > 1}


def _tool_call_limit_notice(parallel_tool_calls: int | None) -> str:
    if parallel_tool_calls is None:
        return ""
    return (
        "Per-turn tool budget: call at most "
        f"{parallel_tool_calls} tool(s) in each assistant turn. "
        "Extra tool calls in the same turn will be ignored."
    )


def _parse_tool_arguments(raw_arguments: str) -> Any:
    if not raw_arguments:
        return {}
    try:
        return json.loads(raw_arguments)
    except json.JSONDecodeError:
        return raw_arguments


def _tool_calls_payload(tool_calls) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for index, tc in enumerate(tool_calls or []):
        raw_arguments = tc.function.arguments or ""
        payload.append({
            "index": index,
            "id": tc.id,
            "type": getattr(tc, "type", "function"),
            "name": tc.function.name,
            "arguments": _parse_tool_arguments(raw_arguments),
            "arguments_raw": raw_arguments,
        })
    return payload


def _reconstruct_tool_calls_text(tool_calls) -> str:
    blocks: list[str] = []
    for tc in tool_calls or []:
        raw_arguments = tc.function.arguments or ""
        blocks.append(
            "<tool_call>:\n"
            f"name={tc.function.name}\n"
            f"arguments={raw_arguments}\n"
            "</tool_call>"
        )
    return "\n\n".join(blocks)

try:
    from repo_retrieval import RepoRetrieval
    _HAS_REPO_RETRIEVAL = True
except ImportError:
    _HAS_REPO_RETRIEVAL = False

PROVER_SYSTEM_PROMPT = load("prover_system")
PROVER_USER_TEMPLATE = load("prover_user")

def _max_tokens() -> int:
    return int(os.environ.get("GOEDEL_PROVER_MAX_TOKENS", "64000"))


DEFAULT_NODE_MAX_PROVE_TURNS = 8
DEFAULT_NODE_MAX_NEGATION_PROBE_TURNS = 4

SYSTEM_SUFFIX_WITH_REPO_SEARCH = """
## Tool-First Workflow

You have three tools: lean_compile, repo_search, mathlib_search.

**Key insight**: the prompt already contains repo context in <used_repo_defs>,
<repo_lemmas>, and <local_ctx>. Read these first — the lemmas you need are
likely already visible.

Workflow:
1. Draft a proof using the visible repo definitions.
2. Call lean_compile with your proof_body (starting with `:= by` — the harness
   appends proof_body directly after the bare theorem signature, so the leading
   `:=` is required or the submission fails to parse).
3. Read errors, adjust, call lean_compile again.
4. When you need a lemma whose name you do NOT already know:
   - call repo_search for project-specific lemmas
   - call mathlib_search for general Mathlib lemmas
5. A successful lean_compile call is the only accepted proof source.

Prefer lean_compile over search — faster to try a tactic and read the error.
"""

SYSTEM_SUFFIX_WITHOUT_REPO_SEARCH = """
## Tool-First Workflow

You have two tools: lean_compile and mathlib_search. No repository search
tool is available in this experiment.

Workflow:
1. Draft a proof using the declarations and context already visible in the
   prompt.
2. Call lean_compile with your proof_body (starting with `:= by` — the harness
   appends proof_body directly after the bare theorem signature, so the leading
   `:=` is required or the submission fails to parse).
3. Read errors, adjust, and call lean_compile again.
4. When you need a general Mathlib lemma whose name you do not know, call
   mathlib_search.
5. A successful lean_compile call is the only accepted proof source.

Prefer lean_compile over search — faster to try a tactic and read the error.
"""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lean_compile",
            "description": (
                "Compile and verify a proof attempt. "
                "Returns 'Compilation SUCCESSFUL' or detailed Lean error messages."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "proof_body": {
                        "type": "string",
                        "description": (
                            "Proof term starting with ':= by' (the leading ':=' is required — "
                            "this gets appended directly after the bare theorem signature). "
                            "Do NOT include the theorem declaration."
                        ),
                    },
                    "aux_lemmas": {
                        "type": "string",
                        "description": "Optional helper lemma declarations to define before the target theorem.",
                    },
                },
                "required": ["proof_body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "repo_search",
            "description": (
                "Semantic search over the target repository's .lean files. "
                "Use BEFORE mathlib_search for project-specific lemmas, induction principles, or coercions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language or identifier fragment."},
                    "k": {"type": "integer", "description": "Number of results (default 10).", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
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
    },
]


def _tools_for_repo_retrieval(repo_retrieval) -> list[dict[str, Any]]:
    """Expose repo_search only when a real repository index was supplied."""
    if repo_retrieval is not None:
        return list(TOOLS)
    return [
        tool for tool in TOOLS
        if tool["function"]["name"] != "repo_search"
    ]


def _system_suffix_for_repo_retrieval(repo_retrieval) -> str:
    """Keep the system prompt consistent with the tools sent to the model."""
    if repo_retrieval is not None:
        return SYSTEM_SUFFIX_WITH_REPO_SEARCH
    return SYSTEM_SUFFIX_WITHOUT_REPO_SEARCH


# ---------------------------------------------------------------------------
# Result types (unchanged from paper)
# ---------------------------------------------------------------------------

class ProofSignal(str, Enum):
    SOLVED = "solved"
    STATEMENT_WRONG = "statement_wrong"
    PROOF_TOO_HARD = "proof_too_hard"
    BLOCKED_BY_DEPENDENCY = "blocked_by_dependency"
    FORMALLY_NEGATED = "formally_negated"
    # Not one of the paper's four signals: an infra/tooling failure (timeout,
    # unhandled exception) rather than a genuine "the model tried and
    # couldn't" verdict. Kept distinct so refinement (and human diagnosis)
    # doesn't treat a broken harness as evidence the sub-goal is hard.
    INFRA_ERROR = "infra_error"


@dataclass
class ProverResult:
    signal: ProofSignal
    proof_body: str = ""
    analysis: str = ""
    suggested_fix: str = ""
    lean_errors: list[str] = field(default_factory=list)

    def diagnosis_block(self, node_name: str) -> str:
        if self.signal == ProofSignal.FORMALLY_NEGATED:
            return (
                f"/- Diagnosis\n## Diagnosis\nFORMALLY_NEGATED\n\n"
                f"## Analysis\n{self.analysis}\n\n"
                f"## Counterexample Proof\n```lean\n{self.proof_body}\n```\n\n"
                f"## Suggested Fix\n{self.suggested_fix}\n-/"
            )
        return (
            f"/- Diagnosis\n## Diagnosis\n{self.signal.value.upper()}\n\n"
            f"## Analysis\n{self.analysis}\n\n"
            f"## Suggested Fix\n{self.suggested_fix}\n-/"
        )


# ---------------------------------------------------------------------------
# Prover
# ---------------------------------------------------------------------------

class GoedelProver:
    """
    Phase 2 per-node tool-equipped prover.

    Inject a compiler backend (VSBLeanCompiler for VeriSoftBench, LeanCompiler
    for standalone) and optionally a RepoRetrieval for repo_search.
    """

    def __init__(
        self,
        model_id: str = "labs-leanstral-1-5",
        retrieval: MathlibRetrieval | None = None,
        tracer=None,
        api_timeout_s: float | None = 120.0,
        max_prove_turns: int | None = None,
        max_negation_probe_turns: int | None = None,
        parallel_tool_calls: int | None = None,
    ):
        self.model_id = model_id
        # Bounds each individual chat.completions call so a stuck request
        # can't hang a node indefinitely; the orchestrator's node_timeout_s
        # bounds the whole multi-turn tool loop on top of this.
        self.client = make_client(model_id, timeout=api_timeout_s)
        self.retrieval = retrieval or MathlibRetrieval()
        self.tracer = tracer or NullTracer()
        self.parallel_tool_calls = parallel_tool_calls
        self.max_prove_turns = (
            max_prove_turns
            if max_prove_turns is not None
            else DEFAULT_NODE_MAX_PROVE_TURNS
        )
        self.max_negation_probe_turns = (
            max_negation_probe_turns
            if max_negation_probe_turns is not None
            else DEFAULT_NODE_MAX_NEGATION_PROBE_TURNS
        )

    def _emit_usage(self, node_name: str, response) -> None:
        """Log token usage from a chat.completions response."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        prompt = getattr(usage, "prompt_tokens", 0)
        completion = getattr(usage, "completion_tokens", 0)
        total = getattr(usage, "total_tokens", None) or (prompt + completion)
        self.tracer.emit(TraceEvent(
            kind="llm_usage",
            thm_name=node_name,
            args={
                "phase": "phase2", "model": self.model_id,
                "prompt_tokens": prompt, "completion_tokens": completion,
                "total_tokens": total,
            },
        ))

    def _emit_llm_response(
        self,
        node_name: str,
        response,
        *,
        turn: int,
        stage: str,
        max_turns: int,
        operation: str = "",
        tool_calls_processed: int = 0,
        tool_calls_dropped: int = 0,
    ) -> None:
        choice = response.choices[0]
        msg = choice.message
        tool_call_payload = _tool_calls_payload(msg.tool_calls)
        self.tracer.emit(TraceEvent(
            kind="llm_response",
            thm_name=node_name,
            turn=turn,
            result=msg.content or "",
            args={
                "phase": "phase2",
                "model": getattr(response, "model", self.model_id),
                "response_id": getattr(response, "id", None),
                "finish_reason": getattr(choice, "finish_reason", None),
                "stage": stage,
                "operation": operation,
                "max_turns": max_turns,
                "tool_call_count": len(tool_call_payload),
                "tool_calls_processed": tool_calls_processed,
                "tool_calls_dropped": tool_calls_dropped,
                "content": msg.content,
                "content_len": len(msg.content or ""),
                "tool_calls": tool_call_payload,
                "reconstructed_tool_calls_text": _reconstruct_tool_calls_text(msg.tool_calls),
            },
        ))

    def prove_node(
        self,
        compiler: AbstractLeanCompiler,
        node_name: str,
        node_stmt: str,
        sys_prompt: str = "",
        user_prompt: str = "",
        nl_statement: str = "",
        nl_proof_sketch: str = "",
        repo_retrieval=None,
        parent_lemma_decls: str = "",
    ) -> ProverResult:
        """Attempt to prove a single node, timing it and emitting a final_verify trace event."""
        t0 = time.time()
        result = self._prove_node_inner(
            compiler, node_name, node_stmt, sys_prompt, user_prompt,
            nl_statement, nl_proof_sketch, repo_retrieval,
            parent_lemma_decls=parent_lemma_decls,
        )
        self.tracer.emit(TraceEvent(
            kind="final_verify",
            thm_name=node_name,
            ok=result.signal == ProofSignal.SOLVED,
            args={
                "wall_time_s": time.time() - t0,
                "proof": result.proof_body,
                "error": result.analysis,
            },
        ))
        return result

    def _prove_node_inner(
        self,
        compiler: AbstractLeanCompiler,
        node_name: str,
        node_stmt: str,
        sys_prompt: str = "",
        user_prompt: str = "",
        nl_statement: str = "",
        nl_proof_sketch: str = "",
        repo_retrieval=None,
        parent_lemma_decls: str = "",
    ) -> ProverResult:
        """Attempt to prove a single node using the chat.completions tool loop."""
        # Stashed on self rather than threaded through every _process_response /
        # _probe_negation call - one GoedelProver instance proves exactly one
        # node (see the module-level prove_node() factory), so this is safe.
        self._parent_lemma_decls = parent_lemma_decls
        active_tools = _tools_for_repo_retrieval(repo_retrieval)
        system_suffix = _system_suffix_for_repo_retrieval(repo_retrieval)
        tool_limit_notice = _tool_call_limit_notice(self.parallel_tool_calls)
        augmented_sys = (sys_prompt or PROVER_SYSTEM_PROMPT).strip() + "\n\n" + system_suffix.strip()
        if tool_limit_notice:
            augmented_sys += "\n\n" + tool_limit_notice

        if not user_prompt:
            user_prompt = render(
                PROVER_USER_TEMPLATE,
                canonical_stmt=node_stmt,
                nl_statement=nl_statement,
                nl_proof_sketch=nl_proof_sketch,
                parent_proofs="",
            )

        self.tracer.emit(TraceEvent(
            kind="theorem_start",
            thm_name=node_name,
            args={"thm_stmt": node_stmt},
        ))

        # `messages` is mutated in place by _process_response (assistant
        # tool-call message + tool result messages appended each turn) - it
        # replaces the Responses API's previous_response_id chaining.
        messages: list[dict] = [
            {"role": "system", "content": augmented_sys},
            {"role": "user", "content": user_prompt},
        ]
        max_tokens = _max_tokens()

        # Force first call to lean_compile
        response_operation = "prove_node_initial"
        response = chat_completion_with_retry(
            self.client,
            tracer=self.tracer,
            thm_name=node_name,
            phase="phase2",
            model_id=self.model_id,
            operation="prove_node_initial",
            trace_args={
                "stage": "prove",
                "turn": 1,
                "max_turns": self.max_prove_turns,
            },
            model=self.model_id,
            messages=messages,
            tools=active_tools,
            max_completion_tokens=max_tokens,
            **_parallel_tool_calls_kwargs(self.parallel_tool_calls),
            **_tool_choice_kwargs({"type": "function", "function": {"name": "lean_compile"}}),
        )
        self._emit_usage(node_name, response)

        last_text = ""
        all_lean_errors: list[str] = []

        for turn in range(1, self.max_prove_turns + 1):
            had_tool_calls, text, proof, compile_ok, tools_called, compile_errors = self._process_response(
                response,
                messages,
                compiler,
                node_name,
                repo_retrieval,
                turn=turn,
                stage="prove",
                max_turns=self.max_prove_turns,
                node_decl=node_stmt,
                operation=response_operation,
            )
            all_lean_errors.extend(compile_errors)
            if text:
                last_text = text
            if compile_ok:
                return ProverResult(signal=ProofSignal.SOLVED, proof_body=proof)

            had_search = any(t in ("repo_search", "mathlib_search") for t in tools_called)
            had_compile = "lean_compile" in tools_called

            if not had_tool_calls:
                break
            if turn >= self.max_prove_turns:
                break

            next_choice = (
                {"type": "function", "function": {"name": "lean_compile"}}
                if (had_search and not had_compile) else "required"
            )

            response_operation = "prove_node_next"
            response = chat_completion_with_retry(
                self.client,
                tracer=self.tracer,
                thm_name=node_name,
                phase="phase2",
                model_id=self.model_id,
                operation="prove_node_next",
                trace_args={
                    "stage": "prove",
                    "turn": turn + 1,
                    "max_turns": self.max_prove_turns,
                },
                model=self.model_id,
                messages=messages,
                tools=active_tools,
                max_completion_tokens=max_tokens,
                **_parallel_tool_calls_kwargs(self.parallel_tool_calls),
                **_tool_choice_kwargs(next_choice),
            )
            self._emit_usage(node_name, response)

        # Probe negation if we couldn't prove it
        negation = self._probe_negation(
            compiler,
            node_name,
            messages,
            max_tokens,
            node_decl=node_stmt,
        )
        if negation:
            return negation

        return ProverResult(signal=_classify_failure(last_text),
                            analysis=last_text[:500], lean_errors=all_lean_errors)

    # ------------------------------------------------------------------
    # Response processing
    # ------------------------------------------------------------------

    def _process_response(
        self,
        response,
        messages: list[dict],
        compiler: AbstractLeanCompiler,
        node_name: str,
        repo_retrieval,
        *,
        turn: int,
        stage: str,
        max_turns: int,
        node_decl: str = "",
        operation: str = "",
    ) -> tuple[bool, str, str, bool, list[str], list[str]]:
        """Append this turn's assistant + tool-result messages to `messages`
        in place (chat.completions is stateless - the caller re-sends the
        full history each call) and return what happened this turn.

        Returns (had_tool_calls, last_text, proof, any_compile_ok,
        tools_called, compile_errors) - had_tool_calls replaces the old
        Responses-API tool_results list as the "was anything called this
        turn" signal.
        """
        choice = response.choices[0]
        msg = choice.message
        last_text = ""
        compiled_proof = ""   # proof body that actually compiled — never overwritten by message text
        any_compile_ok = False
        tools_called: list[str] = []
        compile_errors: list[str] = []
        tool_calls = list(msg.tool_calls or [])
        limit = self.parallel_tool_calls
        calls_to_process = tool_calls if limit is None else tool_calls[:limit]
        calls_to_drop = [] if limit is None else tool_calls[limit:]
        self._emit_llm_response(
            node_name,
            response,
            turn=turn,
            stage=stage,
            max_turns=max_turns,
            operation=operation,
            tool_calls_processed=len(calls_to_process),
            tool_calls_dropped=len(calls_to_drop),
        )

        assistant_msg: dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        if msg.content:
            last_text = msg.content
            self.tracer.emit(TraceEvent(
                kind="model_text", thm_name=node_name, turn=turn,
                result=msg.content,
            ))

        for tc in calls_to_process:
            fn = tc.function.name
            args = json.loads(tc.function.arguments)
            tools_called.append(fn)
            tool_ok = False

            self.tracer.emit(TraceEvent(
                kind="tool_call", thm_name=node_name, turn=turn,
                call_id=tc.id, tool_name=fn,
                args={"stage": stage, "arguments": args},
            ))

            if fn == "lean_compile":
                proof_body = _normalize_node_proof_body(args.get("proof_body", ""))
                aux = args.get("aux_lemmas", "")
                # Splice in already-proved sibling lemmas as real declarations
                # (see BlueprintNode.signature) so the model can reference them
                # by name instead of hitting "unknown identifier".
                parent_decls = getattr(self, "_parent_lemma_decls", "")
                full_aux = f"{parent_decls}\n\n{aux}".strip() if parent_decls else aux
                cr = compiler.check(proof_body, aux_lemmas=full_aux, node_decl=node_decl)
                trace_args = {
                    "success": cr.success,
                    "errors": cr.errors,
                    "warnings": cr.warnings,
                    "goals": cr.goals,
                    "raw_output": cr.raw_output,
                    "validated": cr.validated,
                }
                if cr.success:
                    result = "Compilation SUCCESSFUL. Proof is correct."
                    any_compile_ok = True
                    tool_ok = True
                    compiled_proof = proof_body
                else:
                    errs = "\n".join(cr.errors)
                    result = f"Compilation FAILED.\n{errs}\n\nFix errors and call lean_compile again."
                    compile_errors.extend(cr.errors)

            elif fn == "repo_search" and repo_retrieval is not None:
                hits = repo_retrieval.search(args["query"], args.get("k", 10))
                result = "\n\n".join(h.format() for h in hits) or "No results in repo."
                trace_args = None

            elif fn == "mathlib_search":
                hits = self.retrieval.search(args["query"], args.get("k", 10))
                result = "\n\n".join(h.format() for h in hits) or "No results found."
                trace_args = None

            else:
                result = f"Tool unavailable: {fn}"
                trace_args = None

            self.tracer.emit(TraceEvent(
                kind="tool_result", thm_name=node_name, turn=turn,
                call_id=tc.id, tool_name=fn,
                args=trace_args,
                result=result, ok=tool_ok,
            ))

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        for tc in calls_to_drop:
            result = (
                "Tool call ignored: per-turn tool-call limit is "
                f"{self.parallel_tool_calls}. Continue with at most "
                f"{self.parallel_tool_calls} tool call(s) per assistant turn."
            )
            self.tracer.emit(TraceEvent(
                kind="tool_call_dropped",
                thm_name=node_name,
                turn=turn,
                call_id=tc.id,
                tool_name=tc.function.name,
                args={
                    "stage": stage,
                    "parallel_tool_calls": self.parallel_tool_calls,
                    "tool_calls_returned": len(tool_calls),
                    "tool_calls_processed": len(calls_to_process),
                },
                result=result,
                ok=False,
            ))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        return bool(msg.tool_calls), last_text, compiled_proof, any_compile_ok, tools_called, compile_errors

    # ------------------------------------------------------------------
    # Negation probe (Section 4.3 / Figure 1)
    # ------------------------------------------------------------------

    def _probe_negation(
        self,
        compiler: AbstractLeanCompiler,
        node_name: str,
        messages: list[dict],
        max_tokens: int,
        node_decl: str,
    ) -> ProverResult | None:
        if self.max_negation_probe_turns <= 0:
            self.tracer.emit(TraceEvent(
                kind="negation_probe_skipped",
                thm_name=node_name,
                args={"reason": "node_max_negation_probe_turns=0"},
            ))
            return None

        try:
            negation_node_decl = _build_negation_node_decl(node_decl, node_name)
        except ValueError as exc:
            self.tracer.emit(TraceEvent(
                kind="negation_probe_skipped",
                thm_name=node_name,
                args={"reason": str(exc)},
            ))
            return None

        prompt = (
            f"You could not prove the statement. Try to show it is FALSE.\n"
            "Prove the following generated Lean declaration, which keeps the same "
            "parameters and hypotheses but negates the conclusion:\n"
            "```lean\n"
            f"{negation_node_decl}\n"
            "```\n"
            "Call lean_compile with only the proof_body for this declaration. "
            "Tactics: `omega`, `decide`, `norm_num`, `push_neg; linarith`, `simp`.\n"
            "Call lean_compile. If it succeeds, the original statement is formally refuted."
        )
        tool_limit_notice = _tool_call_limit_notice(self.parallel_tool_calls)
        if tool_limit_notice:
            prompt += "\n\n" + tool_limit_notice
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            tools=TOOLS,
            max_completion_tokens=max_tokens,
            **_parallel_tool_calls_kwargs(self.parallel_tool_calls),
            **_tool_choice_kwargs({"type": "function", "function": {"name": "lean_compile"}}),
        )
        self._emit_usage(node_name, response)

        for turn in range(1, self.max_negation_probe_turns + 1):
            msg = response.choices[0].message
            tool_calls = list(msg.tool_calls or [])
            limit = self.parallel_tool_calls
            calls_to_process = tool_calls if limit is None else tool_calls[:limit]
            calls_to_drop = [] if limit is None else tool_calls[limit:]
            self._emit_llm_response(
                node_name,
                response,
                turn=turn,
                stage="negation_probe",
                max_turns=self.max_negation_probe_turns,
                operation="negation_probe",
                tool_calls_processed=len(calls_to_process),
                tool_calls_dropped=len(calls_to_drop),
            )
            assistant_msg: dict = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_msg)

            if not msg.tool_calls:
                break

            for tc in calls_to_process:
                args = json.loads(tc.function.arguments)
                self.tracer.emit(TraceEvent(
                    kind="tool_call",
                    thm_name=node_name,
                    turn=turn,
                    call_id=tc.id,
                    tool_name=tc.function.name,
                    args={"stage": "negation_probe", "arguments": args},
                ))

                if tc.function.name == "lean_compile":
                    parent_decls = getattr(self, "_parent_lemma_decls", "")
                    proof_body = _normalize_node_proof_body(args.get("proof_body", ""))
                    cr = compiler.check(
                        proof_body,
                        aux_lemmas=parent_decls,
                        node_decl=negation_node_decl,
                    )
                    if cr.success:
                        self.tracer.emit(TraceEvent(
                            kind="tool_result",
                            thm_name=node_name,
                            turn=turn,
                            call_id=tc.id,
                            tool_name=tc.function.name,
                            args={
                                "success": cr.success,
                                "errors": cr.errors,
                                "warnings": cr.warnings,
                                "goals": cr.goals,
                                "raw_output": cr.raw_output,
                                "validated": cr.validated,
                                "negation_node_decl": negation_node_decl,
                            },
                            result="Compilation SUCCESSFUL. Negation proof is correct.",
                            ok=True,
                        ))
                        return ProverResult(
                            signal=ProofSignal.FORMALLY_NEGATED,
                            proof_body=proof_body,
                            analysis=f"{node_name} is formally refuted — a proof of ¬(statement) was found.",
                            suggested_fix="The statement is mathematically FALSE. Revise it.",
                        )
                    output = f"Compilation FAILED.\n" + "\n".join(cr.errors)
                    trace_args = {
                        "success": cr.success,
                        "errors": cr.errors,
                        "warnings": cr.warnings,
                        "goals": cr.goals,
                        "raw_output": cr.raw_output,
                        "validated": cr.validated,
                        "negation_node_decl": negation_node_decl,
                    }
                else:
                    output = "Tool unavailable in negation probe."
                    trace_args = None
                self.tracer.emit(TraceEvent(
                    kind="tool_result",
                    thm_name=node_name,
                    turn=turn,
                    call_id=tc.id,
                    tool_name=tc.function.name,
                    args=trace_args,
                    result=output,
                    ok=False,
                ))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})

            for tc in calls_to_drop:
                output = (
                    "Tool call ignored: per-turn tool-call limit is "
                    f"{self.parallel_tool_calls}. Continue with at most "
                    f"{self.parallel_tool_calls} tool call(s) per assistant turn."
                )
                self.tracer.emit(TraceEvent(
                    kind="tool_call_dropped",
                    thm_name=node_name,
                    turn=turn,
                    call_id=tc.id,
                    tool_name=tc.function.name,
                    args={
                        "stage": "negation_probe",
                        "parallel_tool_calls": self.parallel_tool_calls,
                        "tool_calls_returned": len(tool_calls),
                        "tool_calls_processed": len(calls_to_process),
                    },
                    result=output,
                    ok=False,
                ))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})

            if turn >= self.max_negation_probe_turns:
                break

            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_completion_tokens=max_tokens,
                **_parallel_tool_calls_kwargs(self.parallel_tool_calls),
            )
            self._emit_usage(node_name, response)

        return None


def _build_negation_node_decl(node_decl: str, node_name: str) -> str:
    """Return a theorem declaration proving the negated conclusion of a node.

    The prover's lean_compile tool checks proof bodies by splicing them into a
    declaration containing `:= by sorry_using [...]`.  Negation probing must use
    the same contract; otherwise backends receive a bare `:= by ...` at top
    level, which Lean rejects before checking the proof.
    """
    decl = lemma_to_theorem(extract_current_node_decl(node_decl)).strip()
    proof_match = BLUEPRINT_PROOF_RE.search(decl)
    if proof_match:
        signature = decl[:proof_match.start()].strip()
    else:
        signature = decl.split(":=", 1)[0].strip()

    if not signature:
        raise ValueError("empty node declaration")

    head = re.match(r"^\s*(?:theorem|lemma)\s+\S+", signature)
    if not head:
        raise ValueError("node declaration is not a theorem/lemma")

    neg_name = _lean_safe_negation_name(node_name)
    signature = f"theorem {neg_name}" + signature[head.end():]
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
    if not re.match(r"[A-Za-z_]", ident):
        ident = f"neg_{ident}"
    return ident


def _find_top_level_colon(text: str) -> int | None:
    depth = 0
    i = 0
    in_string = False
    line_comment = False
    block_comment_depth = 0
    pairs = {"(": ")", "[": "]", "{": "}"}
    closers = set(pairs.values())

    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""

        if line_comment:
            if ch == "\n":
                line_comment = False
            i += 1
            continue

        if block_comment_depth:
            if ch == "/" and nxt == "-":
                block_comment_depth += 1
                i += 2
                continue
            if ch == "-" and nxt == "/":
                block_comment_depth -= 1
                i += 2
                continue
            i += 1
            continue

        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue

        if ch == "-" and nxt == "-":
            line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "-":
            block_comment_depth = 1
            i += 2
            continue
        if ch == '"':
            in_string = True
            i += 1
            continue
        if ch in pairs:
            depth += 1
        elif ch in closers and depth > 0:
            depth -= 1
        elif ch == ":" and depth == 0:
            return i
        i += 1

    return None


def _normalize_node_proof_body(proof_body: str) -> str:
    """Normalize tool proof_body for node_decl substitution.

    LeanCompiler.check(..., node_decl=...) substitutes the proof into
    `:= by sorry_using [...]` as `:= {proof_body}`, so the proof body must be
    `by ...`, not `:= by ...`.
    """
    body = proof_body.strip()
    if body.startswith(":= by"):
        return "by" + body[len(":= by"):]
    if body.startswith(":="):
        return body[len(":="):].lstrip()
    return body


def _classify_failure(analysis: str) -> ProofSignal:
    """`STATEMENT_WRONG` is reserved for cases with real evidence of falsity:
    `_probe_negation` formally proving the negation (a separate signal,
    FORMALLY_NEGATED) or the model's own analysis explicitly claiming
    false/counterexample (word-boundary matched so identifiers like
    `t_false`/`progress_false` don't false-positive on the bare substring
    "false"). A compiler "type mismatch" is NOT such evidence - it's one of
    the most generic Lean elaboration errors and fires for merely-wrong
    tactics/lemma applications on true theorems just as often as on false
    ones, so it must not be used to infer the goal itself is false."""
    analysis_lower = analysis.lower()
    if re.search(r"\b(false|counterexample)\b", analysis_lower):
        return ProofSignal.STATEMENT_WRONG
    return ProofSignal.PROOF_TOO_HARD


# ---------------------------------------------------------------------------
# Backward-compat function (used by orchestrator.py)
# ---------------------------------------------------------------------------

def prove_node(
    node_name: str,
    canonical_stmt: str,
    parent_proofs: dict[str, str],
    parent_lemma_decls: str,
    compiler: AbstractLeanCompiler,
    retrieval: MathlibRetrieval,
    model: str = "labs-leanstral-1-5",
    node_statement_nl: str = "",
    node_proof_sketch_nl: str = "",
    repo_retrieval=None,
    tracer=None,
    api_timeout_s: float | None = 120.0,
    max_prove_turns: int | None = None,
    max_negation_probe_turns: int | None = None,
    parallel_tool_calls: int | None = None,
) -> ProverResult:
    parent_block = "\n\n".join(
        f"```lean\n-- {n}\n{p}\n```" for n, p in parent_proofs.items()
    )
    user_prompt = render(
        PROVER_USER_TEMPLATE,
        canonical_stmt=canonical_stmt,
        nl_statement=node_statement_nl,
        nl_proof_sketch=node_proof_sketch_nl,
        parent_proofs=parent_block,
    )
    prover = GoedelProver(model_id=model, retrieval=retrieval, tracer=tracer,
                           api_timeout_s=api_timeout_s,
                           max_prove_turns=max_prove_turns,
                           max_negation_probe_turns=max_negation_probe_turns,
                           parallel_tool_calls=parallel_tool_calls)
    return prover.prove_node(
        compiler=compiler,
        node_name=node_name,
        node_stmt=canonical_stmt,
        user_prompt=user_prompt,
        repo_retrieval=repo_retrieval,
        parent_lemma_decls=parent_lemma_decls,
    )

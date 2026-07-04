"""Phase 2: Per-node tool-equipped prover.

Uses the OpenAI Responses API (stateful, previous_response_id chaining).
Three tools: lean_compile, repo_search, mathlib_search.

Compiler backend is injectable — pass a VSBLeanCompiler for VeriSoftBench or
LeanCompiler for standalone Lean projects.

Returns one of four structured signals per the paper:
    solved | statement_wrong | proof_too_hard | formally_negated
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from openai import OpenAI

from lean_compiler import AbstractLeanCompiler, CompilerResult
from mathlib_retrieval import MathlibRetrieval
from goedel_prompts import load, render
from tracer import NullTracer, TraceEvent


def _responses_reasoning_kwargs(model: str) -> dict:
    """Return reasoning kwarg for models that support it in the Responses API."""
    if model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3") or model.startswith("o4"):
        return {"reasoning": {"effort": "low"}}
    return {}

try:
    from repo_retrieval import RepoRetrieval
    _HAS_REPO_RETRIEVAL = True
except ImportError:
    _HAS_REPO_RETRIEVAL = False

PROVER_SYSTEM_PROMPT = load("prover_system")
PROVER_USER_TEMPLATE = load("prover_user")

# From Appendix A: "each node retries up to 4 times." The paper's mechanism
# (discrete full-code resubmissions) differs from this loop (one continuous
# multi-turn conversation), so this matches the stated budget number, not
# the exact retry semantics — a true match would need a different loop shape.
# Token budget capped to 32,000 (below the paper's 65,536) to control cost.
MAX_TOKENS = 32_000
MAX_TOOL_CALLS = 4
NEGATION_PROBE_CALLS = 4

SYSTEM_SUFFIX = """
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
5. Once lean_compile returns SUCCESSFUL, output:
   <lean4_proof>:= by\n  ...\n</lean4_proof>

Prefer lean_compile over search — faster to try a tactic and read the error.
"""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
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
    {
        "type": "function",
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
    {
        "type": "function",
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
]


# ---------------------------------------------------------------------------
# Result types (unchanged from paper)
# ---------------------------------------------------------------------------

class ProofSignal(str, Enum):
    SOLVED = "solved"
    STATEMENT_WRONG = "statement_wrong"
    PROOF_TOO_HARD = "proof_too_hard"
    FORMALLY_NEGATED = "formally_negated"


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
        model_id: str = "gpt-4o",
        retrieval: MathlibRetrieval | None = None,
        tracer=None,
        api_timeout_s: float = 120.0,
    ):
        self.model_id = model_id
        # Bounds each individual Responses API call so a stuck request can't
        # hang a node indefinitely; the orchestrator's node_timeout_s bounds
        # the whole multi-turn tool loop on top of this.
        self.client = OpenAI(timeout=api_timeout_s)
        self.retrieval = retrieval or MathlibRetrieval()
        self.tracer = tracer or NullTracer()

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
    ) -> ProverResult:
        """Attempt to prove a single node, timing it and emitting a final_verify trace event."""
        t0 = time.time()
        result = self._prove_node_inner(
            compiler, node_name, node_stmt, sys_prompt, user_prompt,
            nl_statement, nl_proof_sketch, repo_retrieval,
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
    ) -> ProverResult:
        """Attempt to prove a single node using the Responses API tool loop."""
        augmented_sys = (sys_prompt or PROVER_SYSTEM_PROMPT).strip() + "\n\n" + SYSTEM_SUFFIX.strip()

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

        # Force first call to lean_compile
        response = self.client.responses.create(
            model=self.model_id,
            instructions=augmented_sys,
            input=user_prompt,
            tools=TOOLS,
            tool_choice={"type": "function", "name": "lean_compile"},
            max_output_tokens=MAX_TOKENS,
            **_responses_reasoning_kwargs(self.model_id),
        )

        tool_calls_used = 0
        best_proof_body = ""
        last_text = ""
        tool_results: list[dict] = []
        last_compile_ok = False

        while tool_calls_used < MAX_TOOL_CALLS:
            tool_results, text, proof, compile_ok, tools_called = self._process_response(
                response, compiler, node_name, repo_retrieval, tool_calls_used, node_decl=node_stmt
            )
            if text:
                last_text = text
            if proof:
                best_proof_body = proof
            if compile_ok:
                last_compile_ok = True
                return ProverResult(signal=ProofSignal.SOLVED, proof_body=proof)

            had_search = any(t in ("repo_search", "mathlib_search") for t in tools_called)
            had_compile = "lean_compile" in tools_called
            tool_calls_used += len(tool_results)

            if not tool_results:
                break
            if tool_calls_used >= MAX_TOOL_CALLS:
                break

            next_choice = {"type": "function", "name": "lean_compile"} if (had_search and not had_compile) else "required"

            response = self.client.responses.create(
                model=self.model_id,
                previous_response_id=response.id,
                input=tool_results,
                tools=TOOLS,
                tool_choice=next_choice,
                max_output_tokens=MAX_TOKENS,
                **_responses_reasoning_kwargs(self.model_id),
            )

        # Drain any pending tool calls
        if tool_results:
            response = self.client.responses.create(
                model=self.model_id,
                previous_response_id=response.id,
                input=tool_results,
                tools=TOOLS,
                tool_choice="none",
                max_output_tokens=MAX_TOKENS,
                **_responses_reasoning_kwargs(self.model_id),
            )
            _, drain_text, drain_proof, _, _ = self._process_response(
                response, compiler, node_name, repo_retrieval, tool_calls_used
            )
            if drain_text:
                last_text = drain_text
            if drain_proof:
                best_proof_body = drain_proof

        # Ask for final answer if no proof tag found
        if not best_proof_body:
            response = self.client.responses.create(
                model=self.model_id,
                previous_response_id=response.id,
                input="Output your best proof: <lean4_proof>\nby\n  ...\n</lean4_proof>",
                max_output_tokens=MAX_TOKENS,
                **_responses_reasoning_kwargs(self.model_id),
            )
            _, last_text, best_proof_body, _, _ = self._process_response(
                response, compiler, node_name, repo_retrieval, tool_calls_used
            )

        # Probe negation if we couldn't prove it
        negation = self._probe_negation(compiler, node_name, response.id, MAX_TOKENS)
        if negation:
            return negation

        if best_proof_body:
            signal = _classify_failure([], last_text)
            return ProverResult(signal=signal, proof_body=best_proof_body,
                                analysis=last_text[:500])
        return ProverResult(signal=_classify_failure([], last_text), analysis=last_text[:500])

    # ------------------------------------------------------------------
    # Response processing
    # ------------------------------------------------------------------

    def _process_response(
        self,
        response,
        compiler: AbstractLeanCompiler,
        node_name: str,
        repo_retrieval,
        tool_calls_so_far: int,
        node_decl: str = "",
    ) -> tuple[list[dict], str, str, bool, list[str]]:
        tool_results: list[dict] = []
        last_text = ""
        best_proof = ""
        compiled_proof = ""   # proof body that actually compiled — never overwritten by message text
        any_compile_ok = False
        tools_called: list[str] = []
        turn = tool_calls_so_far + 1

        for item in response.output:
            if item.type == "function_call":
                fn = item.name
                args = json.loads(item.arguments)
                tools_called.append(fn)

                self.tracer.emit(TraceEvent(
                    kind="tool_call", thm_name=node_name, turn=turn,
                    call_id=item.call_id, tool_name=fn, args=args,
                ))

                if fn == "lean_compile":
                    proof_body = args.get("proof_body", "")
                    aux = args.get("aux_lemmas", "")
                    cr = compiler.check(proof_body, aux_lemmas=aux, node_decl=node_decl)
                    if cr.success:
                        result = "Compilation SUCCESSFUL. Proof is correct."
                        any_compile_ok = True
                        compiled_proof = proof_body
                        best_proof = proof_body
                    else:
                        errs = "\n".join(cr.errors)
                        result = f"Compilation FAILED.\n{errs}\n\nFix errors and call lean_compile again."

                elif fn == "repo_search" and repo_retrieval is not None:
                    hits = repo_retrieval.search(args["query"], args.get("k", 10))
                    result = "\n\n".join(h.format() for h in hits) or "No results in repo."

                elif fn == "mathlib_search":
                    hits = self.retrieval.search(args["query"], args.get("k", 10))
                    result = "\n\n".join(h.format() for h in hits) or "No results found."

                else:
                    result = f"Tool unavailable: {fn}"

                self.tracer.emit(TraceEvent(
                    kind="tool_result", thm_name=node_name, turn=turn,
                    call_id=item.call_id, tool_name=fn,
                    result=result[:500], ok=(fn == "lean_compile" and any_compile_ok),
                ))

                tool_results.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": result,
                })

            elif item.type == "message":
                for block in item.content:
                    text = getattr(block, "text", None)
                    if text:
                        last_text = text
                        extracted = _extract_proof_body(text)
                        # Only use message-extracted proof if no compile succeeded yet;
                        # compiled_proof is authoritative and must not be overwritten.
                        if extracted and not any_compile_ok:
                            best_proof = extracted
                        self.tracer.emit(TraceEvent(
                            kind="model_text", thm_name=node_name, turn=turn,
                            result=text[:500],
                        ))

        # Prefer the proof that actually compiled over anything extracted from text
        return tool_results, last_text, compiled_proof or best_proof, any_compile_ok, tools_called

    # ------------------------------------------------------------------
    # Negation probe (Section 4.3 / Figure 1)
    # ------------------------------------------------------------------

    def _probe_negation(
        self,
        compiler: AbstractLeanCompiler,
        node_name: str,
        previous_response_id: str,
        max_tokens: int,
    ) -> ProverResult | None:
        prompt = (
            f"You could not prove the statement. Try to show it is FALSE.\n"
            f"Prove `neg_{node_name}` showing `¬ (conclusion)` with the same hypotheses. "
            "Tactics: `omega`, `decide`, `norm_num`, `push_neg; linarith`, `simp`, `native_decide`.\n"
            "Call lean_compile. If it succeeds, the original statement is formally refuted."
        )
        response = self.client.responses.create(
            model=self.model_id,
            previous_response_id=previous_response_id,
            input=prompt,
            tools=TOOLS,
            tool_choice={"type": "function", "name": "lean_compile"},
            max_output_tokens=max_tokens,
            **_responses_reasoning_kwargs(self.model_id),
        )

        for _ in range(NEGATION_PROBE_CALLS):
            results: list[dict] = []
            for item in response.output:
                if item.type != "function_call":
                    continue
                args = json.loads(item.arguments)
                if item.name == "lean_compile":
                    cr = compiler.check(args.get("proof_body", ""))
                    if cr.success:
                        return ProverResult(
                            signal=ProofSignal.FORMALLY_NEGATED,
                            proof_body=args.get("proof_body", ""),
                            analysis=f"{node_name} is formally refuted — a proof of ¬(statement) was found.",
                            suggested_fix="The statement is mathematically FALSE. Revise it.",
                        )
                    output = f"Compilation FAILED.\n" + "\n".join(cr.errors)
                else:
                    output = "Tool unavailable in negation probe."
                results.append({"type": "function_call_output", "call_id": item.call_id, "output": output})

            if not results:
                break
            response = self.client.responses.create(
                model=self.model_id,
                previous_response_id=response.id,
                input=results,
                tools=TOOLS,
                tool_choice="auto",
                max_output_tokens=max_tokens,
                **_responses_reasoning_kwargs(self.model_id),
            )

        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_proof_body(text: str) -> str:
    import re
    m = re.search(r"<lean4_proof>(.*?)</lean4_proof>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _classify_failure(errors: list[str], analysis: str) -> ProofSignal:
    combined = " ".join(errors).lower() + analysis.lower()
    if any(p in combined for p in ["type mismatch", "false", "counterexample"]):
        return ProofSignal.STATEMENT_WRONG
    return ProofSignal.PROOF_TOO_HARD


# ---------------------------------------------------------------------------
# Backward-compat function (used by orchestrator.py)
# ---------------------------------------------------------------------------

def prove_node(
    node_name: str,
    canonical_stmt: str,
    parent_proofs: dict[str, str],
    compiler: AbstractLeanCompiler,
    retrieval: MathlibRetrieval,
    model: str = "gpt-4o",
    node_statement_nl: str = "",
    node_proof_sketch_nl: str = "",
    repo_retrieval=None,
    tracer=None,
    api_timeout_s: float = 120.0,
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
    prover = GoedelProver(model_id=model, retrieval=retrieval, tracer=tracer, api_timeout_s=api_timeout_s)
    return prover.prove_node(
        compiler=compiler,
        node_name=node_name,
        node_stmt=canonical_stmt,
        user_prompt=user_prompt,
        repo_retrieval=repo_retrieval,
    )

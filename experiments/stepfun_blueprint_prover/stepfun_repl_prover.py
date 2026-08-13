from __future__ import annotations

import asyncio
import copy
import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from blueprint_text import BLUEPRINT_PROOF_RE, extract_current_node_decl
from kimina_lean_compiler import CompilerResult, KiminaLeanCompiler
from node_context import NodeProblem


SYSTEM_PROMPT = (
    "You will be given an unsolved Lean 4 problem. Think carefully and work towards a solution. "
    "At any point, you may use the Lean 4 REPL to check your progress by enclosing your partial "
    "solution between <sketch> and </sketch>. The REPL feedback will be provided between <REPL> "
    "and </REPL>. Continue this process as needed until you arrive at a complete and correct solution."
)
STOP_MARKERS = ("<｜end▁of▁sentence｜>", "<|end_of_sentence|>")


@dataclass
class ProverOutcome:
    status: str
    proof_body: str = ""
    lean_errors: list[str] = field(default_factory=list)
    turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_seconds: float = 0.0
    trajectory: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def strip_completion_artifacts(text: str) -> str:
    candidate = text
    for marker in STOP_MARKERS:
        if marker in candidate:
            candidate = candidate.split(marker, 1)[0]
    think_end = candidate.rfind("</think>")
    if think_end >= 0:
        candidate = candidate[think_end + len("</think>"):]
    return candidate.strip()


def extract_sketch(text: str) -> str | None:
    start = text.find("<sketch>")
    end = text.find("</sketch>", start + len("<sketch>"))
    if start < 0 or end < 0:
        return None
    return text[start + len("<sketch>"):end].strip()


def _last_fenced_block(text: str) -> str | None:
    blocks = re.findall(r"```(?:lean4|lean)?\s*\n(.*?)\n```", text, re.I | re.S)
    return blocks[-1].strip() if blocks else None


def extract_proof_body(text: str) -> str | None:
    candidate = strip_completion_artifacts(text)
    block = _last_fenced_block(candidate)
    if block is not None:
        candidate = block
    matches = list(re.finditer(r":=\s*by\b", candidate))
    if matches:
        match = matches[-1]
        body = "by" + candidate[match.end():]
        return body.strip()
    match = re.search(r"(?m)^\s*by\b", candidate)
    if match:
        return candidate[match.start():].strip()
    return None


def repl_feedback(result: CompilerResult) -> str:
    try:
        raw = json.loads(result.raw_output) if result.raw_output else {}
        response = raw.get("response") or {}
        rows = response.get("results") or []
        if rows and isinstance(rows[0], dict):
            payload = copy.deepcopy(rows[0].get("response"))
            if isinstance(payload, dict):
                payload.pop("env", None)
                return json.dumps(payload, ensure_ascii=False)
    except (ValueError, TypeError):
        pass
    return json.dumps({
        "success": result.success,
        "errors": result.errors,
        "goals": result.goals,
        "warnings": result.warnings,
        "failure_kind": result.failure_kind,
    }, ensure_ascii=False)


def check_node_safely(
    compiler: KiminaLeanCompiler,
    proof_body: str,
    *,
    node_decl: str,
    parent_lemma_decls: str,
    header: str,
) -> CompilerResult:
    """Assemble a node without treating Lean backslashes as re.sub escapes."""
    if not header.strip():
        return CompilerResult(
            False,
            errors=["Node assembly rejected: node compilation requires an explicit blueprint header"],
            failure_kind="assembly",
        )
    decl_text = extract_current_node_decl(node_decl)
    decl_text, replacements = BLUEPRINT_PROOF_RE.subn(
        lambda _match: f":= {proof_body}", decl_text, count=1,
    )
    if replacements != 1:
        return CompilerResult(
            False,
            errors=[
                "Node assembly rejected: proof node declaration must contain exactly one "
                "`:= by sorry_using [...]` placeholder"
            ],
            failure_kind="assembly",
        )
    parts = [header.rstrip()]
    if parent_lemma_decls.strip():
        parts.append(parent_lemma_decls.strip())
    parts.append(decl_text.strip())
    return compiler.check("\n\n".join(parts) + "\n")


class StepFunReplProver:
    def __init__(self, *, client, tokenizer, compiler: KiminaLeanCompiler, config: dict[str, Any]):
        self.client = client
        self.tokenizer = tokenizer
        self.compiler = compiler
        self.model = str(config["name"])
        self.max_context = int(config["max_context_tokens"])
        self.temperature = float(config["temperature"])
        self.top_p = float(config["top_p"])
        self.top_k = int(config["top_k"])
        self.seed = int(config.get("seed", 42))
        self.stop_token_ids = list(config["stop_token_ids"])
        self.include_stop = bool(config.get("include_stop_str_in_output", True))
        self.api_semaphore = asyncio.Semaphore(int(config["api_concurrency"]))

    def initial_prompt(self, problem: NodeProblem) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"```lean4\n{problem.complete_lean.strip()}\n```"},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    def token_count(self, prompt: str) -> int:
        # The vLLM completions endpoint applies the tokenizer's normal special
        # token handling.  Counting without it underestimates every request by
        # one BOS token and makes max_tokens exceed the 40960 context by one.
        return len(self.tokenizer.encode(prompt, add_special_tokens=True))

    async def _complete(self, prompt: str, max_tokens: int):
        async with self.api_semaphore:
            return await self.client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                seed=self.seed,
                extra_body={
                    "top_k": self.top_k,
                    "stop_token_ids": self.stop_token_ids,
                    "include_stop_str_in_output": self.include_stop,
                    "skip_special_tokens": False,
                },
            )

    async def prove(self, problem: NodeProblem) -> ProverOutcome:
        started = time.monotonic()
        prompt = self.initial_prompt(problem)
        trajectory: list[dict[str, Any]] = []
        total_completion = 0
        turn = 0
        last_errors: list[str] = []
        while True:
            turn += 1
            prompt_tokens = self.token_count(prompt)
            remaining = self.max_context - prompt_tokens
            if remaining <= 0:
                return ProverOutcome(
                    "context_exhausted", lean_errors=last_errors, turns=turn - 1,
                    prompt_tokens=prompt_tokens, completion_tokens=total_completion,
                    elapsed_seconds=time.monotonic() - started, trajectory=trajectory,
                )
            request_started = time.monotonic()
            try:
                response = await self._complete(prompt, remaining)
            except Exception as exc:  # noqa: BLE001
                return ProverOutcome(
                    "infra_error", lean_errors=[f"LLM API {type(exc).__name__}: {exc}"],
                    turns=turn - 1, prompt_tokens=prompt_tokens,
                    completion_tokens=total_completion,
                    elapsed_seconds=time.monotonic() - started, trajectory=trajectory,
                )
            choice = response.choices[0]
            text = choice.text or ""
            usage = getattr(response, "usage", None)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            actual_prompt_tokens = int(getattr(usage, "prompt_tokens", prompt_tokens) or prompt_tokens)
            total_completion += completion_tokens
            stop_reason = getattr(choice, "stop_reason", None)
            if stop_reason is None:
                stop_reason = (getattr(choice, "model_extra", None) or {}).get("stop_reason")
            event: dict[str, Any] = {
                "turn": turn,
                "request_prompt_tokens": prompt_tokens,
                "api_prompt_tokens": actual_prompt_tokens,
                "max_tokens": remaining,
                "completion_tokens": completion_tokens,
                "finish_reason": getattr(choice, "finish_reason", None),
                "stop_reason": stop_reason,
                "text": text,
                "llm_elapsed_seconds": time.monotonic() - request_started,
            }
            trajectory.append(event)
            prompt += text
            sketch_stop = str(stop_reason) == "151666" or "</sketch>" in text
            if sketch_stop:
                sketch = extract_sketch(text)
                proof = extract_proof_body(sketch or "")
                if proof is None:
                    feedback = json.dumps({"repl_err": "unusable <sketch>; provide Lean proof code"})
                    event["extract_status"] = "failed"
                else:
                    lean_started = time.monotonic()
                    result = await asyncio.to_thread(
                        check_node_safely,
                        self.compiler,
                        proof,
                        node_decl=problem.node_decl,
                        parent_lemma_decls=problem.parent_lemma_decls,
                        header=problem.header,
                    )
                    last_errors = list(result.diagnostics)
                    feedback = repl_feedback(result)
                    event.update({
                        "extract_status": "proof_body",
                        "proof_body": proof,
                        "lean_success": result.success and not result.has_sorry,
                        "lean_errors": result.diagnostics,
                        "lean_failure_kind": result.failure_kind,
                        "lean_timings": result.timings,
                        "lean_elapsed_seconds": time.monotonic() - lean_started,
                        "repl_feedback": feedback,
                    })
                    if result.failure_kind == "infra":
                        return ProverOutcome(
                            "infra_error", lean_errors=last_errors, turns=turn,
                            prompt_tokens=actual_prompt_tokens,
                            completion_tokens=total_completion,
                            elapsed_seconds=time.monotonic() - started, trajectory=trajectory,
                        )
                prompt += f"\n<REPL>\n{feedback}\n</REPL>"
                continue
            if getattr(choice, "finish_reason", None) == "length":
                return ProverOutcome(
                    "output_truncated", lean_errors=last_errors, turns=turn,
                    prompt_tokens=actual_prompt_tokens, completion_tokens=total_completion,
                    elapsed_seconds=time.monotonic() - started, trajectory=trajectory,
                )
            proof = extract_proof_body(text)
            if proof is None:
                return ProverOutcome(
                    "extract_failed", lean_errors=["final output has no usable proof body"],
                    turns=turn, prompt_tokens=actual_prompt_tokens,
                    completion_tokens=total_completion,
                    elapsed_seconds=time.monotonic() - started, trajectory=trajectory,
                )
            lean_started = time.monotonic()
            result = await asyncio.to_thread(
                check_node_safely,
                self.compiler,
                proof,
                node_decl=problem.node_decl,
                parent_lemma_decls=problem.parent_lemma_decls,
                header=problem.header,
            )
            event.update({
                "extract_status": "proof_body",
                "proof_body": proof,
                "final_lean_success": result.success and not result.has_sorry,
                "final_lean_errors": result.diagnostics,
                "final_lean_failure_kind": result.failure_kind,
                "final_lean_timings": result.timings,
                "final_lean_elapsed_seconds": time.monotonic() - lean_started,
            })
            status = "solved" if result.success and not result.has_sorry else (
                "infra_error" if result.failure_kind == "infra" else "lean_error"
            )
            return ProverOutcome(
                status, proof_body=proof if status == "solved" else "",
                lean_errors=list(result.diagnostics), turns=turn,
                prompt_tokens=actual_prompt_tokens, completion_tokens=total_completion,
                elapsed_seconds=time.monotonic() - started, trajectory=trajectory,
            )

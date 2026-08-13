from __future__ import annotations

import asyncio
import time
from typing import Any

from kimina_lean_compiler import KiminaLeanCompiler
from node_context import NodeProblem
from stepfun_repl_prover import ProverOutcome, check_node_safely, extract_proof_body


GOEDEL_USER_PROMPT = """Complete the following Lean 4 code:

```lean4
{lean_code}```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof."""


def correction_prompt(errors: list[str]) -> str:
    diagnostics = "\n".join(errors) if errors else "The response did not contain a usable Lean proof."
    return f"""The previous Lean 4 candidate did not verify.

Lean compiler feedback:
```text
{diagnostics}
```

Revise the proof. Preserve the theorem statement and imports. First give a concise updated proof plan, then output the complete corrected Lean 4 code in a ```lean4 fenced block. Do not use `sorry` or `admit`."""


class GoedelSelfCorrectProver:
    """Goedel-Prover-V2 native generation followed by two Lean-feedback revisions."""

    def __init__(self, *, client, tokenizer, compiler: KiminaLeanCompiler, config: dict[str, Any]):
        self.client = client
        self.tokenizer = tokenizer
        self.compiler = compiler
        self.model = str(config["name"])
        self.max_context = int(config["max_context_tokens"])
        self.initial_max_tokens = int(config.get("initial_max_tokens", 32768))
        self.correction_max_tokens = int(config.get("correction_max_tokens", 8192))
        self.max_rounds = int(config.get("self_correction_rounds", 2)) + 1
        self.temperature = float(config.get("temperature", 0.6))
        self.top_p = float(config.get("top_p", 0.95))
        self.top_k = int(config.get("top_k", 20))
        self.seed = int(config.get("seed", 30))
        self.api_semaphore = asyncio.Semaphore(int(config["api_concurrency"]))

    def initial_messages(self, problem: NodeProblem) -> list[dict[str, Any]]:
        return [{
            "role": "user",
            "content": GOEDEL_USER_PROMPT.format(lean_code=problem.complete_lean.strip()),
        }]

    def token_count(self, messages: list[dict[str, Any]]) -> int:
        encoded = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
        )
        input_ids = encoded.get("input_ids") if hasattr(encoded, "get") else encoded
        return len(input_ids)

    async def _complete(self, messages: list[dict[str, Any]], max_tokens: int):
        async with self.api_semaphore:
            return await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                seed=self.seed,
                extra_body={"top_k": self.top_k},
            )

    async def prove(self, problem: NodeProblem) -> ProverOutcome:
        started = time.monotonic()
        messages = self.initial_messages(problem)
        trajectory: list[dict[str, Any]] = []
        total_completion = 0
        last_errors: list[str] = []
        for turn in range(1, self.max_rounds + 1):
            prompt_tokens = self.token_count(messages)
            remaining = self.max_context - prompt_tokens
            if remaining <= 0:
                return ProverOutcome(
                    "context_exhausted", lean_errors=last_errors, turns=turn - 1,
                    prompt_tokens=prompt_tokens, completion_tokens=total_completion,
                    elapsed_seconds=time.monotonic() - started, trajectory=trajectory,
                )
            round_cap = self.initial_max_tokens if turn == 1 else self.correction_max_tokens
            max_tokens = min(remaining, round_cap)
            request_started = time.monotonic()
            try:
                response = await self._complete(messages, max_tokens)
            except Exception as exc:  # noqa: BLE001
                return ProverOutcome(
                    "infra_error", lean_errors=[f"LLM API {type(exc).__name__}: {exc}"],
                    turns=turn - 1, prompt_tokens=prompt_tokens,
                    completion_tokens=total_completion,
                    elapsed_seconds=time.monotonic() - started, trajectory=trajectory,
                )
            choice = response.choices[0]
            message = choice.message
            text = message.content or ""
            reasoning = getattr(message, "reasoning_content", None)
            if reasoning is None:
                reasoning = (getattr(message, "model_extra", None) or {}).get("reasoning_content")
            usage = getattr(response, "usage", None)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            actual_prompt_tokens = int(getattr(usage, "prompt_tokens", prompt_tokens) or prompt_tokens)
            total_completion += completion_tokens
            proof = extract_proof_body(text)
            event: dict[str, Any] = {
                "turn": turn,
                "request_prompt_tokens": prompt_tokens,
                "api_prompt_tokens": actual_prompt_tokens,
                "max_tokens": max_tokens,
                "completion_tokens": completion_tokens,
                "finish_reason": getattr(choice, "finish_reason", None),
                "reasoning_content": reasoning,
                "text": text,
                "extract_status": "proof_body" if proof else "failed",
                "llm_elapsed_seconds": time.monotonic() - request_started,
            }
            trajectory.append(event)
            if proof is None:
                last_errors = ["response has no usable Lean proof body"]
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
                event.update({
                    "proof_body": proof,
                    "lean_success": result.success and not result.has_sorry,
                    "lean_errors": result.diagnostics,
                    "lean_failure_kind": result.failure_kind,
                    "lean_timings": result.timings,
                    "lean_elapsed_seconds": time.monotonic() - lean_started,
                })
                if result.success and not result.has_sorry:
                    return ProverOutcome(
                        "solved", proof_body=proof, turns=turn,
                        prompt_tokens=actual_prompt_tokens,
                        completion_tokens=total_completion,
                        elapsed_seconds=time.monotonic() - started, trajectory=trajectory,
                    )
                if result.failure_kind == "infra":
                    return ProverOutcome(
                        "infra_error", lean_errors=last_errors, turns=turn,
                        prompt_tokens=actual_prompt_tokens,
                        completion_tokens=total_completion,
                        elapsed_seconds=time.monotonic() - started, trajectory=trajectory,
                    )
            if turn == self.max_rounds:
                status = "output_truncated" if getattr(choice, "finish_reason", None) == "length" else (
                    "extract_failed" if proof is None else "lean_error"
                )
                return ProverOutcome(
                    status, lean_errors=last_errors, turns=turn,
                    prompt_tokens=actual_prompt_tokens,
                    completion_tokens=total_completion,
                    elapsed_seconds=time.monotonic() - started, trajectory=trajectory,
                )
            messages.extend([
                {
                    "role": "assistant",
                    "content": text,
                    "reasoning_content": reasoning or "",
                },
                {"role": "user", "content": correction_prompt(last_errors)},
            ])
        raise AssertionError("unreachable")

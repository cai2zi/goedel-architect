"""Phase 3: Blueprint refinement.

Uses the verbatim system prompt from Appendix C.3 of the paper.
Takes failed node diagnostics and produces a revised @[blueprint]-annotated Lean file.
"""
from __future__ import annotations

import re

from openai import OpenAI

from blueprint import Blueprint, _parse_blueprint, _reasoning_kwargs
from lean_compiler import AbstractLeanCompiler
from orchestrator import OrchestratorResult
from goedel_prompts import load, render
from prover import ProofSignal

REFINEMENT_SYSTEM_PROMPT = load("refinement_system")
REFINEMENT_USER_TEMPLATE = load("refinement_user")

# Appendix A specifies 262,144 (matches DeepSeek-V4-Flash's completion budget).
# OpenAI's chat.completions API hard-caps max_completion_tokens at 128,000
# regardless of model, and this is capped further to 32,000 to control cost.
MAX_TOKENS = 32_000
MAX_RETRIES = 8


def refine_blueprint(
    blueprint: Blueprint,
    orch_result: OrchestratorResult,
    compiler: AbstractLeanCompiler,
    model: str = "gpt-4o",
    repo_context: str | None = None,
) -> Blueprint:
    """
    Produce a revised blueprint by feeding failure diagnostics back to the LLM.

    Returns a new Blueprint with the revised Lean file.
    Uses the verbatim Appendix C.3 system prompt.
    """
    client = OpenAI()

    annotated_lean = _annotate_with_verdicts(blueprint, orch_result)
    messages = [
        {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
        {"role": "user", "content": _build_refinement_user_prompt(annotated_lean, repo_context)},
    ]

    for attempt in range(MAX_RETRIES):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=MAX_TOKENS,
            **_reasoning_kwargs(model),
        )
        content = response.choices[0].message.content
        lean_code = _extract_lean_code(content)

        result = compiler.check_blueprint(lean_code, blueprint.target_theorem)
        if result.success or result.validation_successful:
            return _parse_blueprint(lean_code, blueprint.target_theorem)

        # Feed compile errors back for next attempt
        error_feedback = "\n".join(result.errors) or result.raw_output[-2000:]
        messages.append({"role": "assistant", "content": content})
        messages.append({
            "role": "user",
            "content": (
                f"lean_compile reported errors (attempt {attempt + 1}/{MAX_RETRIES}):\n\n"
                f"{error_feedback}\n\n"
                "Fix the issues and call lean_compile again."
            ),
        })

    raise RuntimeError(f"Refinement failed after {MAX_RETRIES} attempts")


def _annotate_with_verdicts(blueprint: Blueprint, orch_result: OrchestratorResult) -> str:
    """
    Insert PROVED / UNPROVED / FORMALLY_NEGATED markers and diagnosis blocks
    into the blueprint Lean file, as expected by the refinement prompt.

    Formally-negated nodes (red in Figure 1) get a distinct marker and carry
    the machine-checked counterexample proof so the refinement LLM can fix the
    statement accordingly.
    """
    lines = blueprint.lean_file.splitlines()
    output_lines: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        output_lines.append(line)

        name_match = re.search(r"(lemma|theorem)\s+(\w+)", line)
        if name_match:
            name = name_match.group(2)
            nr = orch_result.node_results.get(name)
            if nr:
                if nr.result.signal == ProofSignal.SOLVED:
                    output_lines.append("-- PROVED")
                elif nr.result.signal == ProofSignal.FORMALLY_NEGATED:
                    # Distinct red-node marker (Section 4.3 / Figure 1)
                    output_lines.append("-- FORMALLY_NEGATED")
                    output_lines.append(nr.result.diagnosis_block(name))
                else:
                    output_lines.append("-- UNPROVED")
                    output_lines.append(nr.result.diagnosis_block(name))
        i += 1

    return "\n".join(output_lines)


def _build_refinement_user_prompt(annotated_lean: str, repo_context: str | None = None) -> str:
    return render(REFINEMENT_USER_TEMPLATE, annotated_lean=annotated_lean, repo_context=repo_context or "")


def _extract_lean_code(content: str) -> str:
    match = re.search(r"```(?:lean)?\n(.*?)```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content.strip()

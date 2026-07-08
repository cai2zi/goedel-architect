"""Phase 3: Blueprint refinement.

Uses the verbatim system prompt from Appendix C.3 of the paper.
Takes failed node diagnostics and produces a revised @[blueprint]-annotated Lean file.
"""
from __future__ import annotations

import re

from openai import OpenAI

from blueprint import (
    Blueprint,
    REPO_SEARCH_SUFFIX,
    _call_with_repo_search,
    _extract_lean_code,
    _parse_blueprint,
    _reasoning_kwargs,
)
from lean_compiler import AbstractLeanCompiler
from orchestrator import OrchestratorResult
from goedel_prompts import load, render
from prover import ProofSignal

REFINEMENT_SYSTEM_PROMPT = load("refinement_system")
REFINEMENT_USER_TEMPLATE = load("refinement_user")

# Appendix A specifies 262,144 (matches DeepSeek-V4-Flash's completion budget).
# OpenAI's chat.completions API hard-caps max_completion_tokens at 128,000
# regardless of model, and this is capped further to 64,000 to control cost.
MAX_TOKENS = 64_000
MAX_RETRIES = 8


def refine_blueprint(
    blueprint: Blueprint,
    orch_result: OrchestratorResult,
    compiler: AbstractLeanCompiler,
    model: str = "gpt-4o",
    repo_context: str | None = None,
    history: list[str] | None = None,
    iteration: int = 0,
    max_iterations: int = 0,
    repo_retrieval=None,
    tracer=None,
    thm_name: str = "",
) -> Blueprint:
    """
    Produce a revised blueprint by feeding failure diagnostics back to the LLM.

    Returns a new Blueprint with the revised Lean file.
    Uses the verbatim Appendix C.3 system prompt.

    history: mutable list of every prior round's annotated blueprint, owned by
        the caller (pipeline.py) and shared across the whole refinement loop
        for one theorem. Refinement renames nodes round to round, so there's
        no reliable way to detect "this is the same sub-goal that already
        failed twice under a different name" by matching identifiers - instead
        of building that matching heuristic, the full history is handed to the
        model so it can recognize repetition itself and decide whether to
        change strategy or accept a node as an unresolved gap, rather than
        cosmetically re-decomposing the same stuck problem every round.
    """
    client = OpenAI()

    annotated_lean = _annotate_with_verdicts(blueprint, orch_result)
    if history is not None:
        history.append(annotated_lean)
        prior_rounds = history[:-1]
    else:
        prior_rounds = []

    system_content = REFINEMENT_SYSTEM_PROMPT
    if repo_retrieval is not None:
        system_content = system_content.strip() + "\n" + REPO_SEARCH_SUFFIX
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": _build_refinement_user_prompt(
            annotated_lean, repo_context, prior_rounds=prior_rounds,
            iteration=iteration, max_iterations=max_iterations,
        )},
    ]

    last_error_feedback = ""
    for attempt in range(MAX_RETRIES):
        response = _call_with_repo_search(
            client, model, messages, repo_retrieval, _reasoning_kwargs(model), MAX_TOKENS,
            tracer=tracer, thm_name=thm_name, phase="phase3",
        )
        content = response.choices[0].message.content
        lean_code = _extract_lean_code(content)

        result = compiler.check_blueprint(lean_code, blueprint.target_theorem)
        if result.success:
            parsed = _parse_blueprint(lean_code, blueprint.target_theorem)
            if parsed.nodes:
                parsed.fully_validated = result.validated
                print(f"  [refine] attempt {attempt + 1}/{MAX_RETRIES}: check_blueprint OK", flush=True)
                return parsed
            # Compiles, but has zero @[blueprint]-annotated declarations - an
            # empty node set would make all_proved() vacuously true downstream
            # with no actual proof recorded, so this must be retried rather
            # than accepted (mirrors generate_blueprint's same guard).
            print(f"  [refine] attempt {attempt + 1}/{MAX_RETRIES}: check_blueprint OK but zero nodes, retrying", flush=True)
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": (
                    f"The file compiled, but contains no `@[blueprint ...]`-annotated "
                    f"declarations (attempt {attempt + 1}/{MAX_RETRIES}). Re-emit the "
                    "blueprint with proper annotations."
                ),
            })
            continue

        # Feed compile errors back for next attempt
        error_feedback = "\n".join(result.errors) or result.raw_output[-2000:]
        last_error_feedback = error_feedback
        print(f"  [refine] attempt {attempt + 1}/{MAX_RETRIES}: check_blueprint FAILED - "
              f"{error_feedback[:300]!r}", flush=True)
        messages.append({"role": "assistant", "content": content})
        messages.append({
            "role": "user",
            "content": (
                f"lean_compile reported errors (attempt {attempt + 1}/{MAX_RETRIES}):\n\n"
                f"{error_feedback}\n\n"
                "Fix the issues and call lean_compile again."
            ),
        })

    raise RuntimeError(
        f"Refinement failed after {MAX_RETRIES} attempts. "
        f"Last check_blueprint error:\n{last_error_feedback[-2000:]}"
    )


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
                elif nr.result.signal == ProofSignal.INFRA_ERROR:
                    # Infrastructure/tooling failure (timeout, exception), not
                    # a genuine proof-difficulty verdict - flagged distinctly
                    # so the refinement model doesn't treat it as evidence the
                    # sub-goal itself needs re-decomposing.
                    output_lines.append(
                        "-- INFRA_ERROR (infrastructure/tooling failure, not a "
                        "genuine proof-difficulty signal)"
                    )
                    output_lines.append(nr.result.diagnosis_block(name))
                else:
                    output_lines.append("-- UNPROVED")
                    output_lines.append(nr.result.diagnosis_block(name))
        i += 1

    return "\n".join(output_lines)


def _build_refinement_user_prompt(
    annotated_lean: str,
    repo_context: str | None = None,
    prior_rounds: list[str] | None = None,
    iteration: int = 0,
    max_iterations: int = 0,
) -> str:
    prior_rounds_text = ""
    if prior_rounds:
        blocks = [
            f"### Round {i + 1}\n\n```lean\n{text}\n```"
            for i, text in enumerate(prior_rounds)
        ]
        prior_rounds_text = "\n\n".join(blocks)

    round_info = ""
    if max_iterations:
        round_info = (
            f"This is refinement round {iteration + 1} of {max_iterations}. "
            f"{len(prior_rounds)} earlier round(s) already attempted this theorem "
            "(see 'Earlier rounds' below, if present)."
        )

    return render(
        REFINEMENT_USER_TEMPLATE,
        annotated_lean=annotated_lean,
        repo_context=repo_context or "",
        prior_rounds=prior_rounds_text,
        round_info=round_info,
    )

"""Phase 3: Blueprint refinement.

Uses the verbatim system prompt from Appendix C.3 of the paper.
Takes failed node diagnostics and produces a revised @[blueprint]-annotated Lean file.
"""
from __future__ import annotations

import re

from llm_client import make_client

from blueprint import (
    Blueprint,
    _call_blueprint_model,
    _emit_lean_check_result,
    _extract_lean_code,
    _max_tokens as _blueprint_max_tokens,
    _parse_blueprint,
    _reasoning_kwargs,
    format_phase2_contract_errors,
    phase2_contract_errors,
    phase2_standalone_contract_errors,
)
from kimina_lean_compiler import KiminaInfrastructureError, KiminaLeanCompiler
from orchestrator import OrchestratorResult
from goedel_prompts import load, render
from prover import ProofSignal

REFINEMENT_SYSTEM_PROMPT = load("refinement_system")
REFINEMENT_USER_TEMPLATE = load("refinement_user")

MAX_RETRIES = 8

# Cap how many earlier rounds are replayed into the refinement prompt. Full
# history is still kept in the checkpoint (round-count messaging stays
# accurate), but only the most recent N rounds are rendered as ### Round
# blocks - this bounds prompt growth on theorems needing many refinement
# rounds instead of re-sending every prior round's full annotated Lean file
# every time.
MAX_HISTORY_ROUNDS = 3

# Matches one @[blueprint ...] node's full text block (attribute + decl +
# body), used to locate and edit individual node blocks by name rather than
# scanning line by line - mirrors blueprint.py::_parse_blueprint's pattern.
_NODE_BLOCK_RE = re.compile(
    r'@\[blueprint\s*(.*?)\]\s*\n\s*(def|lemma|theorem|noncomputable def|abbrev)\s+(\w+)(.*?)(?=@\[blueprint|\Z)',
    re.DOTALL,
)
_PROOF_SKETCH_ATTR_RE = re.compile(r'\(proof\s*:=\s*/--.*?-/\)\s*', re.DOTALL)

# Recovers a compact (name, verdict) trail from an already-annotated round's
# text, used to summarize rounds too old to replay in full (see
# _summarize_dropped_rounds).
_VERDICT_RE = re.compile(
    r'(?:lemma|theorem)\s+(\w+).*?\n--\s*(PROVED|UNPROVED|FORMALLY_NEGATED|INFRA_ERROR)',
    re.DOTALL,
)


def refine_blueprint(
    blueprint: Blueprint,
    orch_result: OrchestratorResult,
    compiler: KiminaLeanCompiler,
    model: str = "labs-leanstral-1-5",
    history: list[str] | None = None,
    iteration: int = 0,
    max_iterations: int = 0,
    tracer=None,
    thm_name: str = "",
    max_retries: int = MAX_RETRIES,
    phase2_contract_check_concurrency: int = 1,
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
    client = make_client(model)

    annotated_lean = _annotate_with_verdicts(blueprint, orch_result)
    if history is not None:
        history.append(annotated_lean)
        prior_all = history[:-1]
        total_prior_rounds = len(prior_all)
        prior_rounds = prior_all[-MAX_HISTORY_ROUNDS:]
        dropped_rounds_summary = _summarize_dropped_rounds(
            prior_all[:-MAX_HISTORY_ROUNDS] if total_prior_rounds > MAX_HISTORY_ROUNDS else []
        )
    else:
        total_prior_rounds = 0
        prior_rounds = []
        dropped_rounds_summary = ""

    messages = [
        {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
        {"role": "user", "content": _build_refinement_user_prompt(
            annotated_lean, prior_rounds=prior_rounds,
            iteration=iteration, max_iterations=max_iterations,
            total_prior_rounds=total_prior_rounds,
            dropped_rounds_summary=dropped_rounds_summary,
        )},
    ]

    last_error_feedback = ""
    for attempt in range(max_retries):
        response = _call_blueprint_model(
            client,
            model,
            messages,
            _reasoning_kwargs(model),
            _blueprint_max_tokens(),
            tracer=tracer, thm_name=thm_name, phase="phase3",
        )
        content = response.choices[0].message.content
        lean_code = _extract_lean_code(content)

        result = compiler.check_blueprint(lean_code, blueprint.target_theorem)
        _emit_lean_check_result(
            tracer,
            thm_name=thm_name,
            phase="phase3",
            attempt=attempt + 1,
            target=blueprint.target_theorem,
            result=result,
        )
        if result.failure_kind == "infra":
            raise KiminaInfrastructureError(
                "\n".join(result.diagnostics) or result.raw_output[-2000:]
            )
        if result.success:
            parsed = _parse_blueprint(lean_code, blueprint.target_theorem)
            if parsed.nodes:
                contract_errors = phase2_contract_errors(parsed)
                if not contract_errors:
                    contract_errors = phase2_standalone_contract_errors(
                        parsed,
                        compiler,
                        concurrency=phase2_contract_check_concurrency,
                    )
                if contract_errors:
                    last_error_feedback = (
                        "The file compiled, but the blueprint is not usable by Phase 2:\n\n"
                        f"{format_phase2_contract_errors(contract_errors)}"
                    )
                    print(
                        f"  [refine] attempt {attempt + 1}/{max_retries}: "
                        "check_blueprint OK but phase2 contract FAILED",
                        flush=True,
                    )
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"{last_error_feedback}\n\n"
                            "Fix the Phase 2 standalone contract and re-emit the whole "
                            "blueprint. Avoid ambient declarations such as `variable`, "
                            "`section`, `namespace`, `axiom`, and `partial def`; make "
                            "parameters explicit and emit helper definitions as "
                            "`@[blueprint]` definition nodes."
                        ),
                    })
                    continue
                print(f"  [refine] attempt {attempt + 1}/{max_retries}: check_blueprint OK", flush=True)
                return parsed
            # Compiles, but has zero @[blueprint]-annotated declarations - an
            # A zero-node response cannot contain the required root proof node.
            print(f"  [refine] attempt {attempt + 1}/{max_retries}: check_blueprint OK but zero nodes, retrying", flush=True)
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": (
                    f"The file compiled, but contains no `@[blueprint ...]`-annotated "
                    f"declarations (attempt {attempt + 1}/{max_retries}). Re-emit the "
                    "blueprint with proper annotations."
                ),
            })
            continue

        # Feed compile errors back for next attempt
        error_feedback = "\n".join(result.diagnostics) or result.raw_output[-2000:]
        last_error_feedback = error_feedback
        print(f"  [refine] attempt {attempt + 1}/{max_retries}: check_blueprint FAILED - "
              f"{error_feedback[:300]!r}", flush=True)
        messages.append({"role": "assistant", "content": content})
        messages.append({
            "role": "user",
            "content": (
                f"lean_compile reported errors (attempt {attempt + 1}/{max_retries}):\n\n"
                f"{error_feedback}\n\n"
                "Fix the issues and call lean_compile again."
            ),
        })

    raise RuntimeError(
        f"Refinement failed after {max_retries} attempts. "
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
    lean_file = _strip_proof_sketch_for_solved(blueprint.lean_file, orch_result)
    lines = lean_file.splitlines()
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
                elif nr.result.signal == ProofSignal.BLOCKED_BY_DEPENDENCY:
                    # The prover intentionally did not attempt this node because
                    # an upstream dependency failed. Keep this separate from a
                    # genuine proof attempt that exhausted its tool budget.
                    output_lines.append("-- BLOCKED_BY_DEPENDENCY")
                    output_lines.append(nr.result.diagnosis_block(name))
                else:
                    output_lines.append("-- UNPROVED")
                    output_lines.append(nr.result.diagnosis_block(name))
        i += 1

    return "\n".join(output_lines)


def _strip_proof_sketch_for_solved(lean_file: str, orch_result: OrchestratorResult) -> str:
    """Drop the natural-language `(proof := /-- ... -/)` sketch attribute for
    already-solved nodes before they're replayed into the refinement prompt.

    A solved node only needs to be usable as a citable fact going forward
    (its name + statement) - the NL plan that was used to derive its
    now-verified proof is dead weight once the proof is done, and this text
    gets re-sent every round a solved node still appears in the blueprint.
    """
    def replace(m: re.Match) -> str:
        attrs_block, kind_kw, name, rest = m.group(1), m.group(2), m.group(3), m.group(4)
        nr = orch_result.node_results.get(name)
        if nr and nr.result.signal == ProofSignal.SOLVED:
            new_attrs = _PROOF_SKETCH_ATTR_RE.sub("", attrs_block).strip()
            return f"@[blueprint {new_attrs}]\n{kind_kw} {name}{rest}"
        return m.group(0)

    return _NODE_BLOCK_RE.sub(replace, lean_file)


def _summarize_dropped_rounds(dropped_rounds: list[str]) -> str:
    """One compact line per round dropped from the replayed history window,
    recovering just (node name -> verdict) pairs from that round's already-
    annotated text - keeps a cheap "don't repeat this" memory trace for
    rounds too old to afford replaying in full (see MAX_HISTORY_ROUNDS).
    """
    lines = []
    for i, text in enumerate(dropped_rounds, start=1):
        verdicts = _VERDICT_RE.findall(text)
        if not verdicts:
            continue
        verdict_str = ", ".join(f"{name}={v}" for name, v in verdicts)
        lines.append(f"Round {i}: {verdict_str}")
    return "\n".join(lines)


def _build_refinement_user_prompt(
    annotated_lean: str,
    prior_rounds: list[str] | None = None,
    iteration: int = 0,
    max_iterations: int = 0,
    total_prior_rounds: int | None = None,
    dropped_rounds_summary: str = "",
) -> str:
    total_prior_rounds = total_prior_rounds if total_prior_rounds is not None else len(prior_rounds or [])

    prior_rounds_text = ""
    if prior_rounds:
        # prior_rounds may be truncated to the most recent MAX_HISTORY_ROUNDS -
        # number blocks by their true round number, not their index here, so
        # the model isn't misled into thinking round 5 was actually round 1.
        start_round = total_prior_rounds - len(prior_rounds) + 1
        blocks = [
            f"### Round {start_round + i}\n\n```lean\n{text}\n```"
            for i, text in enumerate(prior_rounds)
        ]
        prior_rounds_text = "\n\n".join(blocks)

    round_info = ""
    if max_iterations:
        omitted = total_prior_rounds - len(prior_rounds)
        omitted_note = (
            f" ({omitted} earliest round(s) omitted here for brevity)" if omitted > 0 else ""
        )
        round_info = (
            f"This is refinement round {iteration + 1} of {max_iterations}. "
            f"{total_prior_rounds} earlier round(s) already attempted this theorem{omitted_note} "
            "(see 'Earlier rounds' below, if present)."
        )
        if dropped_rounds_summary:
            round_info += (
                "\n\nCompact per-node verdict trail for the earliest rounds not shown "
                f"in full above:\n{dropped_rounds_summary}"
            )

    return render(
        REFINEMENT_USER_TEMPLATE,
        annotated_lean=annotated_lean,
        prior_rounds=prior_rounds_text,
        round_info=round_info,
    )

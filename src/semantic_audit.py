"""Optional one-request semantic audit for COT-grounded Lean blueprints.

This module is deliberately independent of the main pipeline.  Callers may
route only selected examples through it without changing Blueprint generation
or proof scheduling.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Literal

from llm_client import chat_completion_with_retry, make_client
from semantic_fidelity import _strip_lean_comments
from tracer import TraceEvent


AuditMode = Literal["risk", "full"]
SEMANTIC_AUDIT_FLAG_RE = re.compile(r"\[\[SEMANTIC_AUDIT=(PASS|FAIL)\]\]")
_COT_CLAIM_ID_RE = re.compile(r"\[COT_CLAIM\s+((?:S\d{3}\.)?C\d{3,})(?:\s+[^\]]*)?\]")
_CLAIMS_LINE_RE = re.compile(r"\[\[CLAIMS=([^\]]*)\]\]")
_CLAIM_ENTRY_RE = re.compile(r"((?:S\d{3}\.)?C\d{3,}):(OK|MISSING|MISMATCH)")
_BLUEPRINT_PROSE_FIELD_RE = re.compile(
    r"\((?:statement|proof)\s*:=\s*"
    r"(?:/--.*?-/|\"(?:\\.|[^\"\\])*\")\)\s*",
    re.DOTALL,
)
_SORRY_USING_BODY_RE = re.compile(
    r":=\s*by\s+sorry_using\s*\[[^\]]*\]",
    re.DOTALL,
)


SYSTEM_PROMPT = """You are a proposition-coverage auditor, not a proof judge.

Compare the original problem, claimed answer, numbered source COT, and the
formal-only Lean view. Use this mechanical decision procedure:

1. Inventory every mathematical proposition explicitly asserted by each
   `[COT_CLAIM ...]`, including false, contradictory, and unsupported claims.
   `[COT_CONTEXT ...]` is narration-only and requires no Lean proposition.
2. Find a Lean proposition or definition body that states each claim about the
   same objects, bindings, assumptions, quantifiers, relations, constants,
   branches, and polarity.
3. Check that the root asks the original problem's question about the same
   object and preserves the COT's claimed final answer.
4. PASS exactly when this proposition inventory is covered without a
   substantive replacement, omission, weakening, strengthening, or silent
   correction. Mathematical truth and provability are irrelevant.

The Lean view intentionally removes prose, proof bodies, and dependency lists;
every displayed `by sorry` means "this source proposition is unproved here".
A deterministic checker has already verified step IDs, root reachability, and
graph closure. Never judge whether a node is used, whether its parents prove it,
whether the root logically follows, or whether an extra dependency is needed.
Do not repair or re-grade the COT. A faithfully translated mathematically wrong
COT must PASS.

An unsupported jump is preserved by stating its conclusion as its own Lean
lemma with `by sorry`; no missing-justification lemma is required. If the COT
counts a restricted family and then explicitly writes `N = K`, the required
formal node is simply `N = K`. Do not demand an unstated necessity, converse,
or set equivalence. A weaker node `restrictedCount = K` is not enough.
Coverage may pass through transparent definitions: `restrictedCount := 27^3`
together with `N = restrictedCount` exactly states `N = 27^3` and MUST NOT be
rejected merely because the equality is split across two declarations.

Concrete calibration: if the source separately asserts (a) a single-cube
biconditional, (b) an all-27-divisible sufficiency statement for a sum, (c) the
unsupported total-count jump, and (d) the final result, then a view containing
`cube_divisibility_condition : ... ↔ ...`,
`all_divisible_by_27_implies_sum_divisible : ... → ...`,
`total_valid_triples : N = 27^3 := by sorry`, and a root about that same `N`
MUST PASS. Requiring another "only if" lemma is itself an audit error.

Near-miss calibration: if that source contains the one-variable biconditional
and only an all-three-divisible sufficiency statement, a Blueprint that omits
the one-variable proposition and instead states
`sumDivisible ↔ allThreeDivisible` MUST FAIL. The added aggregate converse is
not the source's one-variable claim, even when the total-count gap is retained.

FAIL only for a formal mismatch: changed object identity/binding, truth
condition, quantifier/polarity, number, premise/filter, relation direction,
case/branch, final conclusion/answer, or a substantive source claim absent from
formal Lean. `True`, self-equality, an unconstrained existential, a hard-coded
answer definition, or an added premise are failures when they replace source
content. A source `assume` may become a hypothesis, but a source `let x := v`
must retain the binding to `v`. Splitting/merging helpers, equivalent notation,
specialization used by the COT, and declaration order are not failures.

Lean identifier names and conventional subscripts carry no semantics. Never
infer "inner", "outer", incidence, tangency, validity, or another relation from
a name alone; require the relation in a formal type/body, and follow the
source's explicit equations even when its naming is unconventional. Conversely,
numeric definitions such as `sphereRadius := 11` do not formalize the source's
sphere, torus, tangency, incidence, or two-circle constraints unless formal
relations connect those objects and quantities.

If inconsistent source prose says "outer = R+a = 9" and "inner = R-a = 3"
but then explicitly computes `r_i - r_o = 9 - 3`, a formal `r_i := 9` and
`r_o := 3` preserves the source's algebra. Do not swap them based on conventional
subscript meanings. Missing formal tangency or circle relations may still FAIL.

The FIRST LINE must be exactly one marker:
[[SEMANTIC_AUDIT=PASS]]
[[SEMANTIC_AUDIT=FAIL]]

The SECOND LINE must be exactly one compact inventory containing every supplied
claim ID once, in source order, with status OK, MISSING, or MISMATCH:
`[[CLAIMS=S001.C001:OK,S002.C001:MISSING]]`.

PASS is legal only when every status is OK. FAIL requires at least one MISSING
or MISMATCH. For FAIL, output at most two diagnostic lines after the inventory,
each `S004.C002: ...` and at most 45 words. Each line must cite one explicit
source claim and its mismatching Lean signature. Do not duplicate a root cause.
Do not output chain-of-thought, proof critique, a rewritten proof, or discussion.
Decide before emitting the first token, never revise the inventory afterward,
and stop after the permitted diagnostic lines. Emit exactly one decision marker
and one complete inventory."""


@dataclass(frozen=True)
class ParsedSemanticAudit:
    passed: bool
    flag: Literal["PASS", "FAIL"]
    diagnostics: str
    claim_statuses: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class SemanticAuditResult:
    passed: bool
    flag: Literal["PASS", "FAIL"]
    diagnostics: str
    raw_content: str
    reasoning_content: str
    model: str
    mode: AuditMode
    finish_reason: str | None
    truncated: bool
    request_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    claim_statuses: tuple[tuple[str, str], ...] = ()


class SemanticAuditFormatError(ValueError):
    """The model response did not contain one unambiguous final audit flag."""

    def __init__(
        self,
        reason: str,
        *,
        raw_content: str,
        markers: tuple[str, ...] = (),
    ) -> None:
        super().__init__(f"semantic audit format error: {reason}")
        self.reason = reason
        self.raw_content = raw_content
        self.markers = markers


def semantic_audit_formal_view(blueprint_lean: str) -> str:
    """Keep formal statements/bodies while hiding proof-grading distractions."""
    without_prose_fields = _BLUEPRINT_PROSE_FIELD_RE.sub("", blueprint_lean)
    without_comments = _strip_lean_comments(without_prose_fields)
    return _SORRY_USING_BODY_RE.sub(":= by sorry", without_comments).strip()


def _max_tokens() -> int:
    value = int(os.environ.get("GOEDEL_SEMANTIC_AUDIT_MAX_TOKENS", "1024"))
    if value <= 0:
        raise ValueError("GOEDEL_SEMANTIC_AUDIT_MAX_TOKENS must be positive")
    return value


def build_semantic_audit_messages(
    numbered_cot: str,
    blueprint_lean: str,
    *,
    mode: AuditMode,
    informal_statement: str = "",
    claimed_answer: str = "",
) -> list[dict[str, str]]:
    if mode not in {"risk", "full"}:
        raise ValueError("semantic audit mode must be 'risk' or 'full'")
    problem_text = informal_statement.strip() or "(not supplied by caller)"
    answer_text = claimed_answer.strip() or "(not supplied by caller)"
    formal_blueprint = semantic_audit_formal_view(blueprint_lean)
    user = f"""Audit mode: {mode}

## Original informal problem statement

{problem_text}

## Claimed answer from the original COT

{answer_text}

## Numbered original COT

{numbered_cot}

## Formal-only Blueprint Lean audit view

```lean
{formal_blueprint}
```

Audit all step-to-node mappings in this single response. Confirm that the root
and answer-bearing nodes refer to the exact object asked for in the original
problem and preserve the source COT's claimed answer. Remember that an
incorrect source solution must still PASS when it is translated faithfully;
do not repair or re-grade it. Emit the unique decision marker as the first
line and the complete compact claim inventory as the second line, before any
concise diagnostics."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _claim_ids(numbered_cot: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_COT_CLAIM_ID_RE.findall(numbered_cot)))


def parse_semantic_audit(
    content: str,
    *,
    expected_claim_ids: tuple[str, ...] = (),
) -> ParsedSemanticAudit:
    matches = list(SEMANTIC_AUDIT_FLAG_RE.finditer(content))
    markers = tuple(match.group(1) for match in matches)
    if not matches:
        raise SemanticAuditFormatError(
            "flag missing", raw_content=content, markers=markers,
        )
    if len(set(markers)) > 1:
        raise SemanticAuditFormatError(
            "conflicting flags", raw_content=content, markers=markers,
        )
    if len(matches) != 1:
        raise SemanticAuditFormatError(
            "duplicate flag", raw_content=content, markers=markers,
        )
    match = matches[0]
    lines = content.splitlines()
    first_line = lines[0] if lines else ""
    if SEMANTIC_AUDIT_FLAG_RE.fullmatch(first_line.strip()) is None:
        raise SemanticAuditFormatError(
            "flag is not the first line", raw_content=content, markers=markers,
        )
    flag = match.group(1)
    claim_statuses: tuple[tuple[str, str], ...] = ()
    diagnostic_lines = lines[1:]
    if expected_claim_ids:
        if len(lines) < 2 or _CLAIMS_LINE_RE.fullmatch(lines[1].strip()) is None:
            raise SemanticAuditFormatError(
                "complete claim inventory is not the second line",
                raw_content=content,
                markers=markers,
            )
        inventory = _CLAIMS_LINE_RE.fullmatch(lines[1].strip())
        assert inventory is not None
        payload = inventory.group(1)
        entries = payload.split(",") if payload else []
        parsed_entries: list[tuple[str, str]] = []
        for entry in entries:
            entry_match = _CLAIM_ENTRY_RE.fullmatch(entry.strip())
            if entry_match is None:
                raise SemanticAuditFormatError(
                    "malformed claim inventory entry",
                    raw_content=content,
                    markers=markers,
                )
            parsed_entries.append((entry_match.group(1), entry_match.group(2)))
        actual_ids = tuple(claim_id for claim_id, _status in parsed_entries)
        if actual_ids != expected_claim_ids:
            raise SemanticAuditFormatError(
                "claim inventory is incomplete, duplicated, or out of source order",
                raw_content=content,
                markers=markers,
            )
        statuses = tuple(status for _claim_id, status in parsed_entries)
        if flag == "PASS" and any(status != "OK" for status in statuses):
            raise SemanticAuditFormatError(
                "PASS contains a non-OK claim status",
                raw_content=content,
                markers=markers,
            )
        if flag == "FAIL" and all(status == "OK" for status in statuses):
            raise SemanticAuditFormatError(
                "FAIL contains no missing or mismatched claim",
                raw_content=content,
                markers=markers,
            )
        claim_statuses = tuple(parsed_entries)
        diagnostic_lines = lines[2:]
    diagnostics = "\n".join(diagnostic_lines).strip()
    return ParsedSemanticAudit(
        passed=flag == "PASS",
        flag=flag,
        diagnostics=diagnostics,
        claim_statuses=claim_statuses,
    )


def _message_parts(response: Any) -> tuple[str, str, str | None]:
    choice = response.choices[0]
    message = choice.message
    content = str(getattr(message, "content", None) or "")
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning is None and getattr(message, "model_extra", None):
        reasoning = message.model_extra.get("reasoning_content")
    return content, str(reasoning or ""), getattr(choice, "finish_reason", None)


def _request_id(response: Any) -> str:
    return str(
        getattr(response, "request_id", None)
        or getattr(response, "_request_id", None)
        or getattr(response, "id", None)
        or ""
    )


def _usage(response: Any) -> tuple[int, int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    if prompt is None:
        prompt = getattr(usage, "input_tokens", 0)
    if completion is None:
        completion = getattr(usage, "output_tokens", 0)
    prompt = int(prompt or 0)
    completion = int(completion or 0)
    total = int(getattr(usage, "total_tokens", 0) or prompt + completion)
    return prompt, completion, total


def run_semantic_audit(
    model: str,
    numbered_cot: str,
    blueprint_lean: str,
    *,
    mode: AuditMode = "risk",
    informal_statement: str = "",
    claimed_answer: str = "",
    client: Any | None = None,
    tracer=None,
    thm_name: str = "",
    phase: str = "semantic_audit",
    max_tokens: int | None = None,
) -> SemanticAuditResult:
    """Run one batched semantic-audit completion and return a strict result.

    Transport retries are handled by the shared retry layer; no format-repair
    request is issued, so a successful API call consumes exactly one model
    completion.
    """
    messages = build_semantic_audit_messages(
        numbered_cot,
        blueprint_lean,
        mode=mode,
        informal_statement=informal_statement,
        claimed_answer=claimed_answer,
    )
    expected_claim_ids = _claim_ids(numbered_cot)
    token_budget = _max_tokens() if max_tokens is None else int(max_tokens)
    if token_budget <= 0:
        raise ValueError("max_tokens must be positive")
    resolved_client = client if client is not None else make_client(model)
    response = chat_completion_with_retry(
        resolved_client,
        tracer=tracer,
        thm_name=thm_name,
        phase=phase,
        model_id=model,
        operation="semantic_audit",
        trace_args={
            "mode": mode,
            "informal_statement_chars": len(informal_statement),
            "claimed_answer_chars": len(claimed_answer),
            "claim_count": len(expected_claim_ids),
        },
        model=model,
        messages=messages,
        temperature=0,
        max_completion_tokens=token_budget,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content, reasoning, finish_reason = _message_parts(response)
    prompt_tokens, completion_tokens, total_tokens = _usage(response)
    request_id = _request_id(response)
    truncated = str(finish_reason or "").lower() == "length"

    if tracer is not None:
        tracer.emit(TraceEvent(
            kind="llm_usage",
            thm_name=thm_name,
            args={
                "phase": phase,
                "model": model,
                "operation": "semantic_audit",
                "mode": mode,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        ))
        tracer.emit(TraceEvent(
            kind="llm_response",
            thm_name=thm_name,
            result=content,
            args={
                "phase": phase,
                "model": model,
                "operation": "semantic_audit",
                "mode": mode,
                "request_id": request_id,
                "finish_reason": finish_reason,
                "truncated": truncated,
                "reasoning_content": reasoning,
            },
        ))

    if truncated:
        raise SemanticAuditFormatError(
            "response was truncated; marker is not a complete audit decision",
            raw_content=content,
            markers=tuple(match.group(1) for match in SEMANTIC_AUDIT_FLAG_RE.finditer(content)),
        )
    parsed = parse_semantic_audit(content, expected_claim_ids=expected_claim_ids)
    return SemanticAuditResult(
        passed=parsed.passed,
        flag=parsed.flag,
        diagnostics=parsed.diagnostics,
        claim_statuses=parsed.claim_statuses,
        raw_content=content,
        reasoning_content=reasoning,
        model=model,
        mode=mode,
        finish_reason=finish_reason,
        truncated=truncated,
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )

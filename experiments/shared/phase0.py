from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from lean_compiler import LeanCompiler
from llm_client import make_client


SYSTEM_PROMPT = """You are a Lean 4 formalizer for contest math problems.
Output exactly one Lean theorem statement and nothing else.
Do not include imports, comments, explanations, examples, or proof content.
The theorem must use the requested theorem name.
The theorem must end with ':= by'.
Formalize the natural-language problem together with the proposed final answer.
Do not output a vacuous theorem such as True or a theorem unrelated to the problem."""


USER_TEMPLATE = """Requested theorem name:
{theorem_name}

Problem:
{question}

Candidate final answer:
{candidate_answer}

Natural-language solution:
{nl_proof}

Return only one Lean theorem statement ending with ':= by'."""


REPAIR_TEMPLATE = """The previous theorem statement failed Lean checking.

Previous statement:
{theorem_stmt}

Lean errors:
{errors}

Return a corrected Lean theorem statement using theorem name {theorem_name}.
Do not include imports or proof content. End with ':= by'."""


@dataclass
class Phase0Result:
    theorem_stmt: str
    success: bool
    error: str
    attempts: int


def _extract_code(text: str) -> str:
    fenced = re.findall(r"```(?:lean|lean4)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced[0]
    return text.strip()


def normalize_theorem_statement(text: str, theorem_name: str) -> str:
    code = _extract_code(text)
    code = "\n".join(
        line for line in code.splitlines()
        if not line.strip().startswith("import ") and not line.strip().startswith("--")
    ).strip()
    match = re.search(r"\btheorem\s+([A-Za-z_][A-Za-z0-9_']*)", code)
    if match:
        code = code[:match.start()] + code[match.start():]
        code = re.sub(r"\btheorem\s+[A-Za-z_][A-Za-z0-9_']*", f"theorem {theorem_name}", code, count=1)

    by_idx = code.find(":= by")
    if by_idx >= 0:
        return code[: by_idx + len(":= by")].strip()

    assign_idx = code.find(":=")
    if assign_idx >= 0:
        code = code[:assign_idx].rstrip()
    return f"{code.rstrip()} := by".strip()


def _lean_check_code(theorem_stmt: str) -> str:
    stmt = theorem_stmt.strip()
    if not stmt.endswith(":= by"):
        stmt = f"{stmt} := by"
    return "import Mathlib\nimport Aesop\n\n" + stmt + "\n  sorry\n"


def _check_theorem(theorem_stmt: str) -> tuple[bool, str]:
    compiler = LeanCompiler()
    result = compiler._run_lean(_lean_check_code(theorem_stmt))
    if result.get("success"):
        return True, ""
    return False, str(result.get("errors") or result.get("stderr") or result)


def formalize_candidate(
    *,
    question: str,
    candidate_answer: str,
    nl_proof: str,
    theorem_name: str,
    model: str,
    max_attempts: int = 3,
) -> Phase0Result:
    if not candidate_answer:
        return Phase0Result("", False, "empty canonical_extracted_answer", 0)

    client = make_client()
    max_tokens = int(os.environ.get("GOEDEL_PHASE0_MAX_TOKENS", "4096"))
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                theorem_name=theorem_name,
                question=question,
                candidate_answer=candidate_answer,
                nl_proof=nl_proof or "",
            ),
        },
    ]
    last_stmt = ""
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or ""
        theorem_stmt = normalize_theorem_statement(content, theorem_name)
        last_stmt = theorem_stmt
        ok, error = _check_theorem(theorem_stmt)
        if ok:
            return Phase0Result(theorem_stmt, True, "", attempt)
        last_error = error
        messages.append({"role": "assistant", "content": theorem_stmt})
        messages.append({
            "role": "user",
            "content": REPAIR_TEMPLATE.format(
                theorem_name=theorem_name,
                theorem_stmt=theorem_stmt,
                errors=error[:12000],
            ),
        })
    return Phase0Result(last_stmt, False, last_error, max_attempts)

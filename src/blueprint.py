"""Phase 1: Blueprint generation.

Calls the LLM with the verbatim system prompt from the paper (prompts/blueprint_system.md)
and validates the resulting @[blueprint]-annotated Lean file via LeanArchitect.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

from lean_compiler import AbstractLeanCompiler, LeanCompiler, CompilerResult
from goedel_prompts import load, render


def _reasoning_kwargs(model: str) -> dict:
    """Return reasoning_effort kwarg for models that support it (gpt-5.x series)."""
    if model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3") or model.startswith("o4"):
        return {"reasoning_effort": "low"}
    return {}

BLUEPRINT_SYSTEM_PROMPT = load("blueprint_system")
BLUEPRINT_USER_TEMPLATE = load("blueprint_user")

# Appendix A specifies 262,144 (matches DeepSeek-V4-Flash's completion budget).
# OpenAI's chat.completions API hard-caps max_completion_tokens at 128,000
# regardless of model, and this is capped further to 32,000 to control cost.
MAX_TOKENS = 32_000
MAX_RETRIES = 8


@dataclass
class BlueprintNode:
    name: str
    kind: str  # "definition" | "lemma" | "theorem"
    statement: str
    proof_sketch: str
    dependencies: list[str] = field(default_factory=list)
    lean_declaration: str = ""


@dataclass
class Blueprint:
    nodes: list[BlueprintNode]
    lean_file: str  # full compilable @[blueprint]-annotated Lean file
    target_theorem: str

    def node_by_name(self, name: str) -> BlueprintNode | None:
        return next((n for n in self.nodes if n.name == name), None)

    def nodes_by_name(self) -> dict[str, BlueprintNode]:
        return {n.name: n for n in self.nodes}

    def dependency_order(self) -> list[BlueprintNode]:
        """Topological order (definitions first, theorem last)."""
        ordered: list[BlueprintNode] = []
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            node = self.node_by_name(name)
            if node:
                for dep in node.dependencies:
                    visit(dep)
                ordered.append(node)

        for node in self.nodes:
            visit(node.name)
        return ordered


def generate_blueprint(
    theorem_stmt: str,
    nl_proof: str | None = None,
    model: str = "gpt-4o",
    compiler: AbstractLeanCompiler | None = None,
    repo_context: str | None = None,
) -> Blueprint:
    """
    Generate a @[blueprint]-annotated Lean dependency graph for `theorem_stmt`.

    Uses the verbatim system prompt from Appendix C.1 of the paper.
    Validates via lean_compile after each LLM attempt (up to MAX_RETRIES).
    """
    client = OpenAI()

    user_content = _build_user_prompt(theorem_stmt, nl_proof, repo_context)
    messages = [
        {"role": "system", "content": BLUEPRINT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    last_lean_code = None
    for attempt in range(MAX_RETRIES):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=MAX_TOKENS,
            **_reasoning_kwargs(model),
        )
        lean_code = _extract_lean_code(response.choices[0].message.content)
        last_lean_code = lean_code

        if compiler is not None:
            target = _extract_target_name(lean_code, theorem_stmt)
            result = compiler.check_blueprint(lean_code, target)
            if result.success or result.validation_successful:
                return _parse_blueprint(lean_code, _extract_target_name(lean_code, theorem_stmt))
            # Feed errors back to the model for the next attempt
            error_feedback = "\n".join(result.errors) or result.raw_output[-2000:]
            messages.append({"role": "assistant", "content": response.choices[0].message.content})
            messages.append({
                "role": "user",
                "content": (
                    f"lean_compile reported errors (attempt {attempt + 1}/{MAX_RETRIES}):\n\n"
                    f"{error_feedback}\n\n"
                    "Fix the issues and call lean_compile again."
                ),
            })
        else:
            return _parse_blueprint(lean_code, _extract_target_name(lean_code, theorem_stmt))

    # All attempts failed compilation — use the last generated blueprint anyway.
    # Phase 2/3 will encounter and surface type errors during node proving.
    if last_lean_code:
        return _parse_blueprint(last_lean_code, _extract_target_name(last_lean_code, theorem_stmt))
    raise RuntimeError(f"Blueprint generation failed after {MAX_RETRIES} attempts")


def _build_user_prompt(theorem_stmt: str, nl_proof: str | None, repo_context: str | None = None) -> str:
    return render(BLUEPRINT_USER_TEMPLATE, theorem_stmt=theorem_stmt, nl_proof=nl_proof or "", repo_context=repo_context or "")


def _extract_lean_code(content: str) -> str:
    """Extract the Lean code block from the LLM response."""
    match = re.search(r"```(?:lean)?\n(.*?)```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If the model didn't use a code fence, treat the whole response as Lean
    return content.strip()


def _parse_blueprint(lean_code: str, target_theorem: str) -> Blueprint:
    """
    Parse @[blueprint]-annotated Lean code into a Blueprint datastructure.

    Extracts node names, kinds, statements, proof sketches, and sorry_using deps.
    """
    nodes: list[BlueprintNode] = []
    # Match @[blueprint ...] blocks followed by a declaration
    pattern = re.compile(
        r'@\[blueprint\s*(.*?)\]\s*\n\s*(def|lemma|theorem|noncomputable def|abbrev)\s+(\w+)(.*?)(?=@\[blueprint|\Z)',
        re.DOTALL,
    )
    for m in pattern.finditer(lean_code):
        attrs_block = m.group(1)
        kind_kw = m.group(2).strip()
        name = m.group(3)
        rest = m.group(4)

        kind = "definition" if kind_kw in ("def", "noncomputable def", "abbrev") else kind_kw

        statement = _extract_attr(attrs_block, "statement")
        proof_sketch = _extract_attr(attrs_block, "proof")

        # Extract sorry_using [...] dependencies
        dep_match = re.search(r"sorry_using\s*\[([^\]]*)\]", rest)
        deps = [d.strip() for d in dep_match.group(1).split(",") if d.strip()] if dep_match else []

        nodes.append(BlueprintNode(
            name=name,
            kind=kind,
            statement=statement,
            proof_sketch=proof_sketch,
            dependencies=deps,
            lean_declaration=m.group(0),
        ))

    return Blueprint(nodes=nodes, lean_file=lean_code, target_theorem=target_theorem)


def _extract_attr(attrs: str, key: str) -> str:
    match = re.search(rf"\({key}\s*:=\s*/--\s*(.*?)\s*-/\)", attrs, re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_target_name(lean_code: str, fallback: str) -> str:
    """Extract the main theorem name from the blueprint Lean code.

    The main theorem is the last `theorem` declaration (which must equal the
    targeted identifier per the blueprint system prompt).
    """
    matches = re.findall(r"\btheorem\s+(\w+)", lean_code)
    if matches:
        return matches[-1]
    # Fallback: extract identifier from the theorem statement
    m = re.search(r"\btheorem\s+(\w+)", fallback)
    return m.group(1) if m else "main_theorem"

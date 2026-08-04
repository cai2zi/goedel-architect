"""Phase 1: Blueprint generation.

Calls the LLM with the verbatim system prompt from the paper (prompts/blueprint_system.md)
and validates the resulting @[blueprint]-annotated Lean file via LeanArchitect.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
import os
import re
from pathlib import Path

from blueprint_text import (
    BLUEPRINT_DECL_KW as _BLUEPRINT_DECL_KW,
    BLUEPRINT_PROOF_RE,
    extract_current_node_decl,
    extract_blueprint_signature,
    lemma_to_theorem,
    proof_body_to_decl_suffix,
    strip_blueprint_attr,
)
from kimina_lean_compiler import (
    CompileRequest,
    CompilerResult,
    KiminaInfrastructureError,
    KiminaLeanCompiler,
    MATHLIB_HEADER,
)
from llm_client import chat_completion_with_retry, make_client
from goedel_prompts import load, render
from tracer import TraceEvent


def _reasoning_kwargs(model: str) -> dict:
    """Return reasoning_effort kwarg for models that support it (gpt-5.x series)."""
    if model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3") or model.startswith("o4"):
        return {"reasoning_effort": "low"}
    return {}

BLUEPRINT_SYSTEM_PROMPT = load("blueprint_system")
BLUEPRINT_USER_TEMPLATE = load("blueprint_user")
ROBUSTPA_BLUEPRINT_SYSTEM_PROMPT = load("robustpa_blueprint_system")
ROBUSTPA_BLUEPRINT_USER_TEMPLATE = load("robustpa_blueprint_user")

# Appendix A specifies 262,144 (matches DeepSeek-V4-Flash's completion budget).
# Read at call time so experiment YAML environment settings apply after import.
def _max_tokens() -> int:
    return int(os.environ.get("GOEDEL_BLUEPRINT_MAX_TOKENS", "262144"))


MAX_RETRIES = 8


class BlueprintGenerationError(RuntimeError):
    """Terminal Phase-1 failure with the last unusable candidate attached."""

    def __init__(
        self,
        message: str,
        *,
        last_candidate: str = "",
        diagnostics: list[str] | None = None,
        attempt: int = 0,
        finish_reason: str | None = None,
        failure_stage: str = "model_output",
    ) -> None:
        super().__init__(message)
        self.last_candidate = last_candidate
        self.diagnostics = list(diagnostics or [])
        self.attempt = attempt
        self.finish_reason = finish_reason
        self.failure_stage = failure_stage


def _emit_usage(tracer, thm_name: str, phase: str, model: str, response) -> None:
    """Log token usage from a chat.completions/responses API response, if a
    tracer was given. `response.usage` is present on both APIs but with
    different field names, so normalize to prompt/completion/total."""
    if tracer is None:
        return
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    if prompt is None:
        prompt = getattr(usage, "input_tokens", 0)
    if completion is None:
        completion = getattr(usage, "output_tokens", 0)
    total = getattr(usage, "total_tokens", None) or (prompt + completion)
    tracer.emit(TraceEvent(
        kind="llm_usage",
        thm_name=thm_name,
        args={
            "phase": phase, "model": model,
            "prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": total,
        },
    ))


def _tool_calls_payload(message) -> list[dict]:
    return [
        {
            "id": tc.id,
            "type": getattr(tc, "type", "function"),
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            },
        }
        for tc in message.tool_calls or []
    ]


def _emit_llm_response(
    tracer,
    *,
    thm_name: str,
    phase: str,
    model: str,
    response,
    attempt: int,
    turn: int,
) -> None:
    if tracer is None:
        return
    choice = response.choices[0]
    msg = choice.message
    tracer.emit(TraceEvent(
        kind="llm_response",
        thm_name=thm_name,
        turn=turn,
        result=msg.content or "",
        args={
            "phase": phase,
            "model": model,
            "attempt": attempt,
            "finish_reason": getattr(choice, "finish_reason", None),
            "tool_calls": _tool_calls_payload(msg),
        },
    ))


def _emit_lean_check_result(
    tracer,
    *,
    thm_name: str,
    phase: str,
    attempt: int,
    target: str,
    result: CompilerResult,
) -> None:
    if tracer is None:
        return
    tracer.emit(TraceEvent(
        kind="lean_check_result",
        thm_name=thm_name,
        args={
            "phase": phase,
            "attempt": attempt,
            "target": target,
            "errors": result.errors,
            "warnings": result.warnings,
            "goals": result.goals,
            "raw_output": result.raw_output,
            "failure_kind": result.failure_kind,
        },
        ok=result.success,
    ))


def _append_assistant_turn(messages: list[dict], response) -> None:
    """Replay only text; Phase 1/3 do not expose tools."""
    msg = response.choices[0].message
    messages.append({
        "role": "assistant",
        "content": msg.content or "No valid Lean blueprint was emitted.",
    })


def _call_blueprint_model(
    client,
    model: str,
    messages: list[dict],
    reasoning_kwargs: dict,
    max_tokens: int,
    tracer=None,
    thm_name: str = "",
    phase: str = "",
    attempt: int = 0,
):
    response = chat_completion_with_retry(
        client,
        tracer=tracer,
        thm_name=thm_name,
        phase=phase,
        model_id=model,
        operation="blueprint_generate",
        trace_args={"attempt": attempt, "turn": 1},
        model=model, messages=messages, max_completion_tokens=max_tokens, **reasoning_kwargs,
    )
    _emit_usage(tracer, thm_name, phase, model, response)
    _emit_llm_response(
        tracer,
        thm_name=thm_name,
        phase=phase,
        model=model,
        response=response,
        attempt=attempt,
        turn=1,
    )
    return response


@dataclass
class BlueprintNode:
    name: str
    kind: str  # "definition" | "lemma" | "theorem"
    statement: str
    proof_sketch: str
    dependencies: list[str] = field(default_factory=list)
    lean_declaration: str = ""

    def signature(self) -> str:  # 针对单个 node，提取为类似 theorem l1 (n : ℕ) : n + 0 = n
        """Strip the @[blueprint ...] attribute and sorry_using proof body,
        returning just the declaration up to (not including) ':='.

        Used to re-declare an already-proved dependency as a real, standalone
        lemma (signature + its actual proof) so sibling nodes can reference it
        by name instead of hitting "unknown identifier" - proven dependencies
        are otherwise only ever shown to the model as prompt text, never
        actually compiled into scope.
        """
        return extract_blueprint_signature(self.lean_declaration)

    def full_declaration(self) -> str:
        """Return this node's complete declaration without ``@[blueprint]``.

        Unlike :meth:`signature`, this deliberately preserves a definition's
        outer ``:=`` and its entire right-hand side.  Attribute removal happens
        before declaration detection, so declaration-like words in blueprint
        comments cannot become part of the returned Lean code.
        """
        return extract_current_node_decl(self.lean_declaration)

    def cache_key(self) -> str:
        """Declaration shape plus dependencies for cache staleness checks.

        Proof nodes use their signature; definitions use the complete
        declaration so an RHS change is visible. `signature()` alone only
        covers the text before `:=`, so a node
        whose sorry_using [...] dependency list changes (text AFTER `:=`)
        while its exposed statement stays byte-identical would otherwise be
        invisible to staleness checks, even though its cached proof was
        spliced together with the OLD set of sibling declarations in scope.
        """
        declaration_shape = (
            self.full_declaration() if self.kind == "definition" else self.signature()
        )
        return declaration_shape + "\x00deps:" + ",".join(sorted(self.dependencies))


def render_solved_declaration(node: BlueprintNode, proof_body: str) -> str:
    """Render a solved node as Lean code suitable for dependency context.

    Definitions already carry their executable body in the validated
    blueprint, so cached proof text is intentionally ignored.  Proof nodes
    are reconstructed from their signature and the proof accepted by Lean.
    """
    if node.kind == "definition":
        return node.full_declaration()
    return f"{node.signature()} {proof_body_to_decl_suffix(proof_body)}"


@dataclass
class Blueprint:
    nodes: list[BlueprintNode]
    lean_file: str  # full compilable @[blueprint]-annotated Lean file
    target_theorem: str
    phase2_header: str = MATHLIB_HEADER

    def node_by_name(self, name: str) -> BlueprintNode | None:
        return next((n for n in self.nodes if n.name == name), None)

    def nodes_by_name(self) -> dict[str, BlueprintNode]:
        return {n.name: n for n in self.nodes}

    def dependency_order(self) -> list[BlueprintNode]:
        """Topological order (definitions first, theorem last)."""
        node_map = self.nodes_by_name()
        ordered: list[BlueprintNode] = []
        visited: set[str] = set()
        visiting: set[str] = set()
        stack: list[str] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            if name in visiting:
                cycle_start = stack.index(name) if name in stack else 0
                cycle = stack[cycle_start:] + [name]
                raise ValueError(f"Blueprint dependency cycle: {' -> '.join(cycle)}")
            node = node_map.get(name)
            if node is None:
                return
            visiting.add(name)
            stack.append(name)
            for dep in node.dependencies:
                if dep in node_map:
                    visit(dep)
            stack.pop()
            visiting.remove(name)
            visited.add(name)
            ordered.append(node)

        for node in self.nodes:
            visit(node.name)
        return ordered


def _safe_phase2_header(lean_code: str) -> str:
    """Extract the leading commands Phase 2 may safely preserve."""
    imports: list[str] = []
    other: list[str] = []
    in_block_comment = False
    for raw_line in lean_code.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if in_block_comment:
            if "-/" in stripped:
                in_block_comment = False
            continue
        if stripped.startswith("/-"):
            if "-/" not in stripped[2:]:
                in_block_comment = True
            continue
        if stripped.startswith("--"):
            continue
        if stripped.startswith("import "):
            imports.append(stripped)
            continue
        if stripped.startswith("open ") or stripped.startswith("open scoped "):
            other.append(stripped)
            continue
        if stripped.startswith("set_option "):
            other.append(stripped)
            continue
        break

    if not any(line == "import Mathlib" for line in imports):
        imports.insert(0, "import Mathlib")
    if not any(line == "import Architect" for line in imports):
        insert_at = 1 if imports and imports[0] == "import Mathlib" else len(imports)
        imports.insert(insert_at, "import Architect")
    if not any(line.startswith("set_option autoImplicit ") for line in other):
        other.insert(0, "set_option autoImplicit false")
    return "\n".join(imports + other).rstrip() + "\n\n"


def _transitive_node_deps(node: BlueprintNode, blueprint: Blueprint) -> set[str]:
    seen: set[str] = set()
    stack = list(node.dependencies)
    while stack:
        dep = stack.pop()
        if dep in seen:
            continue
        seen.add(dep)
        dep_node = blueprint.node_by_name(dep)
        if dep_node:
            stack.extend(dep_node.dependencies)
    return seen


def _phase2_preflight_file(blueprint: Blueprint, node: BlueprintNode) -> str:
    parts = [blueprint.phase2_header.rstrip()]
    parts.extend(
        definition.full_declaration()
        for definition in blueprint.nodes
        if definition.kind == "definition" and definition.name != node.name
    )
    ancestor_deps = _transitive_node_deps(node, blueprint)
    parts.extend(
        dep_node.full_declaration()
        for dep_node in blueprint.dependency_order()
        if dep_node.kind != "definition"
        and dep_node.name in ancestor_deps
    )
    parts.append(node.full_declaration())
    return "\n\n".join(part.strip() for part in parts if part.strip()) + "\n"


def phase2_standalone_contract_errors(
    blueprint: Blueprint,
    compiler: KiminaLeanCompiler,
    *,
    limit: int = 12,
    concurrency: int = 1,
) -> list[str]:
    """Compile proof nodes as Phase 2 would see them before accepting a blueprint."""
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")

    nodes = [
        node for node in blueprint.nodes
        if node.kind in {"lemma", "theorem"}
    ]
    results = compiler.check_many(
        [
            CompileRequest(
                _phase2_preflight_file(blueprint, node),
                allow_sorry=True,
                request_id=f"phase2-contract-{index}-{node.name}",
            )
            for index, node in enumerate(nodes)
        ],
        batch_concurrency=concurrency,
    )
    errors: list[str] = []
    for node, result in zip(nodes, results, strict=True):
        if result.failure_kind == "infra":
            message = "\n".join(result.diagnostics) or result.raw_output[-2000:]
            raise KiminaInfrastructureError(message)
        if result.success:
            continue
        message = "\n".join(result.diagnostics) or result.raw_output[-2000:]
        errors.append(
            f"phase2_standalone_failed: node `{node.name}` does not compile when "
            f"assembled as a standalone Phase 2 goal.\n{message}"
        )
    if limit:
        errors = errors[:limit]
    return errors


def phase2_contract_errors(blueprint: Blueprint) -> list[str]:
    """Return structural errors that would make Phase 2 node proving invalid."""
    errors: list[str] = []
    for node in blueprint.nodes:
        if node.kind not in {"lemma", "theorem"}:
            continue
        current_decl = extract_current_node_decl(node.lean_declaration)
        placeholder_count = len(BLUEPRINT_PROOF_RE.findall(current_decl))
        if placeholder_count == 0:
            errors.append(
                f"missing_sorry_using_placeholder: proof node `{node.name}` must contain "
                "`:= by sorry_using [...]`, not a completed proof or plain `sorry`."
            )
        elif placeholder_count > 1:
            errors.append(
                f"multiple_sorry_using_placeholders: proof node `{node.name}` contains "
                f"{placeholder_count} `sorry_using` placeholders; expected exactly one."
            )
    try:
        blueprint.dependency_order()
    except ValueError as exc:
        errors.append(f"dependency_cycle: {exc}")
    return errors


def phase2_contract_error_counts(errors: list[str]) -> dict[str, int]:
    return dict(Counter(error.split(":", 1)[0] for error in errors))


def format_phase2_contract_errors(errors: list[str], limit: int = 12) -> str:
    shown = errors[:limit]
    suffix = "" if len(errors) <= limit else f"\n... and {len(errors) - limit} more"
    return "\n".join(f"- {error}" for error in shown) + suffix


def generate_blueprint(
    theorem_stmt: str,
    nl_proof: str | None = None,
    model: str = "labs-leanstral-1-5",
    *,
    compiler: KiminaLeanCompiler,
    tracer=None,
    thm_name: str = "",
    max_retries: int = MAX_RETRIES,
    phase2_contract_check_concurrency: int = 1,
) -> Blueprint:
    """
    Generate a @[blueprint]-annotated Lean dependency graph for `theorem_stmt`.

    Uses the verbatim system prompt from Appendix C.1 of the paper.
    Validates via lean_compile after each LLM attempt (up to max_retries).

    """
    client = make_client(model)
    user_content = _build_user_prompt(theorem_stmt, nl_proof)
    messages = [
        {"role": "system", "content": BLUEPRINT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    last_lean_code = None
    for attempt in range(max_retries):
        response = _call_blueprint_model(
            client, model, messages, _reasoning_kwargs(model), _max_tokens(),
            tracer=tracer, thm_name=thm_name, phase="phase1", attempt=attempt + 1,
        )
        lean_code = _extract_lean_code(response.choices[0].message.content)
        last_lean_code = lean_code

        target = _extract_target_name(lean_code, theorem_stmt)
        result = compiler.check_blueprint(lean_code, target)
        _emit_lean_check_result(
            tracer,
            thm_name=thm_name,
            phase="phase1",
            attempt=attempt + 1,
            target=target,
            result=result,
        )
        if result.failure_kind == "infra":
            raise KiminaInfrastructureError(
                "\n".join(result.diagnostics) or result.raw_output[-2000:]
            )
        if result.success:
            parsed = _parse_blueprint(lean_code, target)
            if parsed.nodes:
                contract_errors = phase2_contract_errors(parsed)
                if not contract_errors:
                    contract_errors = phase2_standalone_contract_errors(
                        parsed,
                        compiler,
                        concurrency=phase2_contract_check_concurrency,
                    )
                if contract_errors:
                    _append_assistant_turn(messages, response)
                    messages.append({
                        "role": "user",
                        "content": (
                            f"The file compiled, but the blueprint is not usable by Phase 2 "
                            f"(attempt {attempt + 1}/{max_retries}):\n\n"
                            f"{format_phase2_contract_errors(contract_errors)}\n\n"
                            "Fix the blueprint contract and re-emit the whole file. Every "
                            "`lemma` and `theorem` blueprint node must end with exactly one "
                            "`:= by sorry_using [...]` placeholder; do not provide completed "
                            "proofs or plain `sorry` bodies in blueprint proof nodes. "
                            "Definitions may keep executable bodies. The dependency graph "
                            "must be acyclic."
                        ),
                    })
                    continue
                return parsed
            _append_assistant_turn(messages, response)
            messages.append({
                "role": "user",
                "content": (
                    f"The file compiled, but contains no `@[blueprint ...]`-annotated "
                    f"declarations (attempt {attempt + 1}/{max_retries}). You must "
                    "annotate the target theorem (and any helper lemmas) with "
                    "`@[blueprint ...]` and give each a `sorry_using [...]` proof body. "
                    "Re-emit the blueprint with proper annotations."
                ),
            })
            continue
        error_feedback = "\n".join(result.diagnostics) or result.raw_output[-2000:]
        _append_assistant_turn(messages, response)
        messages.append({
            "role": "user",
            "content": (
                f"lean_compile reported errors (attempt {attempt + 1}/{max_retries}):\n\n"
                f"{error_feedback}\n\n"
                "Fix the issues and call lean_compile again."
            ),
        })
    raise RuntimeError(
        f"Blueprint generation failed after {max_retries} attempts "
        "without a validated blueprint"
    )


def generate_blueprint_from_informal(
    informal_statement: str,
    informal_proof: str | None,
    target_name: str,
    model: str = "labs-leanstral-1-5",
    *,
    compiler: KiminaLeanCompiler,
    tracer=None,
    thm_name: str = "",
    max_retries: int = MAX_RETRIES,
    phase2_contract_check_concurrency: int = 1,
) -> Blueprint:
    """Generate and strictly validate a blueprint from informal text only.

    Unlike generate_blueprint(), this entry point has no formal Lean theorem
    signature to preserve. The model must formalize the main theorem itself,
    but the theorem identifier is fixed by target_name so downstream
    checkpointing, validation, and scoring remain stable.
    """
    client = make_client(model)
    messages = [
        {"role": "system", "content": ROBUSTPA_BLUEPRINT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": render(
                ROBUSTPA_BLUEPRINT_USER_TEMPLATE,
                target_name=target_name,
                informal_statement=informal_statement,
                informal_proof=informal_proof or "",
            ),
        },
    ]
    
    last_error_feedback = ""
    last_candidate = ""
    last_diagnostics: list[str] = []
    last_finish_reason: str | None = None
    last_failure_stage = "model_output"
    for attempt in range(max_retries):
        response = _call_blueprint_model(
            client,
            model,
            messages,
            reasoning_kwargs=_reasoning_kwargs(model),
            max_tokens=_max_tokens(),
            tracer=tracer,
            thm_name=thm_name,
            phase="phase1",
            attempt=attempt + 1,
        )
        choice = response.choices[0]
        lean_code = _extract_lean_code(choice.message.content)
        last_candidate = lean_code
        last_finish_reason = getattr(choice, "finish_reason", None)
        emitted_target = _extract_target_name(lean_code, "")
        if emitted_target != target_name:
            last_error_feedback = (
                f"The main theorem must be named `{target_name}`, but the "
                f"latest output's final theorem is `{emitted_target or '<missing>'}`."
            )
            last_diagnostics = [last_error_feedback]
            last_failure_stage = "blueprint_contract"
            _append_assistant_turn(messages, response)
            messages.append({
                "role": "user",
                "content": (
                    f"{last_error_feedback}\n\n"
                    "Re-emit the whole Lean file with exactly one main theorem "
                    f"named `{target_name}`."
                ),
            })
            continue

        result = compiler.check_blueprint(lean_code, target_name)
        _emit_lean_check_result(
            tracer,
            thm_name=thm_name,
            phase="phase1",
            attempt=attempt + 1,
            target=target_name,
            result=result,
        )
        if result.failure_kind == "infra":
            raise KiminaInfrastructureError(
                "\n".join(result.diagnostics) or result.raw_output[-2000:]
            )
        if result.success:
            try:
                parsed = _parse_blueprint(lean_code, target_name)
            except Exception as exc:  # noqa: BLE001
                parsed = None
                last_error_feedback = f"Blueprint parsing failed: {type(exc).__name__}: {exc}"
                last_failure_stage = "parse"
            if parsed is None:
                pass
            elif not parsed.nodes:
                last_error_feedback = (
                    "The file compiled, but contains no `@[blueprint ...]`-annotated declarations."
                )
                last_failure_stage = "parse"
            else:
                contract_errors = phase2_contract_errors(parsed)
                if not contract_errors:
                    contract_errors = phase2_standalone_contract_errors(
                        parsed,
                        compiler,
                        concurrency=phase2_contract_check_concurrency,
                    )
                if not contract_errors:
                    return parsed
                last_error_feedback = (
                    "The file compiled, but the blueprint is not usable by Phase 2:\n\n"
                    f"{format_phase2_contract_errors(contract_errors)}"
                )
                last_failure_stage = "blueprint_contract"
        else:
            last_error_feedback = "\n".join(result.diagnostics) or result.raw_output[-2000:]
            last_failure_stage = "lean_check"
        last_diagnostics = list(result.diagnostics) or [last_error_feedback]

        _append_assistant_turn(messages, response)
        messages.append({
            "role": "user",
            "content": (
                f"lean_compile reported errors (attempt {attempt + 1}/{max_retries}):\n\n"
                f"{last_error_feedback}\n\n"
                "Fix the issues and call lean_compile again."
            ),
        })

    message = (
        f"Informal blueprint generation failed after {max_retries} attempts. "
        f"Last error:\n{last_error_feedback[-2000:]}"
    )
    raise BlueprintGenerationError(
        message,
        last_candidate=last_candidate,
        diagnostics=last_diagnostics,
        attempt=max_retries,
        finish_reason=last_finish_reason,
        failure_stage=last_failure_stage,
    )


def _build_user_prompt(theorem_stmt: str, nl_proof: str | None) -> str:
    return render(BLUEPRINT_USER_TEMPLATE, theorem_stmt=theorem_stmt, nl_proof=nl_proof or "")


# Matches the first line that looks like real Lean source, used to strip a
# leaked non-Lean preamble (e.g. a model hallucinating a tool-call-style tag
# like `<lean_compile>` instead of a code fence) when there's no fence to
# delimit the code block.
_LEAN_START_RE = re.compile(
    r"^\s*(?:import\b|@\[blueprint\b|theorem\b|lemma\b|noncomputable\s+def\b|def\b|abbrev\b)",
    re.MULTILINE,
)


def _extract_lean_code(content: str | None) -> str:
    """Extract the Lean code block from the LLM response."""
    content = content or ""
    fenced_blocks = [
        block.strip()
        for block in re.findall(r"```(?:lean|lean4)?\s*\n(.*?)```", content, re.DOTALL)
        if block.strip()
    ]
    if fenced_blocks:
        blueprint_blocks = [
            block for block in fenced_blocks
            if "@[blueprint" in block and re.search(r"\btheorem\b", block)
        ]
        if blueprint_blocks:
            return blueprint_blocks[-1]
        leanish_blocks = [block for block in fenced_blocks if _LEAN_START_RE.search(block)]
        if leanish_blocks:
            return leanish_blocks[-1]
        return fenced_blocks[-1]
    # No fence - the model may still have prefixed its response with
    # non-Lean text (a leaked tag, an apology, etc.). Start at the first
    # line that looks like real Lean rather than treating the raw response
    # as Lean verbatim.
    start_match = _LEAN_START_RE.search(content)
    if start_match:
        return content[start_match.start():].strip()
    return content.strip()


def _parse_blueprint(lean_code: str, target_theorem: str) -> Blueprint:
    """
    Parse @[blueprint]-annotated Lean code into a Blueprint datastructure.

    Extracts node names, kinds, statements, proof sketches, and sorry_using deps.
    """
    nodes: list[BlueprintNode] = []
    # Match @[blueprint ...] blocks followed by a declaration
    pattern = re.compile(
        rf"@\[blueprint\s*(.*?)\]\s*\n\s*({_BLUEPRINT_DECL_KW})\s+"
        rf"(\w+)(.*?)(?=@\[blueprint|\Z)",
        re.DOTALL,
    )
    for m in pattern.finditer(lean_code):
        attrs_block = m.group(1)
        kind_kw = " ".join(m.group(2).split())
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

    return Blueprint(
        nodes=nodes,
        lean_file=lean_code,
        target_theorem=target_theorem,
        phase2_header=_safe_phase2_header(lean_code),
    )


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

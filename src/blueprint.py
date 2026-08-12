"""Blueprint representation, parsing, and Phase 2 contract validation."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
import re
import threading
import time
from typing import Any

from blueprint_text import (
    BLUEPRINT_DECL_KW as _BLUEPRINT_DECL_KW,
    BLUEPRINT_PROOF_RE,
    extract_current_node_decl,
    extract_blueprint_signature,
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
from llm_client import chat_completion_with_retry
from semantic_fidelity import SemanticIssue
from goedel_prompts import load
from tracer import TraceEvent


def _reasoning_kwargs(model: str) -> dict:
    """Return reasoning_effort kwarg for models that support it (gpt-5.x series)."""
    if model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3") or model.startswith("o4"):
        return {"reasoning_effort": "low"}
    return {}

ROBUSTPA_BLUEPRINT_SYSTEM_PROMPT = load("robustpa_blueprint_system")
ROBUSTPA_BLUEPRINT_USER_TEMPLATE = load("robustpa_blueprint_user")



def _render_step_grounded_proof(cot_manifest_json: str, *, include_ir: bool) -> str:
    from cot_blueprint_refine.formal_steps import decode_formal_step_manifest
    del include_ir
    manifest = decode_formal_step_manifest(cot_manifest_json)
    return "\n\n".join(
        f"[COT_STEP {step['step_id']}]\n{step['source_text']}\n[/COT_STEP {step['step_id']}]"
        for step in manifest["steps"]
    )


def _enabled_semantic_issues(
    issues: list[SemanticIssue],
    *,
    require_step_ids: bool,
    static_gate: bool,
) -> list[SemanticIssue]:
    """Filter semantic checks for downstream refinement callers."""
    if static_gate:
        return issues
    if require_step_ids:
        return [issue for issue in issues if issue.category == "binding"]
    return []


def _emit_semantic_check(
    tracer,
    *,
    thm_name: str,
    phase: str,
    attempt: int,
    issues: list[SemanticIssue],
    turn: int | None = None,
    blocking_issues: list[SemanticIssue] | None = None,
) -> None:
    if tracer is None:
        return
    all_errors = [issue for issue in issues if issue.severity == "error"]
    effective_blocking = all_errors if blocking_issues is None else blocking_issues
    blocking_count = len(effective_blocking)
    warning_count = sum(issue.severity == "warning" for issue in issues)
    tracer.emit(TraceEvent(
        kind="blueprint_semantic_check",
        thm_name=thm_name,
        ok=blocking_count == 0,
        args={
            "phase": phase,
            "attempt": attempt,
            "turn": turn,
            "issue_count": len(issues),
            "error_count": len(all_errors),
            "blocking_count": blocking_count,
            "deferred_error_count": len(all_errors) - blocking_count,
            "warning_count": warning_count,
            "issues": [issue.to_dict() for issue in issues],
        },
    ))

# Blueprint generation failures carry the last complete candidate for audit.
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
        candidate_history: list[str] | None = None,
        candidate_labels: list[str] | None = None,
        validation_details: dict[str, Any] | None = None,
        generation_history: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.last_candidate = last_candidate
        self.diagnostics = list(diagnostics or [])
        self.attempt = attempt
        self.finish_reason = finish_reason
        self.failure_stage = failure_stage
        self.candidate_history = list(candidate_history or [])
        self.candidate_labels = list(candidate_labels or [])
        self.validation_details = dict(validation_details or {})
        self.generation_history = list(generation_history or [])


@lru_cache(maxsize=2)
def _load_phase1_tokenizer_unlocked(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


_PHASE1_TOKENIZER_LOAD_LOCK = threading.Lock()


def _load_phase1_tokenizer(path: str):
    # Hugging Face's lazy module imports are not safe when many Phase-1 worker
    # threads perform the first import concurrently.  Serialize only cache
    # misses/reads; tokenization itself remains concurrent.
    with _PHASE1_TOKENIZER_LOAD_LOCK:
        return _load_phase1_tokenizer_unlocked(path)



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
    payload = []
    for tc in message.tool_calls or []:
        arguments = tc.function.arguments
        if tc.function.name == "lean_compile":
            try:
                decoded = json.loads(arguments or "{}")
                lean_code = decoded.get("lean_code") if isinstance(decoded, dict) else None
            except json.JSONDecodeError:
                lean_code = None
            if isinstance(lean_code, str):
                arguments = json.dumps({
                    "lean_code_sha256": hashlib.sha256(lean_code.encode()).hexdigest(),
                    "lean_code_chars": len(lean_code),
                }, sort_keys=True)
        payload.append({
            "id": tc.id,
            "type": getattr(tc, "type", "function"),
            "function": {
                "name": tc.function.name,
                "arguments": arguments,
            },
        })
    return payload


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
    message_extra = getattr(msg, "model_extra", None) or {}
    reasoning = (
        getattr(msg, "reasoning_content", None)
        or message_extra.get("reasoning_content")
        or message_extra.get("reasoning")
        or ""
    )
    usage = getattr(response, "usage", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    reasoning_tokens = getattr(
        getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None,
    )
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
            "reasoning_content": reasoning,
            "reasoning_characters": len(reasoning),
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "tool_call_tokens": (
                completion_tokens - reasoning_tokens
                if isinstance(completion_tokens, int) and isinstance(reasoning_tokens, int)
                else None
            ),
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
    source_contexts: list[dict[str, Any]] | None = None,
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
            "timings": result.timings,
            "source_contexts": source_contexts or [],
        },
        ok=result.success,
    ))


def _diagnostic_line(value: Any) -> int | None:
    payload = value
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            match = re.search(r"(?:line\s+|:)(\d+)(?::\d+)?", value, re.I)
            return int(match.group(1)) if match else None
    if not isinstance(payload, dict):
        return None
    for key in ("pos", "startPos", "start", "position"):
        position = payload.get(key)
        if isinstance(position, dict) and isinstance(position.get("line"), int):
            return int(position["line"])
    return int(payload["line"]) if isinstance(payload.get("line"), int) else None



def _set_latest_refinement_retry(
    messages: list[dict],
    base_messages: tuple[dict, ...],
    lean_code: str,
    feedback: str,
    *,
    finish_reason: str | None = None,
) -> None:
    """Keep one bounded, useful repair turn without replaying model reasoning.

    Blueprint responses from reasoning models can spend the entire completion
    budget before or around the Lean block.  Replaying that raw response on
    every repair attempt makes the prompt grow monotonically and can leave no
    room for the configured completion budget.  The compiler only repairs the
    latest Lean candidate, so retain the immutable original prompt plus that
    candidate and its latest diagnostic.
    """
    candidate = lean_code.strip()
    candidate_content = (
        f"```lean\n{candidate}\n```"
        if candidate
        else "No valid Lean blueprint was emitted."
    )
    retry_feedback = feedback
    if finish_reason == "length":
        retry_feedback = (
            "The previous response reached its output limit. Emit only one concise, "
            "complete Lean file with no reasoning outside the code block.\n\n"
            + retry_feedback
        )
    messages[:] = [dict(message) for message in base_messages]
    messages.extend([
        {"role": "assistant", "content": candidate_content},
        {"role": "user", "content": retry_feedback},
    ])



def _call_refinement_model(
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
        operation="blueprint_refinement",
        trace_args={"attempt": attempt, "turn": 1, "max_tokens": max_tokens},
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


LEAN_COMPILE_TOOL = {
    "type": "function",
    "function": {
        "name": "lean_compile",
        "description": "Compile and structurally validate one complete Lean Blueprint file.",
        "parameters": {
            "type": "object",
            "properties": {"lean_code": {"type": "string"}},
            "required": ["lean_code"],
            "additionalProperties": False,
        },
    },
}


def _strip_lean_comments(source: str) -> str:
    """Remove nested Lean comments while preserving line boundaries."""
    result: list[str] = []
    index = 0
    block_depth = 0
    in_string = False
    while index < len(source):
        pair = source[index:index + 2]
        char = source[index]
        if block_depth:
            if pair == "/-":
                block_depth += 1
                index += 2
            elif pair == "-/":
                block_depth -= 1
                index += 2
            else:
                if char == "\n":
                    result.append(char)
                index += 1
            continue
        if not in_string and pair == "/-":
            block_depth = 1
            index += 2
            continue
        if not in_string and pair == "--":
            newline = source.find("\n", index + 2)
            if newline < 0:
                break
            result.append("\n")
            index = newline + 1
            continue
        result.append(char)
        if char == '"' and (index == 0 or source[index - 1] != "\\"):
            in_string = not in_string
        index += 1
    return "".join(result)


def _unannotated_local_declaration_errors(blueprint: Blueprint) -> list[str]:
    """Reject local declarations that canonical Blueprint rebuilding would drop."""
    residue = blueprint.lean_file
    for node in blueprint.nodes:
        residue = residue.replace(node.lean_declaration, "", 1)
    residue = _strip_lean_comments(residue)
    unexpected: list[str] = []
    for raw_line in residue.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^(?:import|open(?:\s+scoped)?|set_option)\b", line):
            continue
        unexpected.append(line)
    if not unexpected:
        return []
    preview = "; ".join(unexpected[:8])
    suffix = "" if len(unexpected) <= 8 else f"; ... and {len(unexpected) - 8} more lines"
    return [
        "unannotatedLocalDeclaration: canonical rebuilding preserves only imports/options and "
        "@[blueprint] declarations. Convert every local declaration to a Blueprint "
        f"node or remove it. Unexpected source: {preview}{suffix}"
    ]


@dataclass
class BlueprintNode:
    name: str
    kind: str  # "definition" | "lemma" | "theorem"
    statement: str
    proof_sketch: str
    dependencies: list[str] = field(default_factory=list)
    lean_declaration: str = ""
    # LeanArchitect already persists ``title`` as part of the native
    # ``@[blueprint]`` attribute.  Semantic-fidelity experiments encode the
    # immutable source binding as ``title := \"COT_STEP:S001\"`` so the
    # mapping survives parsing, checkpointing, and export without inventing a
    # custom Lean attribute.
    title: str = ""
    source_step_id: str = ""
    lean_start_line: int = 0
    lean_end_line: int = 0

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
        return (
            declaration_shape
            + "\x00deps:"
            + ",".join(sorted(self.dependencies))
            + "\x00source_step_id:"
            + self.source_step_id
        )


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
    semantic_gate_results: list[dict] = field(default_factory=list)
    semantic_audit_result: dict = field(default_factory=dict)
    candidate_history: list[str] = field(default_factory=list, repr=False)
    candidate_labels: list[str] = field(default_factory=list, repr=False)
    generation_validation: dict = field(default_factory=dict, repr=False)
    generation_history: list[dict] = field(default_factory=list, repr=False)

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


@dataclass(frozen=True)
class _Phase2PreflightCase:
    node_name: str
    lean_code: str
    code_hash: str
    line_ranges: tuple[tuple[int, int, str], ...]


@dataclass(frozen=True)
class Phase2StandaloneIssue:
    code: str
    node_name: str
    error_kind: str
    identifiers: tuple[str, ...]
    diagnostic: str
    preflight_hash: str
    origin_declaration: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "nodeName": self.node_name,
            "errorKind": self.error_kind,
            "identifiers": list(self.identifiers),
            "diagnostic": self.diagnostic,
            "preflightHash": self.preflight_hash,
            "originDeclaration": self.origin_declaration,
        }


@dataclass(frozen=True)
class Phase2StandaloneReport:
    issues: tuple[Phase2StandaloneIssue, ...]
    checked_node_count: int
    cached_node_count: int
    duration_ms: float
    not_run_reason: str = ""

    @property
    def failed_nodes(self) -> list[str]:
        return [issue.node_name for issue in self.issues]


def _phase2_preflight_case(blueprint: Blueprint, node: BlueprintNode) -> _Phase2PreflightCase:
    entries: list[tuple[str, str]] = [("<phase2Header>", blueprint.phase2_header.rstrip())]
    ancestor_deps = _transitive_node_deps(node, blueprint)
    included_proof_nodes = [
        dep_node for dep_node in blueprint.dependency_order()
        if dep_node.kind != "definition" and dep_node.name in ancestor_deps
    ]
    entries.extend(
        (definition.name, definition.full_declaration())
        for definition in blueprint.nodes
        if definition.kind == "definition" and definition.name != node.name
    )
    entries.extend(
        (dep_node.name, dep_node.full_declaration())
        for dep_node in included_proof_nodes
    )
    entries.append((node.name, node.full_declaration()))

    rendered = ""
    ranges: list[tuple[int, int, str]] = []
    for origin, raw_text in entries:
        text = raw_text.strip()
        if not text:
            continue
        if rendered:
            rendered += "\n\n"
        start_line = rendered.count("\n") + 1
        rendered += text
        end_line = rendered.count("\n") + 1
        ranges.append((start_line, end_line, origin))
    rendered += "\n"
    return _Phase2PreflightCase(
        node_name=node.name,
        lean_code=rendered,
        code_hash=hashlib.sha256(rendered.encode()).hexdigest(),
        line_ranges=tuple(ranges),
    )


def _phase2_preflight_file(blueprint: Blueprint, node: BlueprintNode) -> str:
    return _phase2_preflight_case(blueprint, node).lean_code


def _standalone_error_kind(message: str) -> str:
    if re.search(r"Unknown identifier", message, re.I):
        return "unknownIdentifier"
    if re.search(r"Unknown constant", message, re.I):
        return "unknownConstant"
    if re.search(r"(?:application )?type mismatch|Invalid field notation", message, re.I):
        return "typeMismatch"
    if re.search(r"failed to synthesize", message, re.I):
        return "synthesisFailure"
    if re.search(r"unexpected token|unexpected end", message, re.I):
        return "syntaxError"
    return "leanCompileError"


def _standalone_identifiers(message: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(re.findall(
        r"Unknown (?:identifier|constant) `([^`]+)`", message,
    )))


def _preflight_origin(case: _Phase2PreflightCase, result: CompilerResult) -> str:
    for diagnostic in result.errors:
        line = _diagnostic_line(diagnostic)
        if line is None:
            continue
        for start, end, origin in case.line_ranges:
            if start <= line <= end:
                return origin
    return ""


def _emit_standalone_report(
    tracer,
    *,
    thm_name: str,
    round_index: int,
    report: Phase2StandaloneReport,
    phase: str = "phase1B",
) -> None:
    if tracer is None:
        return
    error_counts = dict(Counter(issue.error_kind for issue in report.issues))
    tracer.emit(TraceEvent(
        kind="phase2StandaloneCheckEnd",
        thm_name=thm_name,
        turn=round_index,
        args={
            "phase": phase,
            "round": round_index,
            "checkedNodeCount": report.checked_node_count,
            "cachedNodeCount": report.cached_node_count,
            "failedNodeCount": len(report.issues),
            "errorCounts": error_counts,
            "failedNodes": report.failed_nodes,
            "notRunReason": report.not_run_reason,
        },
        ok=not report.issues and not report.not_run_reason,
        duration_ms=report.duration_ms,
    ))


def phase2_standalone_contract_report(
    blueprint: Blueprint,
    compiler: KiminaLeanCompiler,
    *,
    concurrency: int = 1,
    cache: dict[str, CompilerResult] | None = None,
    tracer=None,
    thm_name: str = "",
    round_index: int = 0,
    trace_phase: str = "phase1B",
) -> Phase2StandaloneReport:
    """Compile proof nodes exactly as Phase 2 will assemble them."""
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    started_ns = time.monotonic_ns()
    proof_nodes = [node for node in blueprint.nodes if node.kind in {"lemma", "theorem"}]
    cases = [_phase2_preflight_case(blueprint, node) for node in proof_nodes]
    result_cache = cache if cache is not None else {}
    uncached = [case for case in cases if case.code_hash not in result_cache]

    if tracer is not None:
        tracer.emit(TraceEvent(
            kind="phase2StandaloneCheckStart",
            thm_name=thm_name,
            turn=round_index,
            args={
                "phase": trace_phase, "round": round_index,
                "checkedNodeCount": len(cases),
                "cachedNodeCount": len(cases) - len(uncached),
            },
        ))

    if uncached:
        results = compiler.check_many(
            [
                CompileRequest(
                    case.lean_code,
                    allow_sorry=True,
                    request_id=f"phase2-contract-{round_index}-{index}-{case.node_name}",
                )
                for index, case in enumerate(uncached)
            ],
            batch_concurrency=concurrency,
        )
        for case, result in zip(uncached, results, strict=True):
            result_cache[case.code_hash] = result

    issues: list[Phase2StandaloneIssue] = []
    cached_count = len(cases) - len(uncached)
    for case in cases:
        result = result_cache[case.code_hash]
        if result.failure_kind == "infra":
            message = "\n".join(result.diagnostics) or result.raw_output[-2000:]
            raise KiminaInfrastructureError(message)
        if result.success:
            issue = None
        else:
            message = "\n".join(result.diagnostics) or result.raw_output[-4000:]
            issue = Phase2StandaloneIssue(
                code="phase2StandaloneFailed",
                node_name=case.node_name,
                error_kind=_standalone_error_kind(message),
                identifiers=_standalone_identifiers(message),
                diagnostic=message[-4000:],
                preflight_hash=case.code_hash,
                origin_declaration=_preflight_origin(case, result),
            )
            issues.append(issue)
        if tracer is not None:
            tracer.emit(TraceEvent(
                kind="phase2StandaloneNodeResult",
                thm_name=thm_name,
                turn=round_index,
                args={
                    "phase": trace_phase, "round": round_index,
                    "nodeName": case.node_name,
                    "preflightHash": case.code_hash,
                    "cacheHit": case.code_hash not in {item.code_hash for item in uncached},
                    "issue": issue.to_dict() if issue is not None else None,
                },
                ok=issue is None,
            ))

    report = Phase2StandaloneReport(
        issues=tuple(issues),
        checked_node_count=len(cases),
        cached_node_count=cached_count,
        duration_ms=(time.monotonic_ns() - started_ns) / 1_000_000,
    )
    _emit_standalone_report(
        tracer, thm_name=thm_name, round_index=round_index, report=report,
        phase=trace_phase,
    )
    return report


def phase2_standalone_contract_errors(
    blueprint: Blueprint,
    compiler: KiminaLeanCompiler,
    *,
    limit: int = 0,
    concurrency: int = 1,
) -> list[str]:
    """Compile proof nodes as Phase 2 would see them before accepting a blueprint."""
    report = phase2_standalone_contract_report(
        blueprint, compiler, concurrency=concurrency,
    )
    errors = [
        f"phase2StandaloneFailed: node `{issue.node_name}` does not compile when "
        f"assembled as a standalone Phase 2 goal; errorKind={issue.error_kind}; "
        f"identifiers={list(issue.identifiers)}; origin={issue.origin_declaration or '<unknown>'}.\n"
        f"{issue.diagnostic}"
        for issue in report.issues
    ]
    if limit:
        errors = errors[:limit]
    return errors


def format_phase2_standalone_issues(
    issues: list[Phase2StandaloneIssue] | tuple[Phase2StandaloneIssue, ...],
) -> str:
    """Group repeated standalone failures without dropping affected nodes."""
    groups: dict[tuple[str, tuple[str, ...]], list[Phase2StandaloneIssue]] = {}
    for issue in issues:
        key = (issue.error_kind, issue.identifiers)
        groups.setdefault(key, []).append(issue)
    sections: list[str] = []
    for (error_kind, identifiers), grouped in groups.items():
        nodes = ", ".join(item.node_name for item in grouped)
        origins = ", ".join(dict.fromkeys(
            item.origin_declaration or "<unknown>" for item in grouped
        ))
        representative = grouped[0].diagnostic
        sections.append(
            f"- {error_kind}: identifiers={list(identifiers) or ['<none>']} "
            f"origins={origins}\n"
            f"  Affected nodes: {nodes}\n"
            f"  Representative diagnostic: {representative}"
        )
    return "\n".join(sections)


def phase2_contract_errors(blueprint: Blueprint) -> list[str]:
    """Return structural errors that would make Phase 2 node proving invalid."""
    errors: list[str] = []
    node_names = set(blueprint.nodes_by_name())
    for node in blueprint.nodes:
        unknown = [name for name in node.dependencies if name not in node_names]
        if unknown:
            errors.append(
                f"nonBlueprintDependency: node `{node.name}` lists names that are not "
                f"@[blueprint] nodes: {unknown}."
            )
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


def canonicalize_blueprint(current: Blueprint, nodes: list[BlueprintNode]) -> Blueprint:
    names = [node.name for node in nodes]
    if len(names) != len(set(names)):
        raise ValueError("duplicateNodeName")
    node_map = {node.name: node for node in nodes}
    root = node_map.get(current.target_theorem)
    if root is None or root.kind != "theorem":
        raise ValueError("missingOrInvalidRoot")
    for node in nodes:
        unknown = [dependency for dependency in node.dependencies if dependency not in node_map]
        if unknown:
            raise ValueError(
                f"unknownDependencies:{node.name}:{','.join(unknown)}"
            )

    draft = Blueprint(
        nodes=nodes,
        lean_file="",
        target_theorem=current.target_theorem,
        phase2_header=current.phase2_header,
    )
    ordered = draft.dependency_order()
    definitions = [node for node in nodes if node.kind == "definition"]
    proofs = [
        node for node in ordered
        if node.kind != "definition" and node.name != current.target_theorem
    ]
    ordered_nodes = definitions + proofs + [root]
    parts = [current.phase2_header.rstrip()]
    parts.extend(node.lean_declaration.strip() for node in ordered_nodes)
    lean_code = "\n\n".join(part for part in parts if part.strip()).strip() + "\n"
    revised = _parse_blueprint(lean_code, current.target_theorem)
    if [node.name for node in revised.nodes] != [node.name for node in ordered_nodes]:
        raise ValueError("editedBlueprintParseMismatch")
    revised.dependency_order()
    return revised



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
        title = _extract_string_attr(attrs_block, "title")
        source_match = re.fullmatch(r"COT_STEP:(S\d{3,})", title.strip())
        source_step_id = source_match.group(1) if source_match else ""

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
            title=title,
            source_step_id=source_step_id,
            lean_start_line=lean_code.count("\n", 0, m.start()) + 1,
            lean_end_line=lean_code.count("\n", 0, m.end()) + 1,
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


def _extract_string_attr(attrs: str, key: str) -> str:
    """Extract a Lean string-valued blueprint attribute.

    ``title`` is deliberately kept machine-readable.  JSON's string decoder
    matches the escapes used by the subset of Lean strings emitted here and
    avoids silently retaining quotes/backslashes in step identifiers.
    """
    match = re.search(rf'\({key}\s*:=\s*("(?:\\.|[^"\\])*")\)', attrs, re.DOTALL)
    if not match:
        return ""
    try:
        return str(json.loads(match.group(1)))
    except json.JSONDecodeError:
        return ""


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

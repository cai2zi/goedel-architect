from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from omegaconf import DictConfig
from tqdm import tqdm

from cot_blueprint_refine.common import (
    REPO_ROOT,
    latest_rows,
    prepared_dir,
    prompt_safe_comment_lines,
    robustpa_dir,
    write_json,
    write_jsonl,
)

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from blueprint import Blueprint, _extract_lean_code, render_solved_declaration  # noqa: E402
from checkpoint import CheckpointState  # noqa: E402
from kimina_lean_compiler import KiminaLeanCompiler  # noqa: E402
from orchestrator import active_node_names  # noqa: E402
from robustpa_refine.io_utils import safe_stem  # noqa: E402


PROMPT_SIGNAL_MAP = {
    "solved": "PROVED",
    "proof_too_hard": "NOT_PROVED",
    "protocol_error": "NOT_PROVED",
    "blocked_by_dependency": "BLOCKED_BY_DEPENDENCY",
    "formally_negated": "FORMALLY_NEGATED",
}

VERIFIED = "VERIFIED"
INVALID_BLUEPRINT_CANDIDATE = "INVALID_BLUEPRINT_CANDIDATE"
INFRA_ERROR = "INFRA_ERROR"


def prompt_signal(raw_signal: str, *, proved: bool = False) -> str:
    if proved:
        return "PROVED"
    return PROMPT_SIGNAL_MAP.get(raw_signal, "NOT_PROVED")


def _status_explanation(signal: str) -> str:
    if signal == "DEFINITION":
        return (
            "Lean accepted this definition and its body as well-typed. This is not a proof "
            "that the definition faithfully represents the original problem or COT step, "
            "and it does not validate any answer or mathematical claim encoded in the body."
        )
    if signal == "NOT_PROVED":
        return (
            "This node was not successfully proved. The corresponding solution step may be "
            "wrong, incomplete, or require a different method; failure alone does not prove it false."
        )
    if signal == "BLOCKED_BY_DEPENDENCY":
        return (
            "This node was not attempted because an upstream dependency failed. This status "
            "does not independently determine whether the node is correct."
        )
    if signal == "FORMALLY_NEGATED":
        return (
            "Lean accepted a proof of the formal negation. The corresponding problem-solving "
            "step is wrong and must be replaced during COT refinement."
        )
    return "Lean successfully checked this declaration and proof."


def render_blueprint_context(
    blueprint: Blueprint,
    state: CheckpointState,
    node_overrides: dict[str, dict[str, Any]] | None = None,
) -> tuple[str, list[dict[str, Any]], bool]:
    active = active_node_names(blueprint)
    parts = [blueprint.phase2_header.rstrip()]
    node_rows: list[dict[str, Any]] = []
    has_infra_error = False

    for node in blueprint.dependency_order():
        if node.name not in active:
            continue
        has_override = node.name in (node_overrides or {})
        result = dict(
            (node_overrides or {}).get(node.name)
            or state.node_results.get(node.name)
            or {}
        )
        raw_signal = str(result.get("signal") or "")
        proof_body = str(
            result.get("proof_body")
            or ("" if has_override else state.proved_cache.get(node.name))
            or ""
        )
        lean_errors = [str(value) for value in (result.get("lean_errors") or [])]
        if raw_signal == "infra_error":
            has_infra_error = True

        if node.kind == "definition":
            signal = "DEFINITION"
            rendered = node.full_declaration().strip()
        else:
            proved = (
                raw_signal == "solved" and bool(proof_body)
                if has_override
                else node.name in state.proved_cache
                or (raw_signal == "solved" and bool(proof_body))
            )
            signal = prompt_signal(raw_signal, proved=proved)
            if signal == "PROVED":
                rendered = render_solved_declaration(node, proof_body).strip()
            else:
                rendered = node.full_declaration().strip()

        block = [
            *prompt_safe_comment_lines(
                "COT_BLUEPRINT_SOURCE_STEP", node.source_step_id or "(unmapped)",
            ),
            *prompt_safe_comment_lines("COT_BLUEPRINT_NODE_STATEMENT", node.statement or "(none)"),
            *prompt_safe_comment_lines("COT_BLUEPRINT_NODE_PROOF_SKETCH", node.proof_sketch or "(none)"),
            f"-- COT_BLUEPRINT_NODE_STATUS: {signal}",
            rendered,
        ]
        if signal != "PROVED":
            block.extend(prompt_safe_comment_lines("STATUS_MEANING", _status_explanation(signal)))
            if proof_body and signal == "NOT_PROVED":
                block.append("-- The following proof attempt did not compile and is reference only:")
                block.extend(f"-- {line}" for line in proof_body.splitlines())
            block.extend(prompt_safe_comment_lines("LEAN_DIAGNOSTICS", "\n".join(lean_errors) or "(none)"))
        parts.append("\n".join(block))
        node_rows.append({
            "name": node.name,
            "kind": node.kind,
            "source_step_id": node.source_step_id,
            "statement": node.statement,
            "proof_sketch": node.proof_sketch,
            "dependencies": list(node.dependencies),
            "raw_signal": raw_signal or (
                "definition" if signal == "DEFINITION"
                else "solved" if signal == "PROVED"
                else "pending"
            ),
            "prompt_signal": signal,
            "proof_body": proof_body,
            "lean_errors": lean_errors,
        })
    return "\n\n".join(part for part in parts if part) + "\n", node_rows, has_infra_error


def _transitive_dependency_names(blueprint: Blueprint, node_name: str) -> set[str]:
    node_map = blueprint.nodes_by_name()
    seen: set[str] = set()
    node = node_map.get(node_name)
    stack = list(node.dependencies if node is not None else [])
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        dependency = node_map.get(name)
        if dependency is not None:
            stack.extend(dependency.dependencies)
    return seen


def revalidate_proved_nodes(
    blueprint: Blueprint,
    state: CheckpointState,
    compiler: KiminaLeanCompiler,
) -> tuple[dict[str, dict[str, Any]], bool]:
    """Recheck cached proofs against the final blueprint; return overrides and infra flag."""
    active = active_node_names(blueprint)
    node_map = blueprint.nodes_by_name()
    definitions = [
        node.full_declaration()
        for node in blueprint.nodes
        if node.kind == "definition"
    ]
    verified: dict[str, str] = {}
    overrides: dict[str, dict[str, Any]] = {}
    for node in blueprint.dependency_order():
        if node.name not in active or node.kind == "definition":
            continue
        stored = dict(state.node_results.get(node.name) or {})
        proof_body = str(stored.get("proof_body") or state.proved_cache.get(node.name) or "")
        claimed_proved = node.name in state.proved_cache or (
            str(stored.get("signal") or "") == "solved" and bool(proof_body)
        )
        if not claimed_proved:
            continue
        unavailable = [
            dep
            for dep in node.dependencies
            if dep in active
            and (dependency := node_map.get(dep)) is not None
            and dependency.kind != "definition"
            and dep not in verified
        ]
        if unavailable:
            overrides[node.name] = {
                "signal": "blocked_by_dependency",
                "proof_body": "",
                "lean_errors": [f"Unresolved dependencies: {', '.join(sorted(unavailable))}"],
            }
            continue
        ancestors = _transitive_dependency_names(blueprint, node.name)
        parent_declarations = definitions + [
            render_solved_declaration(parent, verified[parent.name])
            for parent in blueprint.dependency_order()
            if parent.kind != "definition"
            and parent.name in ancestors
            and parent.name in verified
        ]
        result = compiler.check_node(
            proof_body,
            node_decl=node.lean_declaration,
            parent_lemma_decls="\n\n".join(parent_declarations),
            header=blueprint.phase2_header,
        )
        if result.failure_kind == "infra":
            return overrides, True
        if result.success:
            verified[node.name] = proof_body
            overrides[node.name] = {
                "signal": "solved",
                "proof_body": proof_body,
                "lean_errors": [],
            }
        else:
            overrides[node.name] = {
                "signal": "proof_too_hard",
                "proof_body": proof_body,
                "lean_errors": list(result.diagnostics),
            }
    return overrides, False


def _candidate_from_trace(trace_path: Path) -> str:
    candidate = ""
    if not trace_path.exists():
        return candidate
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_candidate = ""
        if event.get("kind") == "llm_response":
            extracted = _extract_lean_code(str(event.get("result") or ""))
            if "@[blueprint" in extracted or extracted.lstrip().startswith("import "):
                event_candidate = extracted
        elif event.get("kind") == "lean_check_result":
            raw_output = str((event.get("args") or {}).get("raw_output") or "")
            try:
                raw = json.loads(raw_output)
                snippets = raw.get("request", {}).get("snippets", [])
                if snippets:
                    event_candidate = str(snippets[0].get("code") or "")
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass
        if event_candidate.strip():
            candidate = event_candidate.strip()
    return candidate


def recover_invalid_blueprint(result_row: dict[str, Any]) -> tuple[str, str]:
    """Prefer the new saved artifact, then fall back to the historical trace."""
    candidate_path_text = str(result_row.get("failed_blueprint_candidate_path") or "")
    if candidate_path_text:
        candidate_path = Path(candidate_path_text)
        if candidate_path.exists():
            candidate = candidate_path.read_text(encoding="utf-8").strip()
            if candidate:
                return candidate, "saved_artifact"
    trace_path_text = str(result_row.get("trace_path") or "")
    if trace_path_text:
        candidate = _candidate_from_trace(Path(trace_path_text))
        if candidate:
            return candidate, "trace_fallback"
    return "", ""


def _make_compiler(config: DictConfig) -> KiminaLeanCompiler:
    return KiminaLeanCompiler(
        api_url=str(config.blueprint.lean_api_url),
        timeout_s=int(config.blueprint.lean_server_timeout),
        reuse=True,
        debug=False,
        max_inflight_snippets=int(config.blueprint.lean_max_inflight_snippets),
        batch_size=int(config.blueprint.lean_batch_size),
    )


def _base_export_row(
    result_row: dict[str, Any],
    generation: dict[str, Any] | None,
) -> dict[str, Any]:
    source_id = str(result_row.get("source_id") or "")
    return {
        "ID": source_id,
        "source": str((generation or {}).get("source") or result_row.get("split") or ""),
        "problem": str((generation or {}).get("problem") or ""),
        "claimed_answer": str((generation or {}).get("claimed_answer") or ""),
        "original_cot": str((generation or {}).get("post_think_cot") or ""),
        "robustpa_status": str(result_row.get("status") or ""),
        "root_proved": bool(result_row.get("root_proved")),
        "checkpoint_path": str(result_row.get("checkpoint_path") or ""),
    }


def _process_export_row(
    index: int,
    result_row: dict[str, Any],
    generation: dict[str, Any] | None,
    lean_dir: Path,
    compiler: KiminaLeanCompiler,
) -> tuple[int, dict[str, Any], Counter[str]]:
    counts: Counter[str] = Counter()
    base_row = _base_export_row(result_row, generation)
    checkpoint_path = Path(base_row["checkpoint_path"])
    if generation is None:
        base_row.update(
            status="pipeline_error",
            error="missing_generation_input",
            context_quality=INFRA_ERROR,
            refine_eligible=True,
            lean_context="",
            lean_validated=False,
        )
        counts["pipeline_error"] += 1
        counts[f"context_{INFRA_ERROR}"] += 1
        return index, base_row, counts
    if str(result_row.get("status") or "") == "error" or not checkpoint_path.exists():
        candidate, candidate_source = recover_invalid_blueprint(result_row)
        quality = INVALID_BLUEPRINT_CANDIDATE if candidate else INFRA_ERROR
        base_row.update(
            status="pipeline_error",
            error=str(result_row.get("error") or "missing_or_error_checkpoint"),
            context_quality=quality,
            refine_eligible=True,
            lean_context=candidate,
            lean_validated=False,
            lean_validation_errors=[str(result_row.get("error") or "missing_or_error_checkpoint")],
            invalid_blueprint_source=candidate_source,
            nodes=[],
        )
        counts["pipeline_error"] += 1
        counts[f"context_{quality}"] += 1
        return index, base_row, counts
    try:
        state = CheckpointState.load(checkpoint_path)
        blueprint = state.get_blueprint()
        if blueprint is None:
            raise ValueError("checkpoint has no blueprint")
        lean_context, nodes, has_infra_error = render_blueprint_context(blueprint, state)
        if has_infra_error:
            base_row.update(
                status="pipeline_error",
                error="checkpoint contains infra_error node",
                context_quality=INFRA_ERROR,
                refine_eligible=True,
                lean_context=lean_context,
                lean_validated=False,
                nodes=nodes,
            )
            counts["pipeline_error"] += 1
            counts[f"context_{INFRA_ERROR}"] += 1
            return index, base_row, counts
        validation = compiler.check(lean_context, allow_sorry=not state.root_proved)
        if not validation.success and validation.failure_kind == "lean":
            overrides, revalidation_infra = revalidate_proved_nodes(blueprint, state, compiler)
            if revalidation_infra:
                base_row.update(
                    status="pipeline_error",
                    error="proof revalidation failed because the Lean service returned an infra error",
                    context_quality=INFRA_ERROR,
                    refine_eligible=True,
                    lean_context=lean_context,
                    lean_validated=False,
                    lean_validation_errors=validation.diagnostics,
                    nodes=nodes,
                )
                counts["pipeline_error"] += 1
                counts[f"context_{INFRA_ERROR}"] += 1
                return index, base_row, counts
            lean_context, nodes, _ = render_blueprint_context(
                blueprint, state, node_overrides=overrides,
            )
            validation = compiler.check(lean_context, allow_sorry=True)
        lean_path = lean_dir / f"{safe_stem(str(result_row.get('source_id') or ''), prefix='cot_')}.lean"
        lean_path.parent.mkdir(parents=True, exist_ok=True)
        lean_path.write_text(lean_context, encoding="utf-8")
        quality = (
            VERIFIED if validation.success
            else INFRA_ERROR if validation.failure_kind == "infra"
            else INVALID_BLUEPRINT_CANDIDATE
        )
        status = "ready" if quality == VERIFIED else (
            "pipeline_error" if quality == INFRA_ERROR else "export_error"
        )
        base_row.update({
            "status": status,
            "error": "" if validation.success else "\n".join(validation.diagnostics),
            "context_quality": quality,
            "refine_eligible": True,
            "target_theorem": blueprint.target_theorem,
            "lean_context": lean_context,
            "lean_context_path": str(lean_path),
            "lean_context_sha256": hashlib.sha256(lean_context.encode("utf-8")).hexdigest(),
            "lean_validated": validation.success,
            "lean_validation_errors": validation.diagnostics,
            "nodes": nodes,
        })
        counts[status] += 1
        counts[f"context_{quality}"] += 1
        counts.update(f"node_{node['prompt_signal']}" for node in nodes)
        return index, base_row, counts
    except Exception as exc:  # noqa: BLE001
        base_row.update(
            status="pipeline_error",
            error=f"{type(exc).__name__}: {exc}",
            context_quality=INFRA_ERROR,
            refine_eligible=True,
            lean_context="",
            lean_validated=False,
        )
        counts["pipeline_error"] += 1
        counts[f"context_{INFRA_ERROR}"] += 1
        return index, base_row, counts


def _export_workers(config: DictConfig) -> int:
    export_config = config.get("export", {})
    workers = int(export_config.get("workers", 64))
    if workers <= 0:
        raise ValueError("export.workers must be positive")
    return workers


def export_contexts(config: DictConfig) -> dict[str, Any]:
    generation_rows = {
        str(row["name"]): row
        for row in latest_rows(prepared_dir(config) / "generation_inputs.jsonl", "name")
    }
    results_path = robustpa_dir(config) / "results.jsonl"
    result_rows = latest_rows(results_path, "source_id")
    out_dir = Path(str(config.output_base)).expanduser() / str(config.exp_name) / "blueprint_contexts"
    lean_dir = out_dir / "lean"
    indexed_exported: list[dict[str, Any] | None] = [None] * len(result_rows)
    counts: Counter[str] = Counter()
    workers = _export_workers(config)
    compiler = _make_compiler(config)
    try:
        print(
            f"[export] rows={len(result_rows)} workers={workers} "
            f"lean_max_inflight_snippets={int(config.blueprint.lean_max_inflight_snippets)} "
            f"out_dir={out_dir}",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _process_export_row,
                    index,
                    result_row,
                    generation_rows.get(str(result_row.get("source_id") or "")),
                    lean_dir,
                    compiler,
                )
                for index, result_row in enumerate(result_rows)
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="export-blueprint",
                unit="row",
            ):
                index, export_row, row_counts = future.result()
                indexed_exported[index] = export_row
                counts.update(row_counts)
    finally:
        compiler.close()

    exported = [row for row in indexed_exported if row is not None]
    expected = set(generation_rows)
    actual = {str(row.get("ID") or "") for row in exported}
    for missing_id in sorted(expected - actual):
        generation = generation_rows[missing_id]
        exported.append({
            "ID": missing_id,
            "source": generation["source"],
            "problem": generation["problem"],
            "claimed_answer": generation["claimed_answer"],
            "original_cot": generation["post_think_cot"],
            "status": "pipeline_error",
            "error": "missing_robustpa_result",
            "context_quality": INFRA_ERROR,
            "refine_eligible": True,
            "lean_context": "",
            "lean_validated": False,
            "root_proved": False,
        })
        counts["pipeline_error"] += 1
        counts[f"context_{INFRA_ERROR}"] += 1

    output_path = out_dir / "blueprint_contexts.jsonl"
    quality_counts = Counter(str(row.get("context_quality") or INFRA_ERROR) for row in exported)
    invalid_ids = [
        str(row.get("ID") or "")
        for row in exported
        if row.get("context_quality") == INVALID_BLUEPRINT_CANDIDATE
    ]
    infra_ids = [
        str(row.get("ID") or "")
        for row in exported
        if row.get("context_quality") == INFRA_ERROR
    ]
    metrics = {
        "rows": len(exported),
        "counts": dict(sorted(counts.items())),
        "context_quality_counts": dict(sorted(quality_counts.items())),
        "invalid_blueprint_candidate_ids": invalid_ids,
        "infra_error_ids": infra_ids,
        "output": str(output_path),
    }
    write_jsonl(output_path, exported)
    write_json(out_dir / "export_metrics.json", metrics)
    print(f"[export] rows={len(exported)} counts={metrics['counts']} output={output_path}", flush=True)
    print(
        f"[export-context] quality={metrics['context_quality_counts']} "
        f"invalid_ids={invalid_ids} infra_ids={infra_ids}",
        flush=True,
    )
    return metrics

from __future__ import annotations

import hashlib
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

from blueprint import Blueprint, render_solved_declaration  # noqa: E402
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


def prompt_signal(raw_signal: str, *, proved: bool = False) -> str:
    if proved:
        return "PROVED"
    return PROMPT_SIGNAL_MAP.get(raw_signal, "NOT_PROVED")


def _status_explanation(signal: str) -> str:
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
) -> tuple[str, list[dict[str, Any]], bool]:
    active = active_node_names(blueprint)
    parts = [blueprint.phase2_header.rstrip()]
    node_rows: list[dict[str, Any]] = []
    has_infra_error = False

    for node in blueprint.dependency_order():
        if node.name not in active:
            continue
        result = dict(state.node_results.get(node.name) or {})
        raw_signal = str(result.get("signal") or "")
        proof_body = str(result.get("proof_body") or state.proved_cache.get(node.name) or "")
        lean_errors = [str(value) for value in (result.get("lean_errors") or [])]
        if raw_signal == "infra_error":
            has_infra_error = True

        if node.kind == "definition":
            signal = "PROVED"
            rendered = node.full_declaration().strip()
        else:
            proved = node.name in state.proved_cache or (raw_signal == "solved" and bool(proof_body))
            signal = prompt_signal(raw_signal, proved=proved)
            if signal == "PROVED":
                rendered = render_solved_declaration(node, proof_body).strip()
            else:
                rendered = node.full_declaration().strip()

        block = [
            *prompt_safe_comment_lines("COT_BLUEPRINT_NODE_STATEMENT", node.statement or "(none)"),
            *prompt_safe_comment_lines("COT_BLUEPRINT_NODE_PROOF_SKETCH", node.proof_sketch or "(none)"),
            f"-- COT_BLUEPRINT_NODE_STATUS: {signal}",
            rendered,
        ]
        if signal != "PROVED":
            block.extend(prompt_safe_comment_lines("STATUS_MEANING", _status_explanation(signal)))
            block.extend(prompt_safe_comment_lines("LAST_SUBMITTED_PROOF", proof_body or "(none)"))
            block.extend(prompt_safe_comment_lines("LEAN_DIAGNOSTICS", "\n".join(lean_errors) or "(none)"))
        parts.append("\n".join(block))
        node_rows.append({
            "name": node.name,
            "kind": node.kind,
            "statement": node.statement,
            "proof_sketch": node.proof_sketch,
            "dependencies": list(node.dependencies),
            "raw_signal": raw_signal or ("solved" if signal == "PROVED" else "pending"),
            "prompt_signal": signal,
            "proof_body": proof_body,
            "lean_errors": lean_errors,
        })
    return "\n\n".join(part for part in parts if part) + "\n", node_rows, has_infra_error


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
        base_row.update(status="pipeline_error", error="missing_generation_input")
        counts["pipeline_error"] += 1
        return index, base_row, counts
    if str(result_row.get("status") or "") == "error" or not checkpoint_path.exists():
        base_row.update(
            status="pipeline_error",
            error=str(result_row.get("error") or "missing_or_error_checkpoint"),
        )
        counts["pipeline_error"] += 1
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
                nodes=nodes,
            )
            counts["pipeline_error"] += 1
            return index, base_row, counts
        validation = compiler.check(lean_context, allow_sorry=not state.root_proved)
        lean_path = lean_dir / f"{safe_stem(str(result_row.get('source_id') or ''), prefix='cot_')}.lean"
        lean_path.parent.mkdir(parents=True, exist_ok=True)
        lean_path.write_text(lean_context, encoding="utf-8")
        status = "ready" if validation.success else (
            "pipeline_error" if validation.failure_kind == "infra" else "export_error"
        )
        base_row.update({
            "status": status,
            "error": "" if validation.success else "\n".join(validation.diagnostics),
            "target_theorem": blueprint.target_theorem,
            "lean_context": lean_context,
            "lean_context_path": str(lean_path),
            "lean_context_sha256": hashlib.sha256(lean_context.encode("utf-8")).hexdigest(),
            "lean_validated": validation.success,
            "lean_validation_errors": validation.diagnostics,
            "nodes": nodes,
        })
        counts[status] += 1
        counts.update(f"node_{node['prompt_signal']}" for node in nodes)
        return index, base_row, counts
    except Exception as exc:  # noqa: BLE001
        base_row.update(status="export_error", error=f"{type(exc).__name__}: {exc}")
        counts["export_error"] += 1
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
            "root_proved": False,
        })
        counts["pipeline_error"] += 1

    output_path = out_dir / "blueprint_contexts.jsonl"
    metrics = {"rows": len(exported), "counts": dict(sorted(counts.items())), "output": str(output_path)}
    write_jsonl(output_path, exported)
    write_json(out_dir / "export_metrics.json", metrics)
    print(f"[export] rows={len(exported)} counts={metrics['counts']} output={output_path}", flush=True)
    return metrics

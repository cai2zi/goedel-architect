from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from cot_blueprint_refine.common import latest_rows, prepared_dir, write_json
from cot_blueprint_refine.claim_scope_manifest import (
    SCHEMA_NAME as CLAIM_SCOPE_SCHEMA_NAME,
    SCHEMA_VERSION as CLAIM_SCOPE_SCHEMA_VERSION,
    encode_claim_scope_manifest,
    unassigned_spans,
)
from cot_blueprint_refine.cot_steps import (
    MANIFEST_SCHEMA_VERSION,
    SUBCLAIM_BUILDER_VERSION,
    build_cot_steps_from_sections,
    decode_steps,
    encode_steps,
)
from cot_blueprint_refine.cot_manifest_validation import validate_split_manifest
from cot_blueprint_refine.llm_cot_splitter import (
    ATOMIZER_VERSION,
    PROMPT_VERSION,
    SPLITTER_VERSION,
    split_cot_rows,
)
from cot_blueprint_refine.llm_claim_scope import (
    ANNOTATOR_VERSION as CLAIM_SCOPE_ANNOTATOR_VERSION,
    PROMPT_VERSION as CLAIM_SCOPE_PROMPT_VERSION,
    annotate_claim_scope_rows,
)
from cot_blueprint_refine.prepare_inputs import write_generation_artifacts
from semantic_fidelity import parse_cot_manifest


DETERMINISTIC_MODES = {"deterministic", "regex", "deterministic_v1"}
LLM_MODES = {
    "llm", "llm_boundary_v1", "llm_boundary_v4", "llm_boundary_v5",
    "llm_boundary_v6",
    "llm_claim_scope_v1",
}


def _write_claim_scope_metrics(
    artifact_root: Path,
    *,
    rows: list[dict[str, Any]],
    results: dict[str, Any],
) -> dict[str, Any]:
    claim_counts: list[int] = []
    scope_counts: list[int] = []
    context_chars: list[int] = []
    scope_types: Counter[str] = Counter()
    scoped_claims = 0
    total_chars = 0
    for row in rows:
        manifest = json.loads(str(row["cot_manifest_json"]))
        claims = list(manifest["claims"])
        scopes = list(manifest["scopes"])
        claim_counts.append(len(claims))
        scope_counts.append(len(scopes))
        scoped_claims += sum(bool(claim.get("scope_ids")) for claim in claims)
        scope_types.update(str(scope["scope_type"]) for scope in scopes)
        gaps = unassigned_spans(manifest)
        context_chars.append(sum(end - start for start, end, _text in gaps))
        total_chars += len(str(manifest["source_text"]))
    payload = {
        "mode": "llm_claim_scope_v1",
        "schema": CLAIM_SCOPE_SCHEMA_NAME,
        "schema_version": CLAIM_SCOPE_SCHEMA_VERSION,
        "annotator_version": CLAIM_SCOPE_ANNOTATOR_VERSION,
        "prompt_version": CLAIM_SCOPE_PROMPT_VERSION,
        "rows": len(rows),
        "status_counts": {"ok": len(rows)},
        "cached_rows": sum(bool(result.cached) for result in results.values()),
        "request_attempts": sum(
            len(result.attempts) for result in results.values() if not result.cached
        ),
        "claims_per_row": _distribution(claim_counts),
        "scopes_per_row": _distribution(scope_counts),
        "context_chars_per_row": _distribution(context_chars),
        "total_claims": sum(claim_counts),
        "total_scopes": sum(scope_counts),
        "scoped_claim_count": scoped_claims,
        "scope_type_counts": dict(sorted(scope_types.items())),
        "source_chars": total_chars,
        "unassigned_context_chars": sum(context_chars),
        "semantic_span_char_rate": (
            (total_chars - sum(context_chars)) / total_chars if total_chars else None
        ),
        "persisted_step_count": 0,
        "persisted_atom_count": 0,
    }
    write_json(artifact_root / "manifest_metrics.json", payload)
    return payload


def _run_claim_scope(
    config: DictConfig,
    rows: list[dict[str, Any]],
    artifact_root: Path,
) -> dict[str, Any]:
    results = asyncio.run(annotate_claim_scope_rows(rows, config.cot_splitter, artifact_root))
    failures = [result for result in results.values() if not result.ok]
    if len(results) != len(rows) or failures:
        preview = ", ".join(
            f"{result.row_id}:{result.status}:{result.error}" for result in failures[:10]
        )
        raise RuntimeError(
            "LLM Claim/Scope annotation failed; prepared artifacts were not replaced. "
            f"expected={len(rows)} results={len(results)} failures={len(failures)} {preview}"
        )
    rewritten: list[dict[str, Any]] = []
    for row in rows:
        row_id = str(row.get("name") or "")
        result = results[row_id]
        rewritten.append({
            **row,
            "cot_manifest_json": encode_claim_scope_manifest(result.manifest),
            "cot_splitter_mode": "llm_claim_scope_v1",
            "cot_splitter_status": "ok",
            "cot_splitter_version": CLAIM_SCOPE_ANNOTATOR_VERSION,
            "cot_splitter_source_sha256": result.source_sha256,
            "cot_splitter_cache_key": result.cache_key,
            "cot_splitter_step_count": 0,
            "cot_splitter_atom_count": 0,
            "cot_splitter_claim_count": len(result.manifest["claims"]),
            "cot_splitter_scope_count": len(result.manifest["scopes"]),
            "cot_splitter_attempts": len(result.attempts),
            "cot_splitter_cached": result.cached,
            "cot_splitter_fallback": False,
            "cot_manifest_schema": CLAIM_SCOPE_SCHEMA_NAME,
            "cot_manifest_schema_version": CLAIM_SCOPE_SCHEMA_VERSION,
            "cot_splitter_prompt_version": CLAIM_SCOPE_PROMPT_VERSION,
            "cot_splitter_prompt_content_sha256": result.prompt_content_sha256,
        })
    write_generation_artifacts(config, rewritten)
    metrics = _write_claim_scope_metrics(
        artifact_root, rows=rewritten, results=results,
    )
    print(
        f"[cot-split] mode=llm_claim_scope_v1 rows={len(rows)} "
        f"cached={metrics['cached_rows']} attempts={metrics['request_attempts']} "
        f"claims={metrics['total_claims']} scopes={metrics['total_scopes']}",
        flush=True,
    )
    return metrics


def _manifest_counts(rows: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    step_counts: list[int] = []
    claim_counts: list[int] = []
    for row in rows:
        steps = decode_steps(row.get("cot_manifest_json"))
        step_counts.append(len(steps))
        claim_counts.append(sum(len(step.get("claims") or []) for step in steps))
    return step_counts, claim_counts


def _distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"min": None, "max": None, "mean": None, "histogram": {}}
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "histogram": {
            str(value): count for value, count in sorted(Counter(values).items())
        },
    }


def _write_manifest_metrics(
    artifact_root: Path,
    *,
    mode: str,
    rows: list[dict[str, Any]],
    cached_rows: int = 0,
    request_attempts: int = 0,
) -> dict[str, Any]:
    step_counts, claim_counts = _manifest_counts(rows)
    claims_per_step: list[int] = []
    atoms_per_step: list[int] = []
    chars_per_step: list[int] = []
    layout_context_segments = 0
    scope_type_counts: Counter[str] = Counter()
    scoped_claims = 0
    steps_with_layout_context = 0
    rows_with_layout_context = 0
    for row in rows:
        row_has_layout = False
        for step in decode_steps(row.get("cot_manifest_json")):
            step_claims = step.get("claims") or []
            claims_per_step.append(len(step_claims))
            scoped_claims += sum(bool(claim.get("scope_ids")) for claim in step_claims)
            scope_type_counts.update(
                str(segment.get("scope_type") or "unknown")
                for segment in (step.get("segments") or [])
                if segment.get("scope_id")
            )
            atoms_per_step.append(len(step.get("atoms") or step.get("atom_ids") or []))
            chars_per_step.append(len(str(step.get("source_text") or "")))
            layout_count = sum(
                1 for segment in (step.get("segments") or [])
                if segment.get("kind") == "context"
                and segment.get("context_type") in {"heading", "table_layout", "list_layout"}
            )
            layout_context_segments += layout_count
            if layout_count:
                steps_with_layout_context += 1
                row_has_layout = True
        rows_with_layout_context += int(row_has_layout)
    payload = {
        "mode": mode,
        "splitter_version": (
            SPLITTER_VERSION if mode in LLM_MODES else "deterministic-v1"
        ),
        "rows": len(rows),
        "status_counts": {"ok": len(rows)},
        "cached_rows": cached_rows,
        "request_attempts": request_attempts,
        "fallback_rows": 0,
        "steps_per_row": _distribution(step_counts),
        "claims_per_row": _distribution(claim_counts),
        "claims_per_step": _distribution(claims_per_step),
        "atoms_per_step": _distribution(atoms_per_step),
        "chars_per_step": _distribution(chars_per_step),
        "layout_context_segments": layout_context_segments,
        "scope_count": sum(scope_type_counts.values()),
        "scope_type_counts": dict(sorted(scope_type_counts.items())),
        "scoped_claim_count": scoped_claims,
        "steps_with_layout_context": steps_with_layout_context,
        "rows_with_layout_context": rows_with_layout_context,
        "max_atoms_per_step": max(atoms_per_step) if atoms_per_step else None,
        "max_claims_per_step": max(claims_per_step) if claims_per_step else None,
        "steps_with_atoms_gt_4": sum(value > 4 for value in atoms_per_step),
        "steps_with_claims_gt_4": sum(value > 4 for value in claims_per_step),
        "steps_with_chars_gt_700": sum(value > 700 for value in chars_per_step),
        "total_steps": sum(step_counts),
        "total_claims": sum(claim_counts),
    }
    write_json(artifact_root / "manifest_metrics.json", payload)
    return payload


def _run_deterministic(
    config: DictConfig,
    rows: list[dict[str, Any]],
    artifact_root: Path,
    mode: str,
) -> dict[str, Any]:
    for row in rows:
        steps = decode_steps(row.get("cot_manifest_json"))
        if not steps:
            raise RuntimeError(f"deterministic manifest has no steps for {row.get('name')}")
        # This parses every declared step/claim hash and ID without rewriting
        # legacy JSONL/parquet bytes. Historical regex arms remain exact no-ops.
        parse_cot_manifest(row.get("cot_manifest_json"))
    return _write_manifest_metrics(
        artifact_root, mode=mode, rows=rows,
    )


def run_cot_split(config: DictConfig) -> dict[str, Any]:
    """Run the selected COT boundary policy and atomically install manifests."""
    rows = latest_rows(prepared_dir(config) / "generation_inputs.jsonl", "name")
    if not rows:
        raise RuntimeError("COT split stage requires non-empty prepared generation inputs")
    rows.sort(key=lambda row: (str(row.get("source") or ""), int(row.get("row_index", -1))))
    splitter = config.cot_splitter
    mode = str(splitter.get("mode", "deterministic")).strip().lower()
    artifact_root = prepared_dir(config) / "cot_splitter"

    if mode in DETERMINISTIC_MODES:
        metrics = _run_deterministic(config, rows, artifact_root, mode)
        print(
            f"[cot-split] mode={mode} rows={len(rows)} "
            f"steps={metrics['total_steps']} claims={metrics['total_claims']}",
            flush=True,
        )
        return metrics
    if mode == "llm_claim_scope_v1":
        if bool(splitter.get("fallback_on_error", False)):
            raise ValueError("Claim/Scope annotation forbids fallback_on_error=true")
        return _run_claim_scope(config, rows, artifact_root)
    if mode not in LLM_MODES:
        raise ValueError(f"unknown cot_splitter.mode: {mode!r}")
    if bool(splitter.get("fallback_on_error", False)):
        raise ValueError(
            "LLM boundary experiment forbids fallback_on_error=true because it mixes treatments"
        )

    results = asyncio.run(split_cot_rows(rows, splitter, artifact_root))
    failures = [
        result for result in results.values() if not result.ok
    ]
    if len(results) != len(rows) or failures:
        preview = ", ".join(
            f"{result.row_id}:{result.status}:{result.error}" for result in failures[:10]
        )
        raise RuntimeError(
            "LLM COT split did not produce a valid lossless partition for every row; "
            f"expected={len(rows)} results={len(results)} failures={len(failures)} {preview}"
        )

    rewritten: list[dict[str, Any]] = []
    for row in rows:
        row_id = str(row.get("name") or "")
        source = str(row.get("post_think_cot") or "")
        result = results[row_id]
        if "".join(section["source_text"] for section in result.sections) != source.strip():
            raise RuntimeError(f"lossless split invariant failed after routing for {row_id}")
        steps = build_cot_steps_from_sections(
            result.sections,
            structured_subclaims=True,
            splitter_mode=SPLITTER_VERSION,
        )
        if "".join(step["source_text"] for step in steps) != source.strip():
            raise RuntimeError(f"manifest source reconstruction failed for {row_id}")
        validate_split_manifest(source, steps)
        encoded_manifest = encode_steps(steps)
        # Exercise the same canonical validator on the serialized read path
        # before atomically replacing prepared JSONL/parquet artifacts.
        decode_steps(encoded_manifest, source=source)
        rewritten.append({
            **row,
            "cot_manifest_json": encoded_manifest,
            "cot_splitter_mode": mode,
            "cot_splitter_status": "ok",
            "cot_splitter_version": SPLITTER_VERSION,
            "cot_splitter_source_sha256": result.source_sha256,
            "cot_splitter_cache_key": result.cache_key,
            "cot_splitter_atom_count": len(result.atoms),
            "cot_splitter_step_count": len(steps),
            "cot_splitter_claim_count": sum(len(step.get("claims") or []) for step in steps),
            "cot_splitter_attempts": len(result.attempts),
            "cot_splitter_cached": result.cached,
            "cot_splitter_fallback": False,
            "cot_manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "cot_subclaim_builder_version": SUBCLAIM_BUILDER_VERSION,
            "cot_splitter_atomizer_version": str(
                splitter.get("atomizer_version", ATOMIZER_VERSION)
            ),
            "cot_splitter_prompt_version": str(
                splitter.get("prompt_version", PROMPT_VERSION)
            ),
            "cot_splitter_prompt_content_sha256": str(
                getattr(result, "prompt_content_sha256", "") or ""
            ),
        })

    # No output is replaced until every row has passed LLM-format, exact-span,
    # manifest-ID and full-source reconstruction checks above.
    write_generation_artifacts(config, rewritten)
    metrics = _write_manifest_metrics(
        artifact_root,
        mode=mode,
        rows=rewritten,
        cached_rows=sum(result.cached for result in results.values()),
        request_attempts=sum(
            len(result.attempts) for result in results.values() if not result.cached
        ),
    )
    print(
        f"[cot-split] mode={mode} rows={len(rows)} cached={metrics['cached_rows']} "
        f"attempts={metrics['request_attempts']} steps={metrics['total_steps']} "
        f"claims={metrics['total_claims']}",
        flush=True,
    )
    return metrics


__all__ = ["DETERMINISTIC_MODES", "LLM_MODES", "run_cot_split"]

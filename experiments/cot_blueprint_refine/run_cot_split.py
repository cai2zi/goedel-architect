"""Install the single formalization-aware Step manifest treatment."""
from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any

from omegaconf import DictConfig

from cot_blueprint_refine.common import latest_rows, prepared_dir, write_json
from cot_blueprint_refine.formal_step_splitter import (
    ANCHOR_VERSION,
    PROMPT_VERSION,
    SPLITTER_VERSION,
    split_formal_step_rows,
)
from cot_blueprint_refine.formal_steps import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    encode_formal_step_manifest,
    make_formal_step_manifest,
)
from cot_blueprint_refine.prepare_inputs import write_generation_artifacts


def _distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"min": None, "max": None, "mean": None, "histogram": {}}
    return {
        "min": min(values), "max": max(values), "mean": sum(values) / len(values),
        "histogram": {str(key): value for key, value in sorted(Counter(values).items())},
    }


def run_cot_split(config: DictConfig) -> dict[str, Any]:
    rows = latest_rows(prepared_dir(config) / "generation_inputs.jsonl", "name")
    if not rows:
        raise RuntimeError("formal Step split requires non-empty prepared inputs")
    rows.sort(key=lambda row: (str(row.get("source") or ""), int(row.get("row_index", -1))))
    artifact_root = prepared_dir(config) / "formal_step_splitter"
    results = asyncio.run(split_formal_step_rows(rows, config.step_splitter, artifact_root))
    failures = [result for result in results.values() if not result.ok]
    if len(results) != len(rows) or failures:
        preview = ", ".join(
            f"{item.row_id}:{item.error_type}:{item.error}" for item in failures[:10]
        )
        raise RuntimeError(
            "formal Step splitting failed; prepared artifacts were not replaced. "
            f"expected={len(rows)} results={len(results)} failures={len(failures)} {preview}"
        )
    rewritten = []
    step_counts = []
    chars_per_step = []
    outside_prior = 0
    for row in rows:
        row_id = str(row["name"])
        source = str(row["post_think_cot"])
        result = results[row_id]
        manifest = make_formal_step_manifest(source, result.spans)
        step_counts.append(len(manifest["steps"]))
        chars_per_step.extend(len(step["source_text"]) for step in manifest["steps"])
        outside_prior += int(
            len(manifest["steps"]) < result.recommended_min
            or len(manifest["steps"]) > result.recommended_max
        )
        rewritten.append({
            **row,
            "cot_manifest_json": encode_formal_step_manifest(manifest),
            "cot_step_splitter_status": "ok",
            "cot_step_splitter_version": SPLITTER_VERSION,
            "cot_step_splitter_prompt_version": PROMPT_VERSION,
            "cot_step_splitter_anchor_version": ANCHOR_VERSION,
            "cot_step_splitter_source_sha256": result.source_sha256,
            "cot_step_splitter_cache_key": result.cache_key,
            "cot_step_count": len(manifest["steps"]),
            "cot_step_target": result.target_steps,
            "cot_step_recommended_min": result.recommended_min,
            "cot_step_recommended_max": result.recommended_max,
            "cot_step_cot_tokens": result.cot_tokens,
            "cot_step_splitter_attempts": len(result.attempts),
            "cot_step_splitter_cached": result.cached,
            "cot_manifest_schema": SCHEMA_NAME,
            "cot_manifest_schema_version": SCHEMA_VERSION,
        })
    write_generation_artifacts(config, rewritten)
    metrics = {
        "schema": SCHEMA_NAME, "schema_version": SCHEMA_VERSION,
        "splitter_version": SPLITTER_VERSION, "prompt_version": PROMPT_VERSION,
        "anchor_version": ANCHOR_VERSION, "rows": len(rows),
        "cached_rows": sum(result.cached for result in results.values()),
        "request_attempts": sum(len(result.attempts) for result in results.values()),
        "steps_per_row": _distribution(step_counts),
        "chars_per_step": _distribution(chars_per_step),
        "total_steps": sum(step_counts), "outside_recommended_range": outside_prior,
        "exact_source_coverage_rows": len(rows),
    }
    write_json(artifact_root / "manifest_metrics.json", metrics)
    print(
        f"[formal-step-split] rows={len(rows)} steps={sum(step_counts)} "
        f"cached={metrics['cached_rows']} attempts={metrics['request_attempts']} "
        f"outside_prior={outside_prior}", flush=True,
    )
    return metrics


__all__ = ["run_cot_split"]

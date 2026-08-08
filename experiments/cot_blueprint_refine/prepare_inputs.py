from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig

from cot_blueprint_refine.common import (
    claimed_answer,
    extract_post_think,
    latest_rows,
    prepared_dir,
    restore_implicit_think_start,
    tag_counts,
    write_json,
    write_jsonl,
)
from cot_blueprint_refine.cot_steps import encode_steps, split_cot_steps


DATASET_SUBSET = "qwen3_8b_math_verify"
PARQUET_FIELDS = [
    "name", "source", "row_index", "problem", "claimed_answer",
    "post_think_cot", "informal_statement", "informal_proof",
    "cot_manifest_json",
]


def _safe_filename(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]", "_", value).strip("_")
    return text or "unknown"


def make_generation_row(row: dict[str, Any], post_think: str, answer: str) -> dict[str, Any]:
    problem = str(row.get("problem") or "").strip()
    cot_manifest_json = encode_steps(split_cot_steps(post_think))
    informal_statement = (
        "Original problem:\n"
        f"{problem}\n\n"
        "Claimed final answer from the original response:\n"
        f"\\boxed{{{answer}}}\n\n"
        "Formalize and verify the claim that this answer solves the original problem."
    )
    return {
        "name": str(row.get("ID") or ""),
        "source": str(row.get("source") or ""),
        "row_index": int(row.get("row_index", -1)),
        "problem": problem,
        "claimed_answer": answer,
        "post_think_cot": post_think,
        "informal_statement": informal_statement,
        # Keep the source proof byte-for-byte unchanged.  The numbered step
        # manifest travels beside it and is rendered only at prompt time.
        "informal_proof": post_think,
        "cot_manifest_json": cot_manifest_json,
    }


def _reject_row(row: dict[str, Any], reason: str) -> dict[str, Any]:
    raw_cot = str(row.get("raw_cot") or "")
    opens, closes = tag_counts(raw_cot)
    return {
        "ID": str(row.get("ID") or ""),
        "source": str(row.get("source") or ""),
        "row_index": int(row.get("row_index", -1)),
        "reason": reason,
        "finish_reason": row.get("finish_reason"),
        "think_open_count": opens,
        "think_close_count": closes,
    }


def write_generation_artifacts(config: DictConfig, rows: list[dict[str, Any]]) -> None:
    """Atomically keep prepared JSONL and parquet manifests in sync.

    The LLM boundary stage rewrites only ``cot_manifest_json`` after every row
    has passed exact-coverage validation.  Writing through temporary files keeps
    a failed/interrupted split from leaving RobustPA with a mixed manifest set.
    """
    root = prepared_dir(config)
    data_root = root / "data" / DATASET_SUBSET
    data_root.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source"])].append({key: row[key] for key in PARQUET_FIELDS})

    expected_paths: set[Path] = set()
    temporary_paths: list[tuple[Path, Path]] = []
    for source, source_rows in sorted(grouped.items()):
        path = data_root / f"{_safe_filename(source)}-00000-of-00001.parquet"
        temporary = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(pa.Table.from_pylist(source_rows), temporary)
        expected_paths.add(path)
        temporary_paths.append((temporary, path))

    jsonl_path = root / "generation_inputs.jsonl"
    jsonl_temporary = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
    write_jsonl(jsonl_temporary, rows)

    for temporary, path in temporary_paths:
        temporary.replace(path)
    for old_path in data_root.glob("*.parquet"):
        if old_path not in expected_paths:
            old_path.unlink()
    jsonl_temporary.replace(jsonl_path)


def prepare(config: DictConfig) -> dict[str, Any]:
    input_path = Path(str(config.input_predictions)).expanduser()
    raw_rows = latest_rows(input_path, "ID")
    requested_ids = [str(value) for value in (config.include_ids or [])]
    requested = set(requested_ids)
    available = {str(row.get("ID") or "") for row in raw_rows}
    missing_requested = sorted(requested - available)
    if missing_requested:
        raise ValueError(f"Requested IDs are missing from {input_path}: {missing_requested}")

    stats: Counter[str] = Counter()
    with input_path.open(encoding="utf-8") as handle:
        stats["input_lines"] = sum(1 for line in handle if line.strip())
    stats["unique_rows"] = len(raw_rows)
    stats["duplicate_rows"] = stats["input_lines"] - stats["unique_rows"]
    stats["implicit_think_start_restored"] = 0
    rejections: list[dict[str, Any]] = []
    eligible_all: list[dict[str, Any]] = []
    rejected_by_id: dict[str, str] = {}

    for row in raw_rows:
        row_id = str(row.get("ID") or "")
        raw_cot = str(row.get("raw_cot") or "")
        opens, closes = tag_counts(raw_cot)
        if str(row.get("status") or "") != "ok":
            reason = "status_not_ok"
        elif str(row.get("finish_reason") or "") == "length":
            reason = "finish_reason_length"
            stats["finish_reason_length"] += 1
            if opens > closes:
                stats["length_unclosed_think"] += 1
            elif opens == closes and opens > 0:
                stats["length_balanced_think"] += 1
            else:
                stats["length_other_think_shape"] += 1
        else:
            _normalized_cot, restored = restore_implicit_think_start(raw_cot)
            if restored:
                stats["implicit_think_start_restored"] += 1
            post_think, reason = extract_post_think(raw_cot)
            if not reason:
                answer = claimed_answer(post_think)
                if not answer:
                    reason = "missing_post_think_boxed_answer"
                else:
                    eligible_all.append(make_generation_row(row, post_think, answer))
                    stats["eligible_rows"] += 1
                    continue
        stats[f"rejected_{reason}"] += 1
        rejected_by_id[row_id] = reason
        rejections.append(_reject_row(row, reason))

    if requested:
        selected = [row for row in eligible_all if row["name"] in requested]
        rejected_requested = {
            row_id: rejected_by_id[row_id]
            for row_id in requested_ids
            if row_id in rejected_by_id
        }
        if rejected_requested:
            raise ValueError(f"Requested smoke IDs were rejected: {rejected_requested}")
    else:
        selected = list(eligible_all)

    selected.sort(key=lambda row: (row["source"], row["row_index"], row["name"]))
    stats["selected_rows"] = len(selected)
    stats["selected_requested_rows"] = len(requested_ids)

    root = prepared_dir(config)
    stats_payload = {
        **dict(sorted(stats.items())),
        "input_path": str(input_path),
        "data_root": str(root / "data"),
        "dataset_subset": DATASET_SUBSET,
        "requested_ids": requested_ids,
    }
    write_generation_artifacts(config, selected)
    write_jsonl(root / "rejections.jsonl", rejections)
    write_json(root / "preprocessing_stats.json", stats_payload)
    print("[prepare] " + " ".join(
        f"{key}={stats_payload.get(key, 0)}"
        for key in (
            "unique_rows", "finish_reason_length", "length_unclosed_think",
            "length_balanced_think", "implicit_think_start_restored",
            "eligible_rows", "selected_rows",
        )
    ), flush=True)
    print(f"[prepare] data_root={root / 'data'}", flush=True)
    return stats_payload

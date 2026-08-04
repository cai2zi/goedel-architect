from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from cot_blueprint_refine.common import (
    REPO_ROOT,
    claimed_answer,
    extract_post_think,
    latest_rows,
    output_root,
    read_jsonl,
    write_json,
    write_jsonl,
)


MATH_EVAL_ROOT = REPO_ROOT.parent / "math_verify_eval"
sys.path.insert(0, str(MATH_EVAL_ROOT))
from run_math_verify_eval import grade_response  # noqa: E402


def _global_eligible(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("status") or "") != "ok":
            continue
        if str(row.get("finish_reason") or "") == "length":
            continue
        post_think, reason = extract_post_think(str(row.get("raw_cot") or ""))
        if reason or not claimed_answer(post_think):
            continue
        eligible.append({**row, "post_think_cot": post_think})
    return eligible


def _source_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("source") or "")].append(row)
    result: dict[str, dict[str, Any]] = {}
    for source, items in sorted(grouped.items()):
        before = sum(bool(row.get("before_correct")) for row in items)
        after = sum(bool(row.get("after_correct")) for row in items)
        total = len(items)
        result[source] = {
            "total": total,
            "before_correct": before,
            "before_accuracy": before / total if total else 0.0,
            "after_correct": after,
            "after_accuracy": after / total if total else 0.0,
            "refined_ok": sum(row.get("refine_status") == "ok" for row in items),
        }
    return result


def summarize_comparisons(
    comparisons: list[dict[str, Any]],
    *,
    dataset_total: int,
    global_eligible_total: int,
    global_before_correct: int,
    historical_raw_correct: int,
) -> dict[str, Any]:
    total = len(comparisons)
    before_correct = sum(bool(row.get("before_correct")) for row in comparisons)
    after_correct = sum(bool(row.get("after_correct")) for row in comparisons)
    transitions = Counter(str(row.get("transition") or "unknown") for row in comparisons)
    node_status_counts: Counter[str] = Counter()
    for row in comparisons:
        node_status_counts.update(row.get("node_status_counts") or {})
    full_run = total == global_eligible_total
    context_quality_counts = Counter(
        str(row.get("context_quality") or "INFRA_ERROR") for row in comparisons
    )
    invalid_ids = sorted(
        str(row.get("ID") or "")
        for row in comparisons
        if row.get("context_quality") == "INVALID_BLUEPRINT_CANDIDATE"
    )
    infra_ids = sorted(
        str(row.get("ID") or "")
        for row in comparisons
        if row.get("context_quality") == "INFRA_ERROR"
    )
    truncated_ids = sorted(
        str(row.get("ID") or "")
        for row in comparisons
        if bool(row.get("blueprint_truncated"))
    )
    return {
        "dataset": {
            "total": dataset_total,
            "historical_raw_correct": historical_raw_correct,
            "historical_raw_accuracy": historical_raw_correct / dataset_total if dataset_total else 0.0,
            "global_eligible_total": global_eligible_total,
            "strict_post_think_before_correct": global_before_correct,
            "strict_post_think_before_eligible_accuracy": (
                global_before_correct / global_eligible_total if global_eligible_total else 0.0
            ),
            "strict_post_think_before_full_accuracy": (
                global_before_correct / dataset_total if dataset_total else 0.0
            ),
        },
        "selected": {
            "total": total,
            "before_correct": before_correct,
            "before_accuracy": before_correct / total if total else 0.0,
            "after_correct": after_correct,
            "after_accuracy": after_correct / total if total else 0.0,
            "accuracy_delta": (after_correct - before_correct) / total if total else 0.0,
            "blueprint_ready": sum(row.get("blueprint_status") == "ready" for row in comparisons),
            "refined_ok": sum(row.get("refine_status") == "ok" for row in comparisons),
            "transitions": dict(sorted(transitions.items())),
            "node_status_counts": dict(sorted(node_status_counts.items())),
            "context_quality_counts": dict(sorted(context_quality_counts.items())),
            "invalid_blueprint_candidate_ids": invalid_ids,
            "infra_error_ids": infra_ids,
            "blueprint_truncated_count": len(truncated_ids),
            "blueprint_truncated_ids": truncated_ids,
        },
        "full_after": {
            "available": full_run,
            "correct": after_correct if full_run else None,
            "eligible_accuracy": after_correct / global_eligible_total if full_run and global_eligible_total else None,
            "full_accuracy": after_correct / dataset_total if full_run and dataset_total else None,
            "incorrect_or_missing_count": dataset_total - after_correct if full_run else None,
            "unavailable_reason": "" if full_run else "partial/smoke run does not cover all eligible rows",
        },
        "by_source": _source_metrics(comparisons),
    }


def evaluate(config: DictConfig) -> dict[str, Any]:
    root = output_root(config)
    prediction_path = Path(str(config.input_predictions)).expanduser()
    original_rows = latest_rows(prediction_path, "ID")
    original_by_id = {str(row["ID"]): row for row in original_rows}
    eligible_rows = _global_eligible(original_rows)
    global_before_correct = sum(
        grade_response(str(row.get("gold") or ""), str(row["post_think_cot"]))["is_correct"]
        for row in eligible_rows
    )

    generation_rows = latest_rows(root / "prepared" / "generation_inputs.jsonl", "name")
    contexts = {
        str(row.get("ID") or ""): row
        for row in latest_rows(root / "blueprint_contexts" / "blueprint_contexts.jsonl", "ID")
    }
    refined = {
        str(row.get("ID") or ""): row
        for row in latest_rows(root / "refinement" / "refined_predictions.jsonl", "ID")
    }
    comparisons: list[dict[str, Any]] = []
    for generation in generation_rows:
        row_id = str(generation.get("name") or "")
        original = original_by_id.get(row_id)
        if original is None:
            raise ValueError(f"original prediction missing during evaluation: {row_id}")
        before = grade_response(
            str(original.get("gold") or ""), str(generation.get("post_think_cot") or "")
        )
        context = contexts.get(row_id, {})
        refinement = refined.get(row_id, {})
        after = (
            grade_response(str(original.get("gold") or ""), str(refinement.get("refined_cot") or ""))
            if refinement.get("status") == "ok"
            else {"is_correct": False, "math_verify_parse_ok": False, "extracted_pred": []}
        )
        before_correct = bool(before["is_correct"])
        after_correct = bool(after["is_correct"])
        transition = (
            "correct_to_correct" if before_correct and after_correct
            else "correct_to_wrong" if before_correct
            else "wrong_to_correct" if after_correct
            else "wrong_to_wrong"
        )
        node_counts = Counter(
            str(node.get("prompt_signal") or "") for node in (context.get("nodes") or [])
        )
        comparisons.append({
            "ID": row_id,
            "source": str(original.get("source") or ""),
            "problem": str(original.get("problem") or ""),
            "gold": str(original.get("gold") or ""),
            "claimed_answer": str(generation.get("claimed_answer") or ""),
            "before_correct": before_correct,
            "after_correct": after_correct,
            "transition": transition,
            "before_parse_ok": bool(before.get("math_verify_parse_ok")),
            "after_parse_ok": bool(after.get("math_verify_parse_ok")),
            "before_extracted_pred": before.get("extracted_pred", []),
            "after_extracted_pred": after.get("extracted_pred", []),
            "blueprint_status": str(context.get("status") or "missing"),
            "context_quality": str(context.get("context_quality") or "INFRA_ERROR"),
            "root_proved": bool(context.get("root_proved")),
            "refine_status": str(refinement.get("status") or "missing"),
            "blueprint_truncated": bool(refinement.get("blueprint_truncated")),
            "refined_cot": str(refinement.get("refined_cot") or ""),
            "node_status_counts": dict(sorted(node_counts.items())),
        })
    comparisons.sort(key=lambda row: str(row["ID"]))

    metrics = summarize_comparisons(
        comparisons,
        dataset_total=len(original_rows),
        global_eligible_total=len(eligible_rows),
        global_before_correct=global_before_correct,
        historical_raw_correct=sum(bool(row.get("is_correct")) for row in original_rows),
    )
    evaluation_dir = root / "evaluation"
    write_jsonl(evaluation_dir / "comparison.jsonl", comparisons)
    write_json(evaluation_dir / "metrics.json", metrics)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    with (evaluation_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ID", "source", "before_correct", "after_correct", "transition",
                "blueprint_status", "context_quality", "root_proved", "refine_status",
                "blueprint_truncated",
            ],
        )
        writer.writeheader()
        for row in comparisons:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})
    with (evaluation_dir / "metrics_by_source.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "source", "total", "before_correct", "before_accuracy",
            "after_correct", "after_accuracy", "refined_ok",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for source, values in metrics["by_source"].items():
            writer.writerow({"source": source, **values})
    print(
        f"[evaluate] selected={metrics['selected']['total']} "
        f"before={metrics['selected']['before_accuracy']:.6f} "
        f"after={metrics['selected']['after_accuracy']:.6f} "
        f"full_after_available={metrics['full_after']['available']}",
        flush=True,
    )
    print(
        f"[evaluate-context] quality={metrics['selected']['context_quality_counts']} "
        f"invalid_ids={metrics['selected']['invalid_blueprint_candidate_ids']} "
        f"infra_ids={metrics['selected']['infra_error_ids']} "
        f"truncated_ids={metrics['selected']['blueprint_truncated_ids']}",
        flush=True,
    )
    return metrics

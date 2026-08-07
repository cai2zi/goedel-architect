from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from cot_blueprint_refine.common import (
    claimed_answer,
    extract_post_think,
    read_jsonl,
    write_json,
    write_jsonl,
)
from cot_blueprint_refine.evaluate import grade_final_answer


SCORING_MODE = "canonical_last_boxed_answer_math_verify"


def extract_incorrect_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen: set[str] = set()

    for row in rows:
        counts["input_rows"] += 1
        row_id = str(row.get("ID") or "")
        if not row_id:
            counts["rejected_missing_id"] += 1
            continue
        if row_id in seen:
            counts["rejected_duplicate_id"] += 1
            continue
        seen.add(row_id)
        if str(row.get("status") or "") != "ok":
            counts["rejected_status_not_ok"] += 1
            continue
        if str(row.get("finish_reason") or "") == "length":
            counts["rejected_finish_reason_length"] += 1
            continue
        post_think, reason = extract_post_think(str(row.get("raw_cot") or ""))
        if reason:
            counts[f"rejected_{reason}"] += 1
            continue
        answer = claimed_answer(post_think)
        if not answer:
            counts["rejected_missing_post_think_boxed_answer"] += 1
            continue

        grade = grade_final_answer(str(row.get("gold") or ""), answer)
        counts["eligible_rows"] += 1
        if bool(grade["is_correct"]):
            counts["correct_rows"] += 1
            continue

        counts["incorrect_rows"] += 1
        selected.append({
            **row,
            "subset_selection": {
                "scoring_mode": SCORING_MODE,
                "claimed_answer": answer,
                "math_verify_parse_ok": bool(grade["math_verify_parse_ok"]),
                "is_correct": False,
            },
        })

    selected.sort(key=lambda row: (
        str(row.get("source") or ""),
        int(row.get("row_index", -1)),
        str(row.get("ID") or ""),
    ))
    metrics = {
        **dict(sorted(counts.items())),
        "scoring_mode": SCORING_MODE,
        "selected_ids": [str(row["ID"]) for row in selected],
    }
    return selected, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract eligible rows whose canonical final COT answer fails math_verify."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metrics", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected, metrics = extract_incorrect_rows(read_jsonl(args.input))
    metrics_path = args.metrics or args.output.with_name("metrics.json")
    write_jsonl(args.output, selected)
    write_json(metrics_path, {
        **metrics,
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
    })
    print(
        "[incorrect-subset] "
        f"input={metrics.get('input_rows', 0)} eligible={metrics.get('eligible_rows', 0)} "
        f"incorrect={metrics.get('incorrect_rows', 0)} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()

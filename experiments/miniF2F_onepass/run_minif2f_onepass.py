from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from shared.io_utils import (  # noqa: E402
    append_jsonl,
    default_output_root,
    read_jsonl,
    rows_by_id,
    safe_stem,
    unlink_if_exists,
    write_json,
)
from shared.onepass import run_onepass_record  # noqa: E402


DEFAULT_MODEL = "deepseek-v4-flash"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run miniF2F one-pass blueprint/proof experiment.")
    parser.add_argument("--split", choices=["test", "valid"], default="test")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "minif2f")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--problem-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--node-timeout-s", type=int, default=300)
    return parser.parse_args()


def _problem_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("name") or row.get("id") or row.get("problem_id") or f"miniF2F_{index}")


def _theorem_stmt(row: dict[str, Any]) -> str:
    value = row.get("formal_statement") or row.get("statement") or row.get("theorem")
    if not value:
        raise ValueError("miniF2F row has no formal_statement/statement/theorem field")
    return str(value)


def _nl_proof(row: dict[str, Any]) -> str:
    return str(row.get("informal_proof") or row.get("proof") or "")


def _select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    selected: list[tuple[int, dict[str, Any]]] = []
    for idx, row in enumerate(rows, 1):
        problem_id = _problem_id(row, idx)
        if args.problem_id and problem_id != args.problem_id:
            continue
        selected.append((idx, row))
        if args.limit is not None and len(selected) >= args.limit:
            break
    return selected


def main() -> None:
    args = parse_args()
    output_root = args.output_root or default_output_root(REPO_ROOT, "miniF2F_onepass", args.model) / args.split
    output_root.mkdir(parents=True, exist_ok=True)
    data_path = args.data_dir / f"{args.split}.jsonl"
    rows = read_jsonl(data_path)
    selected = _select_rows(rows, args)

    results_path = output_root / "results.jsonl"
    metrics_path = output_root / "metrics.json"
    metrics_csv_path = output_root / "metrics.csv"
    if not args.resume:
        for path in (results_path, metrics_path, metrics_csv_path):
            unlink_if_exists(path)

    done = rows_by_id(results_path)
    print(f"[select] split={args.split} problems={len(selected)} output={output_root}")
    for idx, row in selected:
        source_id = _problem_id(row, idx)
        record_id = safe_stem(source_id, prefix="miniF2F_")
        if args.resume and record_id in done:
            print(f"[resume] skip completed {record_id}")
            continue

        try:
            onepass = run_onepass_record(
                record_id=record_id,
                theorem_stmt=_theorem_stmt(row),
                nl_proof=_nl_proof(row),
                model=args.model,
                output_root=output_root,
                node_timeout_s=args.node_timeout_s,
                resume=args.resume,
            )
            result = {
                **onepass,
                "source_id": source_id,
                "split": args.split,
                "phase0_success": True,
                "success": bool(onepass.get("root_proved")),
            }
        except Exception as exc:
            result = {
                "id": record_id,
                "source_id": source_id,
                "split": args.split,
                "phase0_success": True,
                "blueprint_success": False,
                "root_proved": False,
                "total_nodes": 0,
                "proved_node_count": 0,
                "proved_ratio": 0.0,
                "failed_nodes": [],
                "success": False,
                "error": str(exc),
            }
        append_jsonl(results_path, result)
        done[record_id] = result

    result_rows = [row for row in done.values() if not args.problem_id or row.get("source_id") == args.problem_id]
    if args.limit is not None:
        selected_ids = {safe_stem(_problem_id(row, idx), prefix="miniF2F_") for idx, row in selected}
        result_rows = [row for row in result_rows if row.get("id") in selected_ids]
    total = len(result_rows)
    solved = sum(1 for row in result_rows if row.get("success"))
    metrics = {
        "split": args.split,
        "problem_count": total,
        "solved_count": solved,
        "accuracy": solved / total if total else 0.0,
        "root_proved_count": sum(1 for row in result_rows if row.get("root_proved")),
        "blueprint_success_count": sum(1 for row in result_rows if row.get("blueprint_success")),
    }
    write_json(metrics_path, metrics)
    with metrics_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)

    print(f"[done] results={results_path}")
    print(f"[metrics] {metrics}")


if __name__ == "__main__":
    main()


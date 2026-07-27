from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_DATA_ROOT = Path("/ssd/czx/czx_work/RobustPABench")
DEFAULT_PROBLEM_IDS = [
    "aime_1988_p8",
    "amc12a_2021_p12",
    "mathd_algebra_170",
    "aime_1994_p3",
]


def _row_id(row: dict[str, Any]) -> str:
    return str(row.get("name") or row.get("id") or "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a small RobustPABench parquet subset.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--source-subset", default="global_original")
    parser.add_argument("--target-subset", default="tool_burst_debug")
    parser.add_argument("--split", default="miniF2F")
    parser.add_argument("--problem-id", action="append", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    problem_ids = args.problem_id or DEFAULT_PROBLEM_IDS
    source_path = args.data_root / args.source_subset / f"{args.split}-00000-of-00001.parquet"
    target_dir = args.data_root / args.target_subset
    target_path = target_dir / f"{args.split}-00000-of-00001.parquet"

    table = pq.read_table(source_path)
    rows = table.to_pylist()
    selected = [row for row in rows if _row_id(row) in set(problem_ids)]
    found = {_row_id(row) for row in selected}
    missing = [problem_id for problem_id in problem_ids if problem_id not in found]
    if missing:
        raise SystemExit(f"Missing problem ids in {source_path}: {', '.join(missing)}")

    selected.sort(key=lambda row: problem_ids.index(_row_id(row)))
    target_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(selected, schema=table.schema), target_path)

    print(f"wrote {target_path}")
    print(f"rows={len(selected)}")
    for row in selected:
        print(_row_id(row))


if __name__ == "__main__":
    main()

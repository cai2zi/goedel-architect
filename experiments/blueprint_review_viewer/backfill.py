"""Explicitly create review artifacts for a completed legacy experiment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from blueprint_review_viewer.review_schema import index_entry, write_review_artifact  # noqa: E402
from blueprint_review_viewer.server import DEFAULT_OUTPUT_BASE, experiment_root  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill readonly Blueprint review artifacts; never changes Lean candidates.")
    parser.add_argument("experiment_name", help="experiment directory name under --output-base")
    parser.add_argument(
        "--output-base",
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
        help=f"experiment output base (default: {DEFAULT_OUTPUT_BASE})",
    )
    args = parser.parse_args()
    try:
        root = experiment_root(args.experiment_name, args.output_base)
    except ValueError as error:
        parser.error(str(error))
    results = root / "results.jsonl"
    if not results.is_file():
        raise SystemExit(
            f"experiment {args.experiment_name!r} has no Blueprint results: {results}"
        )
    entries = []
    for line in results.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        path, artifact = write_review_artifact(root, row)
        entries.append(index_entry(path, root, artifact))
    index = root / "review_index.jsonl"
    index.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in entries), encoding="utf-8")
    print(f"[review-backfill] artifacts={len(entries)} index={index}")


if __name__ == "__main__":
    main()

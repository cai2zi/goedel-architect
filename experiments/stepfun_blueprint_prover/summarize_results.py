from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for path in (str(REPO_ROOT / "src"), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from input_loader import load_accepted_blueprints
from run_experiment import RecordRuntime, result_row, summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    sources = load_accepted_blueprints(args.source_root.resolve())
    runtimes = []
    for source in sources:
        checkpoint = (
            args.output_root.resolve() / "checkpoints" / source.subset / source.split
            / f"{__import__('run_experiment').safe_name(source.record_id)}.json"
        )
        if checkpoint.is_file():
            runtimes.append(RecordRuntime(source, checkpoint, json.loads(checkpoint.read_text())))
    rows = [result_row(runtime) for runtime in runtimes]
    print(json.dumps(summarize(rows, runtimes), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

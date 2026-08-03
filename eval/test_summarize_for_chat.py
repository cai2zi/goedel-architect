from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from robustpa_refine.summarize_for_chat import (  # noqa: E402
    _format_new_success_by_refinement_iteration,
    _new_success_by_refinement_iteration,
    aggregate_experiment,
)


class SummarizeForChatTest(unittest.TestCase):
    def test_path_artifacts_do_not_collide_across_subsets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            exp_dir = Path(directory)
            record_id = "robustpa_shared"
            split = "MATH500"
            results = []
            expected: dict[str, dict[str, object]] = {}

            for index, subset in enumerate(("subset_a", "subset_b"), 1):
                unique_id = f"{subset}__{split}__{record_id}"
                result = {
                    "id": unique_id,
                    "record_id": record_id,
                    "source_id": f"source_{index}",
                    "subset": subset,
                    "split": split,
                    "theorem_name": f"theorem_{index}",
                    "status": "solved" if index == 1 else "exhausted",
                    "root_proved": index == 1,
                    "iterations": index - 1,
                }
                results.append(result)
                expected[subset] = result

                trace_path = exp_dir / "traces" / subset / split / f"{record_id}.jsonl"
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                trace_path.write_text(
                    json.dumps({"kind": "theorem_start", "thm_name": unique_id}) + "\n",
                    encoding="utf-8",
                )

                checkpoint_path = exp_dir / "checkpoints" / subset / split / f"{record_id}.json"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_path.write_text(json.dumps({"iteration": index}), encoding="utf-8")

                blueprint_path = (
                    exp_dir
                    / "blueprints"
                    / subset
                    / split
                    / record_id
                    / "round_000_phase1.lean"
                )
                blueprint_path.parent.mkdir(parents=True, exist_ok=True)
                blueprint_path.write_text("theorem root : True := by trivial\n", encoding="utf-8")

            (exp_dir / "results.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in results),
                encoding="utf-8",
            )

            rows = aggregate_experiment(exp_dir)

        path_artifacts = [
            row
            for row in rows
            if row["artifact_type"] in {"trace_event", "checkpoint", "blueprint"}
        ]
        self.assertEqual(len(path_artifacts), 6)
        for row in path_artifacts:
            subset = row["rel_path"].split("/")[1]
            result = expected[subset]
            self.assertEqual(row["id"], result["id"])
            self.assertEqual(row["record_id"], record_id)
            self.assertEqual(row["source_id"], result["source_id"])
            self.assertEqual(row["subset"], subset)
            self.assertEqual(row["split"], split)
            self.assertEqual(row["theorem_name"], result["theorem_name"])
            self.assertEqual(row["root_proved"], str(result["root_proved"]))
        result_iterations = {
            row["subset"]: row["iteration"]
            for row in rows
            if row["artifact_type"] == "result"
        }
        self.assertEqual(result_iterations, {"subset_a": 0, "subset_b": 1})

    def test_new_success_table_uses_result_iterations(self) -> None:
        rows = [
            {"artifact_type": "result", "iteration": 0, "root_proved": "True"},
            {"artifact_type": "result", "iteration": 1, "root_proved": "true"},
            {"artifact_type": "result", "iteration": 1, "root_proved": "False"},
            {"artifact_type": "result", "iteration": 3, "root_proved": "True"},
            {"artifact_type": "round", "iteration": 2, "root_proved": "True"},
        ]
        self.assertEqual(
            _new_success_by_refinement_iteration(rows),
            [
                {"refinement_iterations": 0, "new_success_count": 1},
                {"refinement_iterations": 1, "new_success_count": 1},
                {"refinement_iterations": 2, "new_success_count": 0},
                {"refinement_iterations": 3, "new_success_count": 1},
            ],
        )
        self.assertIn(
            "| new_success_count | 1 | 1 | 0 | 1 |",
            _format_new_success_by_refinement_iteration(rows),
        )


if __name__ == "__main__":
    unittest.main()

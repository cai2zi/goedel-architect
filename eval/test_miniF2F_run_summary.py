from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "miniF2F_onepass"))

from summarize_run import summarize_run  # noqa: E402


try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pq = None


class MiniF2FRunSummaryTest(unittest.TestCase):
    @unittest.skipIf(pq is None, "pyarrow is required for parquet summaries")
    def test_summary_outputs_parquet_and_samples_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "traces").mkdir()
            (root / "checkpoints").mkdir()
            (root / "blueprints").mkdir()
            (root / "metrics.json").write_text(json.dumps({"split": "test"}), encoding="utf-8")
            (root / "results.jsonl").write_text(
                "\n".join([
                    json.dumps({"id": "sample_good", "success": False, "root_proved": False}),
                    json.dumps({
                        "id": "sample_good",
                        "source_id": "good",
                        "split": "test",
                        "success": True,
                        "root_proved": True,
                        "blueprint_success": True,
                        "total_nodes": 1,
                        "proved_node_count": 1,
                        "proved_ratio": 1.0,
                    }),
                    json.dumps({
                        "id": "sample_bad",
                        "source_id": "bad",
                        "split": "test",
                        "success": False,
                        "root_proved": False,
                        "blueprint_success": True,
                        "total_nodes": 1,
                        "proved_node_count": 0,
                        "proved_ratio": 0.0,
                    }),
                ])
                + "\n",
                encoding="utf-8",
            )
            (root / "traces" / "sample_good.jsonl").write_text(
                "\n".join([
                    json.dumps({
                        "kind": "llm_usage",
                        "thm_name": "sample_good",
                        "turn": 0,
                        "args": {
                            "phase": "phase1",
                            "model": "gpt-test",
                            "prompt_tokens": 2,
                            "completion_tokens": 3,
                            "total_tokens": 5,
                        },
                        "ts": 100.0,
                    }),
                    json.dumps({
                        "kind": "lean_check_result",
                        "thm_name": "sample_good",
                        "turn": 0,
                        "args": {
                            "phase": "phase1",
                            "attempt": 2,
                            "target": "sample_good",
                            "errors": [],
                            "warnings": [],
                            "goals": [],
                            "validated": True,
                        },
                        "ok": True,
                        "ts": 101.0,
                    }),
                ])
                + "\n",
                encoding="utf-8",
            )
            (root / "traces" / "sample_bad.jsonl").write_text(
                json.dumps({
                    "kind": "tool_result",
                    "thm_name": "sample_bad",
                    "turn": 1,
                    "tool_name": "lean_compile",
                    "args": {
                        "success": False,
                        "errors": ["Proof contains `sorry`"],
                        "warnings": [],
                        "goals": [],
                        "validated": True,
                    },
                    "ok": False,
                    "ts": 200.0,
                })
                + "\n",
                encoding="utf-8",
            )
            valid_blueprint = (
                "import Mathlib\nimport Architect\n\n"
                "@[blueprint\n"
                "  (statement := /-- True. -/)\n"
                "  (proof := /-- Trivial. -/)]\n"
                "theorem sample_good : True := by sorry_using []\n"
            )
            invalid_blueprint = (
                "import Mathlib\nimport Architect\n\n"
                "@[blueprint\n"
                "  (statement := /-- True. -/)\n"
                "  (proof := /-- Trivial. -/)]\n"
                "theorem sample_bad : True := by trivial\n"
            )
            (root / "checkpoints" / "sample_good.json").write_text(
                json.dumps({
                    "theorem_stmt": "theorem sample_good : True := by",
                    "blueprint_lean_file": valid_blueprint,
                    "blueprint_target": "sample_good",
                    "blueprint_fully_validated": True,
                    "node_results": {"sample_good": {"signal": "solved", "lean_errors": []}},
                    "done": True,
                    "success": True,
                }),
                encoding="utf-8",
            )
            (root / "checkpoints" / "sample_bad.json").write_text(
                json.dumps({
                    "theorem_stmt": "theorem sample_bad : True := by",
                    "blueprint_lean_file": invalid_blueprint,
                    "blueprint_target": "sample_bad",
                    "blueprint_fully_validated": True,
                    "node_results": {
                        "sample_bad": {
                            "signal": "proof_too_hard",
                            "lean_errors": ["Proof contains `sorry`"],
                        },
                    },
                    "done": True,
                    "success": False,
                }),
                encoding="utf-8",
            )

            summary = summarize_run(root, experiment_id="exp.test", timestamp="20260720_120000")

            events_path = Path(summary["events_path"])
            samples_path = Path(summary["samples_path"])
            self.assertTrue(events_path.exists())
            self.assertTrue(samples_path.exists())
            self.assertIn("exp.test", events_path.name)
            self.assertEqual(summary["event_count"], 3)

            table = pq.read_table(events_path)
            self.assertEqual(table.num_rows, 3)
            self.assertIn("args_json", table.column_names)

            with samples_path.open(encoding="utf-8", newline="") as f:
                rows = {row["sample_id"]: row for row in csv.DictReader(f)}
            self.assertEqual(rows["sample_good"]["final_success"], "True")
            self.assertEqual(rows["sample_bad"]["failure_category"], "code_missing_sorry_using_placeholder")
            self.assertEqual(rows["sample_good"]["total_tokens"], "5")


if __name__ == "__main__":
    unittest.main()

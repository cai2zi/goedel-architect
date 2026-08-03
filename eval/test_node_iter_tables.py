from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from robustpa_refine.node_iter_tables import (  # noqa: E402
    _proof_turns,
    collect_node_attempts,
    collect_problem_attempts,
    node_compile_table,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class NodeIterTablesTest(unittest.TestCase):
    def test_current_phase2_events_and_cross_subset_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            exp_dir = Path(directory)
            record_id = "robustpa_shared"
            split = "MATH500"
            results = []
            expected_spans = {"subset_a": 10.0, "subset_b": 100.0}

            for subset, ok in (("subset_a", True), ("subset_b", False)):
                unique_id = f"{subset}__{split}__{record_id}"
                trace_path = exp_dir / "traces" / subset / split / f"{record_id}.jsonl"
                rows = [
                    {
                        "kind": "theorem_start",
                        "phase": "phase2",
                        "iteration": 0,
                        "thm_name": "root",
                        "turn": 0,
                        "ts": 10.0,
                    },
                    {
                        "kind": "llm_usage",
                        "phase": "phase2",
                        "iteration": 0,
                        "thm_name": "root",
                        "turn": 1,
                        "args": {"stage": "prove", "total_tokens": 100},
                        "ts": 11.0,
                    },
                ]
                if not ok:
                    rows.extend(
                        [
                            {
                                "kind": "tool_result",
                                "phase": "phase2",
                                "iteration": 0,
                                "thm_name": "root",
                                "turn": 2,
                                "args": {"stage": "prove"},
                                "ok": False,
                                "ts": 12.0,
                            },
                            {
                                "kind": "llm_usage",
                                "phase": "phase2",
                                "iteration": 0,
                                "thm_name": "root",
                                "turn": 1,
                                "args": {"stage": "negation_probe", "total_tokens": 50},
                                "ts": 13.0,
                            },
                        ]
                    )
                rows.append(
                    {
                        "kind": "node_finished",
                        "phase": "phase2",
                        "iteration": 0,
                        "thm_name": "root",
                        "turn": 0,
                        "args": {
                            "signal": "solved" if ok else "proof_too_hard",
                            "wall_time_s": 5.0,
                        },
                        "ok": ok,
                        "ts": 10.0 + expected_spans[subset],
                    }
                )
                _write_jsonl(trace_path, rows)
                results.append(
                    {
                        "id": unique_id,
                        "record_id": record_id,
                        "subset": subset,
                        "split": split,
                        "iterations": 0,
                        "root_proved": ok,
                        "trace_path": str(trace_path),
                    }
                )

            _write_jsonl(exp_dir / "results.jsonl", results)
            attempts, incomplete, trace_spans = collect_node_attempts(exp_dir)
            problems, missing, trace_only = collect_problem_attempts(exp_dir, trace_spans)

        self.assertEqual(incomplete, [])
        self.assertEqual(missing, [])
        self.assertEqual(trace_only, [])
        self.assertEqual(len(attempts), 2)
        by_subset = {attempt.unique_id.split("__", 1)[0]: attempt for attempt in attempts}
        self.assertEqual(by_subset["subset_a"].proof_turn, 1)
        self.assertEqual(by_subset["subset_b"].proof_turn, 2)
        self.assertEqual(by_subset["subset_a"].signal, "solved")
        self.assertEqual(by_subset["subset_b"].signal, "proof_too_hard")
        self.assertEqual(by_subset["subset_b"].total_tokens, 150)
        self.assertEqual(_proof_turns(attempts), [1, 2])

        problem_spans = {
            problem.unique_id.split("__", 1)[0]: problem.wall_time_s
            for problem in problems
        }
        self.assertEqual(problem_spans, expected_spans)
        table = node_compile_table(attempts, [1, 2])
        self.assertIn("| success | 1 | 0 |", table)
        self.assertIn("| failure | 0 | 1 |", table)


if __name__ == "__main__":
    unittest.main()

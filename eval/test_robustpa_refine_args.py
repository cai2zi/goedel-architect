from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from robustpa_refine.run_robustpa_refine import (  # noqa: E402
    _format_new_success_by_refinement_iteration,
    _metric_row,
    _new_success_by_refinement_iteration,
    _should_skip_existing,
    _validate_args,
)


class RobustPAConfigTest(unittest.TestCase):
    def test_base_config_is_kimina_only_with_new_tool_budget(self) -> None:
        config = OmegaConf.load(
            REPO_ROOT / "experiments/robustpa_refine/configs/base.yaml"
        )
        self.assertNotIn("lean_backend", config)
        self.assertNotIn("parallel_tool_calls", config)
        self.assertEqual(config.node_max_negation_probe_turns, 1)
        self.assertEqual(config.max_tool_calls_per_turn, 3)

    def test_tool_budget_must_be_positive(self) -> None:
        config = OmegaConf.load(
            REPO_ROOT / "experiments/robustpa_refine/configs/base.yaml"
        )
        config.max_tool_calls_per_turn = 0
        with self.assertRaisesRegex(ValueError, "max_tool_calls_per_turn"):
            _validate_args(SimpleNamespace(**OmegaConf.to_container(config, resolve=False)))

    def test_negation_probe_turns_may_be_disabled(self) -> None:
        config = OmegaConf.load(
            REPO_ROOT / "experiments/robustpa_refine/configs/base.yaml"
        )
        config.node_max_negation_probe_turns = 0
        _validate_args(SimpleNamespace(**OmegaConf.to_container(config, resolve=False)))
        config.node_max_negation_probe_turns = -1
        with self.assertRaisesRegex(ValueError, "node_max_negation_probe_turns"):
            _validate_args(SimpleNamespace(**OmegaConf.to_container(config, resolve=False)))

    def test_execution_mode_is_validated(self) -> None:
        config = OmegaConf.load(
            REPO_ROOT / "experiments/robustpa_refine/configs/base.yaml"
        )
        config.execution_mode = "phase1_only"
        _validate_args(SimpleNamespace(**OmegaConf.to_container(config, resolve=False)))
        config.execution_mode = "unknown"
        with self.assertRaisesRegex(ValueError, "execution_mode"):
            _validate_args(SimpleNamespace(**OmegaConf.to_container(config, resolve=False)))

    def test_whole_cot_semantic_source_mode_has_consistent_flags(self) -> None:
        config = OmegaConf.load(
            REPO_ROOT / "experiments/robustpa_refine/configs/base.yaml"
        )
        config.semantic_fidelity_enabled = True
        config.semantic_source_mode = "whole_cot"
        config.semantic_require_step_ids = False
        _validate_args(SimpleNamespace(**OmegaConf.to_container(config, resolve=False)))
        config.semantic_require_step_ids = True
        with self.assertRaisesRegex(ValueError, "cannot require Step IDs"):
            _validate_args(SimpleNamespace(**OmegaConf.to_container(config, resolve=False)))

    def test_resume_treats_terminal_phase1_results_and_errors_as_complete(self) -> None:
        args = SimpleNamespace(resume=True, retry_error_results=False)
        self.assertTrue(_should_skip_existing({"status": "phase1_accepted"}, args))
        self.assertTrue(_should_skip_existing({"status": "exhausted"}, args))
        self.assertTrue(_should_skip_existing({"status": "error"}, args))
        self.assertTrue(_should_skip_existing({"root_proved": True}, args))
        self.assertFalse(_should_skip_existing(None, args))

        args.retry_error_results = True
        self.assertFalse(_should_skip_existing({"status": "error"}, args))

    def test_phase1_acceptance_reports_warning_partition(self) -> None:
        metrics = _metric_row("global", [
            {"status": "phase1_accepted", "semantic_warning_codes": []},
            {"status": "phase1_accepted",
             "semantic_warning_codes": ["nodeNotRootReachable"]},
            {"status": "error", "phase": "phase1",
             "semantic_warning_codes": ["stepNotRootReachable"]},
        ])
        self.assertEqual(metrics["phase1_accepted"], 2)
        self.assertEqual(metrics["phase1_accepted_with_warnings"], 1)
        self.assertEqual(metrics["phase1_accepted_without_warnings"], 1)

    def test_new_successes_are_bucketed_by_refinement_count(self) -> None:
        rows = [
            {"iterations": 0, "root_proved": True},
            {"iterations": 1, "root_proved": True},
            {"iterations": 1, "root_proved": True},
            {"iterations": 2, "root_proved": False},
            {"iterations": 3, "root_proved": True},
        ]
        self.assertEqual(
            _new_success_by_refinement_iteration(rows),
            [
                {"refinement_iterations": 0, "new_success_count": 1},
                {"refinement_iterations": 1, "new_success_count": 2},
                {"refinement_iterations": 2, "new_success_count": 0},
                {"refinement_iterations": 3, "new_success_count": 1},
            ],
        )
        self.assertEqual(
            _format_new_success_by_refinement_iteration(rows),
            "\n".join(
                [
                    "| result \\ refine_iterations | 0 | 1 | 2 | 3 |",
                    "| --- | --- | --- | --- | --- |",
                    "| new_success_count | 1 | 2 | 0 | 1 |",
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()

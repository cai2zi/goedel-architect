from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from robustpa_refine.run_robustpa_refine import (  # noqa: E402
    Record,
    _format_new_success_by_refinement_iteration,
    _load_or_initialize_phase2_checkpoint,
    _metric_row,
    _new_success_by_refinement_iteration,
    _record_paths,
    _should_skip_existing,
    _validate_args,
)
from checkpoint import CheckpointState, RunStatus  # noqa: E402


class RobustPAConfigTest(unittest.TestCase):
    def test_phase2_clones_pristine_seed_and_resume_checks_provenance(self) -> None:
        code = '''import Mathlib
import Architect
@[blueprint (statement := /-- final -/) (proof := /-- direct -/)]
theorem root : (1 + 1 : Nat) = 2 := by sorry_using []
'''
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed_root = root / "source" / "robustpa" / "blueprint"
            output_root = root / "output"
            record = Record(
                "subset__split__record", "record", "source/id", "subset", "split",
                root / "data.parquet", 1, "root", "problem", "cot", "2",
            )
            checkpoint_path, _trace, blueprint_dir = _record_paths(seed_root, record)
            state = CheckpointState(
                informal_statement="problem", informal_proof="cot", claimed_answer="2",
                model="model", status=RunStatus.SOLVED, semantic_fidelity_enabled=True,
                semantic_static_gate=False, semantic_status="strictAccepted",
                node_results={"root": {"signal": "PROVED", "proof_body": "by omega", "lean_errors": []}},
            )
            from blueprint import _parse_blueprint
            state.set_blueprint(_parse_blueprint(code, "root"))
            state.save(checkpoint_path)
            blueprint_dir.mkdir(parents=True, exist_ok=True)
            (blueprint_dir / "round_00_phase1.lean").write_text(code, encoding="utf-8")
            original = checkpoint_path.read_bytes()
            args = SimpleNamespace(
                phase1_seed_root=seed_root,
                phase1_source_experiment_root=root / "source",
            )
            output_checkpoint = _record_paths(output_root, record)[0]
            initialized, _blueprint, resumed, seed_hash, _bp_hash, _seed = (
                _load_or_initialize_phase2_checkpoint(args, record, output_checkpoint)
            )
            self.assertFalse(resumed)
            self.assertEqual(initialized.status, RunStatus.RUNNING)
            self.assertFalse(initialized.node_results)
            self.assertEqual(checkpoint_path.read_bytes(), original)
            resumed_state, _blueprint, resumed, same_hash, *_ = (
                _load_or_initialize_phase2_checkpoint(args, record, output_checkpoint)
            )
            self.assertTrue(resumed)
            self.assertEqual(seed_hash, same_hash)
            self.assertEqual(resumed_state.phase2_resume_count, 1)

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
        config.execution_mode = "phase2_only"
        config.resume = False
        config.max_refinement_iterations = 0
        config.phase1_seed_root = "/tmp/seed"
        config.phase1_source_results_path = "/tmp/seed/results.jsonl"
        config.phase1_source_experiment_root = "/tmp/source"
        _validate_args(SimpleNamespace(**OmegaConf.to_container(config, resolve=False)))
        config.execution_mode = "unknown"
        with self.assertRaisesRegex(ValueError, "execution_mode"):
            _validate_args(SimpleNamespace(**OmegaConf.to_container(config, resolve=False)))

    def test_resume_treats_terminal_phase1_results_and_errors_as_complete(self) -> None:
        args = SimpleNamespace(
            resume=True, retry_error_results=False, execution_mode="full",
        )
        self.assertTrue(_should_skip_existing({"status": "strictAccepted"}, args))
        self.assertTrue(_should_skip_existing({"status": "exhausted"}, args))
        self.assertTrue(_should_skip_existing({"status": "error"}, args))
        self.assertTrue(_should_skip_existing({"root_proved": True}, args))
        self.assertFalse(_should_skip_existing(None, args))

        args.retry_error_results = True
        self.assertFalse(_should_skip_existing({"status": "error"}, args))

    def test_phase2_resume_uses_only_phase2_terminal_results(self) -> None:
        args = SimpleNamespace(
            resume=True, retry_error_results=False, execution_mode="phase2_only",
        )
        self.assertFalse(_should_skip_existing({"status": "strictAccepted"}, args))
        self.assertTrue(_should_skip_existing({"status": "semanticRejected"}, args))
        self.assertTrue(_should_skip_existing({"status": "structuralRejected"}, args))
        self.assertTrue(_should_skip_existing({"status": "solved"}, args))
        self.assertTrue(_should_skip_existing({"status": "phase1Ineligible"}, args))
        args.retry_error_results = True
        self.assertFalse(_should_skip_existing({"status": "semanticRejected"}, args))

    def test_phase1_acceptance_reports_warning_partition(self) -> None:
        metrics = _metric_row("global", [
            {"status": "strictAccepted", "semantic_warning_codes": []},
            {"status": "acceptedWithWarnings",
             "semantic_warning_codes": ["nodeNotRootReachable"]},
            {"status": "error", "phase": "phase1",
             "semantic_warning_codes": ["stepNotRootReachable"]},
        ])
        self.assertEqual(metrics["blueprint_accepted"], 2)
        self.assertEqual(metrics["blueprint_accepted_with_warnings"], 1)
        self.assertEqual(metrics["blueprint_accepted_without_warnings"], 1)

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

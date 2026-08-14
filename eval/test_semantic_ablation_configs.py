from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]

from cot_blueprint_refine.common import load_config  # noqa: E402
from cot_blueprint_refine.run_semantic_ablation_suite import (  # noqa: E402
    BATCHES,
    SUITE_EXP_NAME,
    _run_batch,
    _select_suite_exp_name,
)


PROFILES = {
    "qwen3_8b_397b_wrong76_subtractive_separate_t06": ("separate", 0.6, 76),
    "qwen3_8b_397b_wrong76_subtractive_separate_t00": ("separate", 0.0, 76),
    "qwen3_8b_397b_wrong76_subtractive_joint_t06": ("joint", 0.6, 76),
    "qwen3_8b_397b_wrong76_subtractive_joint_t00": ("joint", 0.0, 76),
    "qwen3_8b_397b_all646_subtractive_separate_t06": ("separate", 0.6, 512),
}

GLOBAL_DEFINITION_PROFILES = {
    "qwen3_8b_397b_wrong76_global_defs_direct_named_t00": "direct",
    "qwen3_8b_397b_wrong76_global_defs_compact_separate_named_t00": "compact_separate",
}

MECHANICAL_CONTRACT_PROFILES = {
    "qwen3_8b_397b_wrong76_mechanical_contract_direct_named_t00": "direct",
    "qwen3_8b_397b_wrong76_mechanical_contract_compact_separate_named_t00":
        "compact_separate",
}


class SemanticAblationConfigTest(unittest.TestCase):
    def test_mechanical_contract_profiles_enable_bounded_retrieval(self) -> None:
        script_dir = ROOT / "experiments/cot_blueprint_refine/script"
        for profile, mode in MECHANICAL_CONTRACT_PROFILES.items():
            with self.subTest(profile=profile):
                config = load_config(profile, [])
                blueprint = config.blueprint
                self.assertEqual(str(config.exp_name), profile)
                self.assertFalse(config.resume)
                self.assertEqual(blueprint.semantic_audit_mode, mode)
                self.assertEqual(blueprint.generation_node_naming, "semantic")
                self.assertEqual(blueprint.execution_mode, "phase1_only")
                self.assertEqual(blueprint.phase1_concurrency, 76)
                self.assertTrue(blueprint.phase1_mathlib_search_enabled)
                self.assertEqual(blueprint.phase1_mathlib_search_max_queries_per_round, 2)
                self.assertEqual(blueprint.phase1_mathlib_search_k, 3)
                self.assertEqual(blueprint.phase1_mathlib_search_timeout_s, 15)
                self.assertFalse(config.judge.enabled)
                launcher = (script_dir / f"{profile}.sh").read_text()
                self.assertIn("run_semantic_ablation_experiment.sh", launcher)
                self.assertIn(profile, launcher)

    def test_global_definition_named_profiles(self) -> None:
        script_dir = ROOT / "experiments/cot_blueprint_refine/script"
        for profile, mode in GLOBAL_DEFINITION_PROFILES.items():
            with self.subTest(profile=profile):
                config = load_config(profile, [])
                blueprint = config.blueprint
                self.assertEqual(str(config.exp_name), profile)
                self.assertFalse(config.resume)
                self.assertEqual(blueprint.semantic_audit_mode, mode)
                self.assertEqual(blueprint.generation_node_naming, "semantic")
                self.assertEqual(blueprint.execution_mode, "phase1_only")
                self.assertEqual(blueprint.phase1_concurrency, 76)
                self.assertEqual(blueprint.generation_temperature, 0.6)
                self.assertEqual(blueprint.semantic_audit_temperature, 0.0)
                self.assertTrue(blueprint.semantic_audit_enable_thinking)
                self.assertFalse(config.judge.enabled)
                launcher = (script_dir / f"{profile}.sh").read_text()
                self.assertIn("run_semantic_ablation_experiment.sh", launcher)
                self.assertIn(profile, launcher)

    def test_five_resolved_configs_match_sampling_matrix(self) -> None:
        for profile, (mode, temperature, concurrency) in PROFILES.items():
            with self.subTest(profile=profile):
                config = load_config(profile, [])
                blueprint = config.blueprint
                self.assertEqual(str(config.exp_name), profile)
                self.assertFalse(config.resume)
                self.assertTrue(config.vllm.use_existing)
                self.assertTrue(config.kimina.use_existing)
                self.assertEqual(blueprint.semantic_audit_mode, mode)
                self.assertEqual(blueprint.semantic_audit_temperature, temperature)
                self.assertTrue(blueprint.semantic_audit_enable_thinking)
                self.assertEqual(blueprint.semantic_audit_top_p, 0.95)
                self.assertEqual(blueprint.semantic_audit_top_k, 20)
                self.assertEqual(blueprint.semantic_audit_min_p, 0.0)
                self.assertEqual(blueprint.semantic_audit_presence_penalty, 0.0)
                self.assertEqual(blueprint.semantic_audit_repetition_penalty, 1.0)
                self.assertEqual(blueprint.joint_semantic_audit_max_tokens, 32768)
                self.assertEqual(blueprint.semantic_format_max_attempts, 2)
                self.assertEqual(blueprint.generation_temperature, 0.6)
                self.assertTrue(blueprint.generation_enable_thinking)
                self.assertEqual(blueprint.generation_max_turns, 8)
                self.assertEqual(blueprint.phase1_concurrency, concurrency)
                self.assertEqual(blueprint.execution_mode, "phase1_only")
                self.assertEqual(blueprint.generation_prompt_profile, "whole_cot_minimal")

    def test_dataset_scope_and_suite_batches(self) -> None:
        for profile in PROFILES:
            config = load_config(profile, [])
            if "all646" in profile:
                self.assertIn("math_verify_eval", config.input_predictions)
            else:
                self.assertIn("original_answer_incorrect", config.input_predictions)
        self.assertEqual(tuple(map(len, BATCHES)), (2, 2, 1))
        self.assertIn("all646", BATCHES[-1][0])
        self.assertEqual(
            {profile for batch in BATCHES for profile in batch}, set(PROFILES),
        )

    def test_five_child_launchers_use_shared_two_slot_helper(self) -> None:
        script_dir = ROOT / "experiments/cot_blueprint_refine/script"
        helper = (script_dir / "run_semantic_ablation_experiment.sh").read_text()
        self.assertIn("slot-0.lock", helper)
        self.assertIn("slot-1.lock", helper)
        for profile in PROFILES:
            launcher = (script_dir / f"{profile}.sh").read_text()
            self.assertIn("run_semantic_ablation_experiment.sh", launcher)
            self.assertIn(profile, launcher)

    def test_suite_runtime_uses_new_attempt_after_interrupted_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            self.assertEqual(_select_suite_exp_name(output), SUITE_EXP_NAME)
            (output / SUITE_EXP_NAME).mkdir()
            attempt = _select_suite_exp_name(output)
            self.assertNotEqual(attempt, SUITE_EXP_NAME)
            self.assertTrue(attempt.startswith(SUITE_EXP_NAME + "_attempt_"))

    def test_suite_batch_passes_isolated_output_base(self) -> None:
        import inspect

        source = inspect.getsource(_run_batch)
        self.assertIn('f"output_base={output_base}"', source)


if __name__ == "__main__":
    unittest.main()

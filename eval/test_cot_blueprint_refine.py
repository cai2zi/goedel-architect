from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow.parquet as pq
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from blueprint import Blueprint, _parse_blueprint  # noqa: E402
from checkpoint import CheckpointState  # noqa: E402
from cot_blueprint_refine.common import (  # noqa: E402
    claimed_answer,
    extract_boxed_contents,
    extract_post_think,
)
from cot_blueprint_refine.evaluate import summarize_comparisons  # noqa: E402
from cot_blueprint_refine.export_blueprint_contexts import (  # noqa: E402
    export_contexts,
    prompt_signal,
    render_blueprint_context,
)
from cot_blueprint_refine.prepare_inputs import make_generation_row, prepare  # noqa: E402
from cot_blueprint_refine.run_cot_refinement import normalize_refined_output  # noqa: E402
from cot_blueprint_refine.run_experiment import ExperimentLock, blueprint_results_complete  # noqa: E402


class CotCleaningTest(unittest.TestCase):
    def test_extracts_only_suffix_after_balanced_think(self) -> None:
        post, reason = extract_post_think("<think>private</think> Public \\boxed{7}")
        self.assertEqual(reason, "")
        self.assertEqual(post, "Public \\boxed{7}")

    def test_rejects_unclosed_think(self) -> None:
        post, reason = extract_post_think("<think>unfinished")
        self.assertEqual(post, "")
        self.assertEqual(reason, "unclosed_think")

    def test_nested_box_and_last_box(self) -> None:
        text = r"First \boxed{\frac{1}{2}}, finally \boxed{3}"
        self.assertEqual(extract_boxed_contents(text), [r"\frac{1}{2}", "3"])
        self.assertEqual(claimed_answer(text), "3")

    def test_refined_output_contract(self) -> None:
        output, error, stripped, count = normalize_refined_output(
            "Step.\n\\boxed{4}", "stop"
        )
        self.assertEqual((error, stripped, count), ("", False, 1))
        self.assertEqual(output, "Step.\n\\boxed{4}")
        output, error, _stripped, count = normalize_refined_output(
            "\\boxed{4} and \\boxed{5}", "stop"
        )
        self.assertEqual(output, "")
        self.assertEqual(error, "conflicting_boxed_answers:found_2")
        self.assertEqual(count, 2)

    def test_refined_output_removes_placeholder_and_duplicate_boxes(self) -> None:
        output, error, stripped, count = normalize_refined_output(
            "Instruction `<think>` and \\boxed{...}. Work gives \\boxed{4}. Final \\boxed{4}",
            "stop",
        )
        self.assertEqual(error, "")
        self.assertFalse(stripped)
        self.assertEqual(count, 3)
        self.assertEqual(extract_boxed_contents(output), ["4"])

    def test_refined_output_ignores_analysis_outside_final_markers(self) -> None:
        raw = (
            r"Analysis mentions <think> and \boxed{wrong}. "
            r"<final_refined_solution>Steps. Final \boxed{2}</final_refined_solution>"
        )
        output, error, stripped, count = normalize_refined_output(raw, "stop")
        self.assertEqual(error, "")
        self.assertFalse(stripped)
        self.assertEqual(count, 1)
        self.assertEqual(output, r"Steps. Final \boxed{2}")


class PrepareInputsTest(unittest.TestCase):
    def test_filters_length_and_does_not_put_gold_in_generation_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "predictions.jsonl"
            rows = [
                {
                    "ID": "ok/1", "source": "demo", "row_index": 0,
                    "problem": "What is 1+1?", "gold": "2", "status": "ok",
                    "finish_reason": "stop",
                    "raw_cot": "<think>reason</think>Answer \\boxed{2}",
                },
                {
                    "ID": "length/open", "source": "demo", "row_index": 1,
                    "problem": "x", "gold": "x", "status": "ok",
                    "finish_reason": "length", "raw_cot": "<think>truncated",
                },
                {
                    "ID": "length/closed", "source": "demo", "row_index": 2,
                    "problem": "x", "gold": "x", "status": "ok",
                    "finish_reason": "length", "raw_cot": "<think>x</think>truncated",
                },
            ]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            config = OmegaConf.create({
                "input_predictions": str(input_path),
                "output_base": str(root / "outputs"),
                "exp_name": "unit",
                "include_ids": [],
            })
            stats = prepare(config)
            self.assertEqual(stats["unique_rows"], 3)
            self.assertEqual(stats["finish_reason_length"], 2)
            self.assertEqual(stats["length_unclosed_think"], 1)
            self.assertEqual(stats["length_balanced_think"], 1)
            self.assertEqual(stats["eligible_rows"], 1)
            parquet_path = next((root / "outputs/unit/prepared/data/qwen3_8b_math_verify").glob("*.parquet"))
            parquet_row = pq.read_table(parquet_path).to_pylist()[0]
            self.assertNotIn("gold", parquet_row)
            self.assertEqual(parquet_row["claimed_answer"], "2")

    def test_generation_statement_contains_claim_not_gold(self) -> None:
        row = {
            "ID": "id", "source": "s", "row_index": 1,
            "problem": "Compute x", "gold": "secret",
        }
        generation = make_generation_row(row, "Proof \\boxed{wrong}", "wrong")
        self.assertIn("Compute x", generation["informal_statement"])
        self.assertIn("\\boxed{wrong}", generation["informal_statement"])
        self.assertNotIn("gold", generation)
        self.assertNotIn("secret", json.dumps(generation))


class BlueprintContextTest(unittest.TestCase):
    @staticmethod
    def _blueprint() -> Blueprint:
        code = """import Mathlib
import Architect

@[blueprint (statement := /-- A checked step. -/) (proof := /-- Trivial. -/)]
lemma checked_step : True := by sorry_using []

@[blueprint (statement := /-- The target. -/) (proof := /-- Uses the step. -/)]
theorem target : True := by sorry_using [checked_step]
"""
        return _parse_blueprint(code, "target")

    def test_solved_proof_is_spliced_and_failed_node_is_annotated(self) -> None:
        blueprint = self._blueprint()
        blueprint.nodes[0].statement = "A checked step.\nWith a second line."
        blueprint.nodes[0].proof_sketch = "Trivial.\nUse True.intro."
        state = CheckpointState(informal_statement="test", model="model")
        state.set_blueprint(blueprint)
        state.proved_cache = {"checked_step": "by trivial"}
        state.node_results = {
            "checked_step": {"signal": "solved", "proof_body": "by trivial", "lean_errors": []},
            "target": {
                "signal": "proof_too_hard",
                "proof_body": "by exact checked_step",
                "lean_errors": ["example error"],
            },
        }
        context, nodes, infra = render_blueprint_context(blueprint, state)
        self.assertFalse(infra)
        self.assertIn("theorem checked_step : True := by trivial", context)
        self.assertIn("COT_BLUEPRINT_NODE_STATEMENT: A checked step.", context)
        self.assertIn("-- With a second line.", context)
        self.assertIn("COT_BLUEPRINT_NODE_PROOF_SKETCH: Trivial.", context)
        self.assertIn("-- Use True.intro.", context)
        self.assertIn("COT_BLUEPRINT_NODE_STATUS: NOT_PROVED", context)
        self.assertNotIn("COT_BLUEPRINT_NODE_NAME", context)
        self.assertNotIn("COT_BLUEPRINT_NODE_KIND", context)
        self.assertIn("theorem target : True := by sorry_using [checked_step]", context)
        self.assertEqual([node["prompt_signal"] for node in nodes], ["PROVED", "NOT_PROVED"])
        self.assertEqual(nodes[0]["statement"], "A checked step.\nWith a second line.")
        self.assertEqual(nodes[0]["proof_sketch"], "Trivial.\nUse True.intro.")

    def test_signal_mapping(self) -> None:
        self.assertEqual(prompt_signal("blocked_by_dependency"), "BLOCKED_BY_DEPENDENCY")
        self.assertEqual(prompt_signal("formally_negated"), "FORMALLY_NEGATED")
        self.assertEqual(prompt_signal("protocol_error"), "NOT_PROVED")
        self.assertEqual(prompt_signal("solved", proved=True), "PROVED")


class ExportContextsTest(unittest.TestCase):
    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    @staticmethod
    def _checkpoint(path: Path) -> None:
        blueprint = BlueprintContextTest._blueprint()
        state = CheckpointState(informal_statement="test", model="model", status="solved")
        state.set_blueprint(blueprint)
        state.proved_cache = {"checked_step": "by trivial", "target": "by trivial"}
        state.node_results = {
            "checked_step": {"signal": "solved", "proof_body": "by trivial", "lean_errors": []},
            "target": {"signal": "solved", "proof_body": "by trivial", "lean_errors": []},
        }
        state.save(path)

    def test_concurrent_export_preserves_result_order_and_counts(self) -> None:
        class FakeCompiler:
            def check(self, _lean_context: str, allow_sorry: bool = False):
                return SimpleNamespace(
                    success=True,
                    failure_kind=None,
                    diagnostics=[],
                )

            def close(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "out" / "unit"
            prepared = output_root / "prepared" / "generation_inputs.jsonl"
            robustpa = output_root / "robustpa" / "blueprint" / "results.jsonl"
            ckpt_b = root / "checkpoints" / "b.json"
            ckpt_a = root / "checkpoints" / "a.json"
            self._checkpoint(ckpt_b)
            self._checkpoint(ckpt_a)
            self._write_jsonl(prepared, [
                {
                    "name": "b", "source": "s", "problem": "problem b",
                    "claimed_answer": "2", "post_think_cot": "cot b",
                },
                {
                    "name": "a", "source": "s", "problem": "problem a",
                    "claimed_answer": "1", "post_think_cot": "cot a",
                },
            ])
            self._write_jsonl(robustpa, [
                {"source_id": "b", "status": "solved", "root_proved": True, "checkpoint_path": str(ckpt_b)},
                {"source_id": "a", "status": "solved", "root_proved": True, "checkpoint_path": str(ckpt_a)},
            ])
            config = OmegaConf.create({
                "output_base": str(root / "out"),
                "exp_name": "unit",
                "blueprint": {
                    "lean_api_url": "unused",
                    "lean_server_timeout": 1,
                    "lean_max_inflight_snippets": 4,
                    "lean_batch_size": 1,
                },
                "export": {"workers": 2},
            })
            with patch(
                "cot_blueprint_refine.export_blueprint_contexts._make_compiler",
                return_value=FakeCompiler(),
            ):
                metrics = export_contexts(config)
            output_path = Path(metrics["output"])
            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["ID"] for row in rows], ["b", "a"])
            self.assertEqual(metrics["counts"]["ready"], 2)
            self.assertEqual(metrics["counts"]["node_PROVED"], 4)
            self.assertIn("COT_BLUEPRINT_NODE_STATEMENT", rows[0]["lean_context"])
            self.assertIn("statement", rows[0]["nodes"][0])
            self.assertIn("proof_sketch", rows[0]["nodes"][0])


class EvaluationMetricsTest(unittest.TestCase):
    def test_full_denominators_and_transitions(self) -> None:
        comparisons = [
            {
                "source": "s", "before_correct": True, "after_correct": False,
                "transition": "correct_to_wrong", "blueprint_status": "ready",
                "refine_status": "ok", "node_status_counts": {"PROVED": 2},
            },
            {
                "source": "s", "before_correct": False, "after_correct": True,
                "transition": "wrong_to_correct", "blueprint_status": "ready",
                "refine_status": "ok", "node_status_counts": {"NOT_PROVED": 1},
            },
        ]
        metrics = summarize_comparisons(
            comparisons,
            dataset_total=4,
            global_eligible_total=2,
            global_before_correct=1,
            historical_raw_correct=2,
        )
        self.assertEqual(metrics["dataset"]["historical_raw_accuracy"], 0.5)
        self.assertEqual(metrics["dataset"]["strict_post_think_before_full_accuracy"], 0.25)
        self.assertEqual(metrics["full_after"]["eligible_accuracy"], 0.5)
        self.assertEqual(metrics["full_after"]["full_accuracy"], 0.25)
        self.assertEqual(metrics["selected"]["node_status_counts"], {"NOT_PROVED": 1, "PROVED": 2})


class ExperimentLockTest(unittest.TestCase):
    def test_rejects_concurrent_writer_for_same_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with ExperimentLock(root):
                with self.assertRaisesRegex(RuntimeError, "already locked"):
                    with ExperimentLock(root):
                        pass


class BlueprintResumeTest(unittest.TestCase):
    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_error_results_are_terminal_when_retry_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "out" / "unit"
            self._write_jsonl(output_root / "prepared" / "generation_inputs.jsonl", [
                {"name": "ok"},
                {"name": "err"},
            ])
            self._write_jsonl(output_root / "robustpa" / "blueprint" / "results.jsonl", [
                {"source_id": "ok", "status": "exhausted", "root_proved": False},
                {"source_id": "err", "status": "error", "root_proved": False},
            ])
            config = OmegaConf.create({
                "output_base": str(root / "out"),
                "exp_name": "unit",
                "resume": True,
                "blueprint": {"retry_error_results": False},
            })
            self.assertTrue(blueprint_results_complete(config))
            config.blueprint.retry_error_results = True
            self.assertFalse(blueprint_results_complete(config))

    def test_run_all_uses_stage_environment_variable(self) -> None:
        script = (REPO_ROOT / "experiments/cot_blueprint_refine/run_all.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('STAGE="${STAGE:-all}"', script)
        self.assertIn('--stage "${STAGE}"', script)


if __name__ == "__main__":
    unittest.main()

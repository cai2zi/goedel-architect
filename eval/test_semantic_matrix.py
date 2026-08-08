from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from cot_blueprint_refine.run_semantic_matrix import (
    FEATURE_TO_BLUEPRINT,
    MatrixRunLock,
    _jsonl_row_by_name,
    _write_jsonl_atomic,
    arm_overrides,
    begin_suite_manifest,
    prepare_arm_manifest,
    selection_sha256,
    select_arms,
    validate_blueprint_results,
    validate_cot_to_blueprint_results,
    validate_evaluation_results,
    validate_refine_results,
)
from cot_blueprint_refine.semantic_quality_report import build_matrix_report


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class SemanticMatrixTest(unittest.TestCase):
    def config(self):
        return OmegaConf.create({
            "output_base": "/tmp/out",
            "matrix": {
                "output_prefix": "matrix",
                "add_arms": ["E0", "E1"],
                "reduce_arms": ["R1"],
                "key_refine_arms": ["E0"],
                "arms": {
                    name: {
                        "label": name,
                        "parent": "legacy_E0" if name == "E0" else "E0",
                        "features": {
                            "fidelity_enabled": name != "E0",
                            "require_step_ids": name != "E0",
                            "static_gate": False,
                            "minimal_ir": False,
                            "freeze_refinement": False,
                            "audit_mode": "none",
                            "max_repair_attempts": 0,
                            "proof_policy": "full",
                            "critical_negation_max_turns": 0,
                        },
                    }
                    for name in ("E0", "E1", "R1")
                },
            },
        })

    def test_select_arms_and_flat_feature_mapping(self):
        config = self.config()
        self.assertEqual(select_arms(config, "add"), ["E0", "E1"])
        self.assertEqual(select_arms(config, "all"), ["E0", "E1", "R1"])
        overrides = arm_overrides(config, "E1", run_id="run1")
        self.assertIn("exp_name=matrix/run1/E1", overrides)
        self.assertIn("blueprint.semantic_fidelity_enabled=true", overrides)
        self.assertIn("blueprint.semantic_require_step_ids=true", overrides)
        self.assertIn("blueprint.proof_policy=full", overrides)

    def test_wrong76_reduction_arms_are_single_step_effective_deltas(self):
        profile_path = (
            Path(__file__).resolve().parents[1]
            / "experiments/cot_blueprint_refine/configs"
            / "qwen3_8b_397b_wrong76_semantic_matrix.yaml"
        )
        profile = OmegaConf.load(profile_path)

        def effective_blueprint(arm_name: str) -> dict:
            effective = OmegaConf.to_container(profile.blueprint, resolve=False)
            self.assertIsInstance(effective, dict)
            effective = dict(effective)
            arm = profile.matrix.arms[arm_name]
            for feature_name, blueprint_name in FEATURE_TO_BLUEPRINT.items():
                effective[blueprint_name] = arm.features[feature_name]
            if arm.get("overrides") and arm.overrides.get("blueprint"):
                effective.update(OmegaConf.to_container(
                    arm.overrides.blueprint, resolve=False,
                ))
            return effective

        def changed(parent: str, child: str) -> set[str]:
            before = effective_blueprint(parent)
            after = effective_blueprint(child)
            return {
                key for key in set(before) | set(after)
                if before.get(key) != after.get(key)
            }

        self.assertEqual(changed("E6", "R1"), {"semantic_audit_mode"})
        self.assertEqual(changed("R1", "R2"), {"semantic_audit_mode"})
        self.assertEqual(
            changed("R2", "R3"), {"semantic_max_repair_attempts"},
        )
        self.assertEqual(changed("R3", "R4"), {"semantic_minimal_ir"})
        self.assertEqual(changed("R4", "R5"), {"max_refinement_iterations"})
        self.assertEqual(changed("R5", "R6"), {
            "proof_policy",
            "generation_max_tokens",
            "blueprint_max_retries",
            "node_max_prove_turns",
            "max_tool_calls_per_turn",
        })

    def test_audit_replay_jsonl_helpers_are_strict_and_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "rows.jsonl"
            _write_jsonl_atomic(path, [{"name": "A", "value": 1}, {"name": "B"}])
            self.assertEqual(_jsonl_row_by_name(path, "A")["value"], 1)
            with self.assertRaisesRegex(ValueError, "found 0"):
                _jsonl_row_by_name(path, "missing")
            _write_jsonl_atomic(path, [{"name": "A"}, {"name": "A"}])
            with self.assertRaisesRegex(ValueError, "found 2"):
                _jsonl_row_by_name(path, "A")

    def test_manifest_refuses_changed_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "E0"
            kwargs = {
                "arm_root": root,
                "arm": "E0",
                "run_id": "run1",
                "input_path": Path("/input.jsonl"),
                "input_sha256": "input",
                "code_sha256": "code",
                "config_sha256": "config",
                "expected_ids": {"A", "B"},
                "label": "baseline",
                "parent": "legacy_E0",
                "dry_run": False,
            }
            prepare_arm_manifest(**kwargs)
            prepare_arm_manifest(**kwargs)
            with self.assertRaisesRegex(RuntimeError, "unsafe resume refused"):
                prepare_arm_manifest(**{**kwargs, "code_sha256": "changed"})
            with self.assertRaisesRegex(RuntimeError, "unsafe resume refused"):
                prepare_arm_manifest(**{**kwargs, "expected_ids": {"C", "D"}})
            manifest = json.loads((root / "semantic_run_manifest.json").read_text())
            self.assertEqual(manifest["selected_ids"], ["A", "B"])
            self.assertEqual(manifest["selection_sha256"], selection_sha256({"A", "B"}))

    def test_record_validation_accepts_terminal_quality_and_semantic_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "robustpa/blueprint/results.jsonl", [
                {"source_id": "A", "status": "solved", "infra_error_node_count": 0},
                {
                    "source_id": "B",
                    "status": "error",
                    "phase": "phase1",
                    "failed_blueprint_failure_stage": "semantic_static_gate",
                    "infra_error_node_count": 0,
                },
                {
                    "source_id": "C",
                    "status": "error",
                    "phase": "phase1",
                    "error": "Lean syntax remained invalid after the model retry budget",
                    "infra_error_node_count": 0,
                },
                {
                    "source_id": "D",
                    "status": "error",
                    "phase": "phase2",
                    "error": "",
                    "semantic_status": "phase2_checkpoint_rejected",
                    "infra_error_node_count": 0,
                },
            ])
            validation = validate_blueprint_results(root, {"A", "B", "C", "D"})
            self.assertTrue(validation["passed"])
            self.assertTrue(validation["quality_warning"])
            self.assertEqual(validation["semantic_rejection_ids"], ["B", "D"])
            self.assertEqual(validation["quality_warning_ids"], ["C"])
            self.assertEqual(validation["blocking_reasons"], {})

    def test_record_validation_blocks_missing_nonterminal_infra_and_service_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "robustpa/blueprint/results.jsonl", [
                {"source_id": "A", "status": "solved"},
                {"source_id": "B", "status": "running"},
                {"source_id": "C", "status": "error", "infra_error_node_count": 1},
                {
                    "source_id": "D",
                    "status": "error",
                    "traceback": "openai.APIConnectionError: server disconnected",
                },
                {"source_id": "F", "status": "error", "phase": "phase2", "error": ""},
                {"source_id": "UNEXPECTED", "status": "solved"},
            ])
            validation = validate_blueprint_results(
                root, {"A", "B", "C", "D", "E", "F"},
            )
            self.assertFalse(validation["passed"])
            self.assertEqual(validation["blocking_reasons"]["missing_ids"], ["E"])
            self.assertEqual(
                validation["blocking_reasons"]["unexpected_ids"], ["UNEXPECTED"],
            )
            self.assertEqual(validation["blocking_reasons"]["nonterminal_ids"], ["B"])
            self.assertEqual(
                validation["blocking_reasons"]["infra_error_ids"], ["C", "D", "F"],
            )

    def test_stage_validators_require_exact_ids_and_noninfra_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "robustpa/blueprint/results.jsonl", [
                {"source_id": "A", "status": "solved"},
                {"source_id": "B", "status": "exhausted"},
            ])
            write_jsonl(root / "prepared/generation_inputs.jsonl", [
                {"name": "A"}, {"name": "B"},
            ])
            context_path = root / "blueprint_contexts/blueprint_contexts.jsonl"
            write_jsonl(context_path, [
                {"ID": "A", "context_quality": "VERIFIED"},
                {"ID": "B", "context_quality": "INVALID_BLUEPRINT_CANDIDATE"},
            ])
            validation = validate_cot_to_blueprint_results(root, {"A", "B"})
            self.assertTrue(validation["passed"])
            self.assertTrue(validation["quality_warning"])

            write_jsonl(context_path, [
                {"ID": "A", "context_quality": "VERIFIED"},
                {"ID": "B", "context_quality": "INFRA_ERROR"},
            ])
            validation = validate_cot_to_blueprint_results(root, {"A", "B"})
            self.assertFalse(validation["passed"])
            self.assertEqual(
                validation["blocking_reasons"]["export"]["infra_context_ids"], ["B"],
            )

    def test_refine_and_evaluation_validators_require_complete_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conversations = root / "conversations"
            conversations.mkdir()
            for row_id in ("A", "B"):
                (conversations / f"{row_id}.json").write_text("{}", encoding="utf-8")
            refined_path = root / "refinement/blueprint/refined_predictions.jsonl"
            write_jsonl(refined_path, [
                {"ID": row_id, "status": "ok", "conversation_path": str(conversations / f"{row_id}.json")}
                for row_id in ("A", "B")
            ])
            write_json(root / "refinement/blueprint/refinement_metrics.json", {"rows": 2})
            self.assertTrue(
                validate_refine_results(root, {"A", "B"}, ["blueprint"])["passed"]
            )
            write_jsonl(refined_path, [{
                "ID": "A", "status": "invalid_output",
                "conversation_path": str(conversations / "A.json"),
            }])
            self.assertFalse(
                validate_refine_results(root, {"A", "B"}, ["blueprint"])["passed"]
            )

            comparison_path = root / "evaluation/blueprint/comparison.jsonl"
            write_jsonl(comparison_path, [
                {"ID": row_id, "refine_status": "ok"} for row_id in ("A", "B")
            ])
            write_json(root / "evaluation/blueprint/metrics.json", {
                "efficiency": {"rows": 2},
            })
            parquet = root / "evaluation/analysis.parquet"
            parquet.parent.mkdir(parents=True, exist_ok=True)
            parquet.write_bytes(b"test")
            write_json(root / "evaluation/metrics.json", {
                "variants": {"blueprint": {}},
                "analysis_artifacts": {"parquet": str(parquet), "parquet_rows": 2},
            })
            self.assertTrue(validate_evaluation_results(
                root, {"A", "B"}, ["blueprint"], judge_enabled=False,
            )["passed"])
            write_jsonl(comparison_path, [{"ID": "A", "refine_status": "ok"}])
            self.assertFalse(validate_evaluation_results(
                root, {"A", "B"}, ["blueprint"], judge_enabled=False,
            )["passed"])

    def test_suite_attempt_archives_interruption_and_clears_current_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "matrix_manifest.json"
            identity = {
                "fingerprint_schema": 2,
                "run_id": "r1",
                "input_sha256": "input",
                "code_sha256": "code",
                "expected_result_count": 2,
                "selected_ids": ["A", "B"],
                "selection_sha256": selection_sha256({"A", "B"}),
            }
            write_json(path, {
                **identity,
                "arms": ["E0"],
                "failures": [{"arm": "_suite", "error": "old"}],
                "finished_at": "old",
                "executions": [{"started_at": "old"}],
            })
            suite, execution_id, known = begin_suite_manifest(
                path=path,
                identity=identity,
                requested_arms=["E1"],
                refine_policy="none",
                input_path=Path("/input.jsonl"),
                code_paths=[],
                legacy_root=Path("/legacy"),
                available_arms=["E0", "E1"],
            )
            self.assertEqual(known, ["E0", "E1"])
            self.assertEqual(suite["failures"], [])
            self.assertNotIn("finished_at", suite)
            self.assertEqual(suite["executions"][0]["status"], "interrupted_before_resume")
            self.assertEqual(suite["executions"][-1]["execution_id"], execution_id)
            with MatrixRunLock(root):
                with self.assertRaisesRegex(RuntimeError, "already locked"):
                    with MatrixRunLock(root):
                        pass

    def test_report_handles_legacy_without_semantic_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            legacy = base / "legacy"
            matrix = base / "matrix"
            checkpoint = legacy / "checkpoint.json"
            trace = legacy / "trace.jsonl"
            write_json(checkpoint, {
                "blueprint_lean_file": (
                    "@[blueprint (statement := /-- COT_STEP_ID: S001 -/)]\n"
                    "theorem root : True := by trivial\n"
                ),
                "blueprint_target": "root",
                "node_results": {"root": {"signal": "solved"}},
            })
            write_jsonl(trace, [{
                "kind": "llm_request_end",
                "duration_ms": 10,
                "args": {
                    "operation": "blueprint_generate",
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                    "finish_reason": "stop",
                    "retry_index": 0,
                },
            }])
            write_jsonl(legacy / "robustpa/blueprint/results.jsonl", [{
                "source_id": "A",
                "status": "solved",
                "root_proved": True,
                "iterations": 0,
                "checkpoint_path": str(checkpoint),
                "trace_path": str(trace),
                "infra_error_node_count": 0,
            }])
            write_jsonl(legacy / "prepared/generation_inputs.jsonl", [{
                "name": "A",
                "claimed_answer": "7",
                "cot_manifest_json": json.dumps([{
                    "step_id": "S001", "source_text": "Therefore 7.",
                }]),
            }])
            subset = base / "subset.json"
            write_json(subset, {"selected_ids": ["A", "B"]})
            write_json(matrix / "E0/semantic_run_manifest.json", {
                "parent": "legacy_E0",
                "blueprint_validation": {"passed": False},
            })

            report, rows, pairwise = build_matrix_report(
                matrix_root=matrix,
                arms=["E0"],
                legacy_root=legacy,
                subset_metrics_path=subset,
                selected_ids={"A"},
            )
            self.assertEqual(report["selected_rows"], 1)
            self.assertEqual(report["subset_rows"], 2)
            summary = report["arms"]["legacy_E0"]
            self.assertEqual(summary["rows"], 1)
            self.assertEqual(summary["cost"]["total_tokens"], 30)
            self.assertEqual(summary["mapping"]["step_coverage"], 1.0)
            self.assertEqual(summary["obvious_violations_heuristic"]["root_true"], 1)
            self.assertFalse(rows[0]["semantic_metadata_available"])
            self.assertEqual(
                report["arms"]["E0"]["quality_gates"]["record_completion_validation"],
                "FAIL",
            )
            self.assertEqual(pairwise, [])


if __name__ == "__main__":
    unittest.main()

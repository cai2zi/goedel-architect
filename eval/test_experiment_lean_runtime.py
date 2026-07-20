from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from kimina_lean_compiler import KiminaLeanCompiler  # noqa: E402
from blueprint import _parse_blueprint  # noqa: E402
from checkpoint import CheckpointState  # noqa: E402
from lean_compiler import CompilerResult, LeanCompiler  # noqa: E402
from miniF2F_onepass.run_minif2f_onepass import _is_completed_result  # noqa: E402
from shared.lean_runtime import (  # noqa: E402
    add_lean_runtime_args,
    make_lean_runtime,
    prepare_lean_runtime_metadata,
)
from shared.onepass import run_onepass_phase1, run_onepass_record  # noqa: E402
from shared.phase0 import Phase0Result, _check_theorem  # noqa: E402
from tts_rerank_math_verify.run_tts_rerank import (  # noqa: E402
    _is_terminal_score,
    _make_phase0_row,
)


def _parse_runtime_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_lean_runtime_args(parser)
    return parser.parse_args(argv)


THEOREM_STMT = "theorem root : True := by"
BLUEPRINT_LEAN = """import Mathlib
import Architect

@[blueprint
  (statement := /-- The root theorem. -/)
  (proof := /-- Trivial. -/)]
theorem root : True := by
  sorry_using []
"""


def _write_blueprint_checkpoint(
    output_root: Path,
    *,
    validated: bool,
    solved: bool = False,
    theorem_stmt: str = THEOREM_STMT,
    blueprint_lean: str = BLUEPRINT_LEAN,
) -> None:
    blueprint = _parse_blueprint(blueprint_lean, "root")
    blueprint.fully_validated = validated
    state = CheckpointState(theorem_stmt=theorem_stmt, model="fake")
    state.set_blueprint(blueprint)
    if solved:
        state.proved_cache = {"root": "by trivial"}
        state.done = True
        state.success = True
    state.save(output_root / "checkpoints" / "test.json")


class ExperimentLeanRuntimeTest(unittest.TestCase):
    def test_defaults_and_environment_cli_priority(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            defaults = _parse_runtime_args([])
        self.assertEqual(defaults.lean_backend, "kimina_server")
        self.assertEqual(defaults.lean_api_url, "http://localhost:8000")
        self.assertEqual(defaults.lean_server_timeout, 300)
        self.assertTrue(defaults.lean_server_reuse)
        self.assertFalse(defaults.lean_server_debug)
        self.assertEqual(defaults.lean_check_concurrency, 8)

        with patch.dict(os.environ, {"KIMINA_API_URL": "http://env-host:9000"}, clear=True):
            from_env = _parse_runtime_args([])
            from_cli = _parse_runtime_args(["--lean-api-url", "http://cli-host:7000"])
        self.assertEqual(from_env.lean_api_url, "http://env-host:9000")
        self.assertEqual(from_cli.lean_api_url, "http://cli-host:7000")

    def test_kimina_and_explicit_local_factory_behavior(self) -> None:
        kimina = make_lean_runtime(_parse_runtime_args([]))
        local = make_lean_runtime(_parse_runtime_args(["--lean-backend", "local"]))
        try:
            self.assertIsInstance(kimina.compiler, KiminaLeanCompiler)
            self.assertIsNone(kimina.compiler_factory)
            self.assertNotIn("api_key", kimina.metadata)
            self.assertNotIn("api_key_env", kimina.metadata)
            self.assertIsInstance(local.compiler, LeanCompiler)
            self.assertIs(local.compiler_factory, LeanCompiler)
        finally:
            kimina.close()
            local.close()

    def test_metadata_write_resume_match_mismatch_and_missing(self) -> None:
        metadata = {
            "backend": "kimina_server",
            "api_url": "http://localhost:8000",
            "timeout_s": 300,
            "reuse": True,
            "debug": False,
            "check_concurrency": 8,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = prepare_lean_runtime_metadata(root, resume=False, metadata=metadata)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), metadata)
            prepare_lean_runtime_metadata(root, resume=True, metadata=metadata)

            different = {**metadata, "check_concurrency": 4}
            with self.assertRaisesRegex(RuntimeError, "different Lean runtime metadata"):
                prepare_lean_runtime_metadata(root, resume=True, metadata=different)

            with self.assertRaisesRegex(RuntimeError, "is missing"):
                prepare_lean_runtime_metadata(root / "old-output", resume=True, metadata=metadata)

    def test_tts_phase0_phase1_and_phase2_share_one_compiler(self) -> None:
        compiler = LeanCompiler()
        metadata = {"backend": "kimina_server"}
        phase0_result = Phase0Result(
            theorem_stmt="theorem root : True := by",
            success=True,
            error="",
            attempts=1,
        )
        blueprint = SimpleNamespace(
            nodes=[object()],
            target_theorem="root",
            lean_file="theorem root : True := by trivial\n",
        )
        orch_result = SimpleNamespace(proved={"root"}, failed=set())
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "tts_rerank_math_verify.run_tts_rerank.formalize_candidate",
                return_value=phase0_result,
            ) as formalize:
                _make_phase0_row(
                    problem={"id": "parent", "question": "q"},
                    rollout={"rollout_id": 1, "canonical_extracted_answer": "a"},
                    rollout_index=1,
                    model="fake",
                    max_attempts=1,
                    compiler=compiler,
                    lean_runtime=metadata,
                )
            with patch("shared.onepass.run_phase1", return_value=blueprint) as phase1:
                with patch("shared.onepass.run_phase2", return_value=orch_result) as phase2:
                    result = run_onepass_record(
                        record_id="test",
                        theorem_stmt="theorem root : True := by",
                        nl_proof="trivial",
                        model="fake",
                        output_root=Path(tmp),
                        compiler=compiler,
                        compiler_factory=None,
                    )

        self.assertTrue(result["root_proved"])
        self.assertIs(formalize.call_args.kwargs["compiler"], compiler)
        self.assertIs(phase1.call_args.kwargs["compiler"], compiler)
        self.assertIs(phase2.call_args.kwargs["compiler"], compiler)
        self.assertIsNone(phase2.call_args.kwargs["compiler_factory"])

    def test_phase0_uses_injected_compiler(self) -> None:
        class RecordingCompiler(LeanCompiler):
            def __init__(self) -> None:
                self.codes: list[str] = []

            def _run_lean(self, code: str) -> CompilerResult:
                self.codes.append(code)
                return CompilerResult(success=True)

        compiler = RecordingCompiler()
        ok, error = _check_theorem("theorem injected : True := by", compiler)

        self.assertTrue(ok)
        self.assertEqual(error, "")
        self.assertEqual(len(compiler.codes), 1)
        self.assertIn("theorem injected", compiler.codes[0])

    def test_resume_reuses_validated_blueprint_and_continues_phase2(self) -> None:
        orch_result = SimpleNamespace(proved={"root"}, failed=set())
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            _write_blueprint_checkpoint(output_root, validated=True)
            with patch("shared.onepass.run_phase1") as phase1:
                with patch("shared.onepass.run_phase2", return_value=orch_result) as phase2:
                    result = run_onepass_record(
                        record_id="test",
                        theorem_stmt=THEOREM_STMT,
                        nl_proof="trivial",
                        model="fake",
                        output_root=output_root,
                        resume=True,
                        compiler=LeanCompiler(),
                        compiler_factory=None,
                    )

            self.assertTrue((output_root / "blueprints" / "test.lean").exists())
            trace = (output_root / "traces" / "test.jsonl").read_text(encoding="utf-8")

        phase1.assert_not_called()
        phase2.assert_called_once()
        self.assertTrue(result["blueprint_reused"])
        self.assertTrue(result["phase1_skipped"])
        self.assertTrue(result["root_proved"])
        self.assertIn('"kind": "resume"', trace)

    def test_resume_regenerates_unvalidated_blueprint(self) -> None:
        blueprint = _parse_blueprint(BLUEPRINT_LEAN, "root")
        blueprint.fully_validated = True
        orch_result = SimpleNamespace(proved={"root"}, failed=set())
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            _write_blueprint_checkpoint(output_root, validated=False)
            with patch("shared.onepass.run_phase1", return_value=blueprint) as phase1:
                with patch("shared.onepass.run_phase2", return_value=orch_result) as phase2:
                    result = run_onepass_record(
                        record_id="test",
                        theorem_stmt=THEOREM_STMT,
                        nl_proof="trivial",
                        model="fake",
                        output_root=output_root,
                        resume=True,
                        compiler=LeanCompiler(),
                        compiler_factory=None,
                    )

        phase1.assert_called_once()
        phase2.assert_called_once()
        self.assertFalse(result["blueprint_reused"])
        self.assertFalse(result["phase1_skipped"])

    def test_resume_regenerates_blueprint_missing_phase2_placeholder(self) -> None:
        invalid_blueprint = """import Mathlib
import Architect

@[blueprint
  (statement := /-- The root theorem. -/)
  (proof := /-- Trivial. -/)]
theorem root : True := by
  trivial
"""
        replacement = _parse_blueprint(BLUEPRINT_LEAN, "root")
        replacement.fully_validated = True
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            _write_blueprint_checkpoint(
                output_root,
                validated=True,
                blueprint_lean=invalid_blueprint,
            )
            with patch("shared.onepass.run_phase1", return_value=replacement) as phase1:
                result = run_onepass_phase1(
                    record_id="test",
                    theorem_stmt=THEOREM_STMT,
                    nl_proof="trivial",
                    model="fake",
                    output_root=output_root,
                    resume=True,
                    compiler=LeanCompiler(),
                )

        phase1.assert_called_once()
        self.assertEqual(result["status"], "ready")
        self.assertFalse(result["blueprint_reused"])
        self.assertFalse(result["phase1_skipped"])
        self.assertEqual(
            result["resume_blueprint_invalid_categories"],
            {"missing_sorry_using_placeholder": 1},
        )

    def test_resume_skips_fully_solved_checkpoint_and_scores_proved_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            _write_blueprint_checkpoint(output_root, validated=True, solved=True)
            with patch("shared.onepass.run_phase1") as phase1:
                with patch("shared.onepass.run_phase2") as phase2:
                    result = run_onepass_record(
                        record_id="test",
                        theorem_stmt=THEOREM_STMT,
                        nl_proof="trivial",
                        model="fake",
                        output_root=output_root,
                        resume=True,
                    )

        phase1.assert_not_called()
        phase2.assert_not_called()
        self.assertTrue(result["root_proved"])
        self.assertEqual(result["proved_nodes"], ["root"])

    def test_resume_rejects_checkpoint_for_different_theorem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            _write_blueprint_checkpoint(
                output_root,
                validated=True,
                theorem_stmt="theorem another : True := by",
            )
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                run_onepass_record(
                    record_id="test",
                    theorem_stmt=THEOREM_STMT,
                    nl_proof="trivial",
                    model="fake",
                    output_root=output_root,
                    resume=True,
                )

    def test_runner_resume_only_skips_terminal_rows(self) -> None:
        self.assertTrue(_is_completed_result({"root_proved": True}))
        self.assertFalse(_is_completed_result({"root_proved": False}))
        self.assertFalse(_is_terminal_score({"phase0_success": True, "root_proved": False}))
        self.assertTrue(_is_terminal_score({"phase0_success": True, "root_proved": True}))
        self.assertTrue(_is_terminal_score({"phase0_success": False, "root_proved": False}))


if __name__ == "__main__":
    unittest.main()

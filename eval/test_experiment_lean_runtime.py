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
from lean_compiler import CompilerResult, LeanCompiler  # noqa: E402
from shared.lean_runtime import (  # noqa: E402
    add_lean_runtime_args,
    make_lean_runtime,
    prepare_lean_runtime_metadata,
)
from shared.onepass import run_onepass_record  # noqa: E402
from shared.phase0 import Phase0Result, _check_theorem  # noqa: E402
from tts_rerank_math_verify.run_tts_rerank import _make_phase0_row  # noqa: E402


def _parse_runtime_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_lean_runtime_args(parser)
    return parser.parse_args(argv)


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


if __name__ == "__main__":
    unittest.main()

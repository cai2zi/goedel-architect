from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kimina_lean_compiler import CompileRequest, KiminaLeanCompiler  # noqa: E402


@unittest.skipUnless(
    os.environ.get("GOEDEL_RUN_LIVE_KIMINA") == "1",
    "set GOEDEL_RUN_LIVE_KIMINA=1 for the live server integration suite",
)
class KiminaLiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = KiminaLeanCompiler(
            api_url=os.environ.get("KIMINA_API_URL", "http://127.0.0.1:8000"),
            timeout_s=600,
            max_inflight_snippets=8,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.compiler.close()

    def test_complete_file_success(self) -> None:
        result = self.compiler.check(
            "import Mathlib\nexample : 1 + 1 = 2 := by norm_num\n",
        )
        self.assertTrue(result.success, result.errors)

    def test_error_has_source_position(self) -> None:
        result = self.compiler.check(
            "import Mathlib\nexample : True := by\n  trivial\n  trivial\n",
        )
        self.assertFalse(result.success)
        diagnostic = json.loads(result.errors[0])
        self.assertEqual(diagnostic["data"], "No goals to be solved")
        self.assertIn("pos", diagnostic)
        self.assertIn("endPos", diagnostic)

    def test_sorry_is_rejected(self) -> None:
        result = self.compiler.check(
            "import Mathlib\ntheorem live_sorry : True := by sorry\n",
        )
        self.assertFalse(result.success)
        self.assertTrue(any("sorry" in error for error in result.errors))

    def test_blueprint_validation(self) -> None:
        code = """import Mathlib
import Architect

@[blueprint (statement := /-- Live root. -/) (proof := /-- Trivial. -/)]
theorem live_blueprint_root : True := by sorry_using []
"""
        result = self.compiler.check_blueprint(code, "live_blueprint_root")
        self.assertTrue(result.success, result.errors)

    def test_node_assembly(self) -> None:
        result = self.compiler.check_node(
            "by exact parent",
            node_decl="theorem live_node : True := by sorry_using [parent]",
            parent_lemma_decls="theorem parent : True := by trivial",
            header="import Mathlib\nimport Architect",
        )
        self.assertTrue(result.success, result.errors)

    def test_batch(self) -> None:
        results = self.compiler.check_many([
            CompileRequest(
                "import Mathlib\nexample : 2 + 2 = 4 := by norm_num\n",
                request_id="live-a",
            ),
            CompileRequest(
                "import Mathlib\nexample : (3 : Nat) < 5 := by norm_num\n",
                request_id="live-b",
            ),
            CompileRequest(
                "import Mathlib\nexample : True := by trivial\n",
                request_id="live-c",
            ),
        ])
        self.assertEqual([result.success for result in results], [True, True, True])


if __name__ == "__main__":
    unittest.main()

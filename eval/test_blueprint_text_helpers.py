"""Regression tests for blueprint attribute and signature text handling."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from blueprint import (  # noqa: E402
    _parse_blueprint,
    extract_blueprint_signature,
    strip_blueprint_attr,
)
from lean_compiler import _assemble_node_attempt, _extract_current_node_decl  # noqa: E402


ZMOD_NODE = """@[blueprint
  (statement := /-- 2003 is congruent to 1 modulo 7. -/)
  (proof := /-- Compute 2003 = 1 [ZMOD 7]. -/)]
lemma mod7_2003_eq_1 :
    (2003 : Int) = (1 : Int) [ZMOD (7 : Int)] := by
  sorry_using []
"""


class BlueprintTextHelpersTest(unittest.TestCase):
    def test_signature_ignores_bracket_inside_blueprint_doc_comment(self) -> None:
        expected = (
            "theorem mod7_2003_eq_1 :\n"
            "    (2003 : Int) = (1 : Int) [ZMOD (7 : Int)]"
        )
        self.assertEqual(extract_blueprint_signature(ZMOD_NODE), expected)

        blueprint = _parse_blueprint(ZMOD_NODE, "mod7_2003_eq_1")
        self.assertEqual(len(blueprint.nodes), 1)
        node = blueprint.nodes[0]
        self.assertEqual(node.signature(), expected)

    def test_strip_preserves_full_multi_node_file(self) -> None:
        lean_code = f"""import Mathlib
import Architect

{ZMOD_NODE}
@[blueprint
  (statement := /-- Root statement with [ZMOD 7]. -/)
  (proof := /-- Apply `mod7_2003_eq_1`. -/)]
theorem root : True := by
  sorry_using [mod7_2003_eq_1]
"""
        stripped = strip_blueprint_attr(lean_code)

        self.assertNotIn("@[blueprint", stripped)
        self.assertIn("import Mathlib", stripped)
        self.assertIn("lemma mod7_2003_eq_1", stripped)
        self.assertIn("theorem root", stripped)
        self.assertNotIn(". -/)]", stripped)

    def test_parser_and_signature_share_noncomputable_def_support(self) -> None:
        lean_code = """@[blueprint
  (statement := /-- A noncomputable helper. -/)]
noncomputable def helper : Nat := 1

@[blueprint
  (statement := /-- The target. -/)
  (proof := /-- Use `helper`. -/)]
theorem root : helper = 1 := by
  sorry_using [helper]
"""
        blueprint = _parse_blueprint(lean_code, "root")

        self.assertEqual([node.name for node in blueprint.nodes], ["helper", "root"])
        self.assertEqual(blueprint.nodes[0].kind, "definition")
        self.assertEqual(blueprint.nodes[1].dependencies, ["helper"])

    def test_definition_fallback_uses_outer_assignment(self) -> None:
        declaration = """@[blueprint
  (statement := /-- Configuration value. -/)]
def config : Nat × Nat := { fst := 1, snd := 2 }
"""
        self.assertEqual(extract_blueprint_signature(declaration), "def config : Nat × Nat")

    def test_node_decl_ignores_declaration_words_inside_blueprint_comments(self) -> None:
        declaration = """@[blueprint
  (statement := /-- The target theorem. -/)
  (proof := /-- The hypotheses are part of the original theorem statement. -/)]
theorem mathd_algebra_478 : True := by
  sorry_using []
"""
        expected = "theorem mathd_algebra_478 : True := by\n  sorry_using []"

        self.assertEqual(_extract_current_node_decl(declaration), expected)

        assembled = _assemble_node_attempt(declaration, "", "by trivial")
        self.assertNotIn("@[blueprint", assembled)
        self.assertNotIn("theorem statement", assembled)
        self.assertIn("theorem mathd_algebra_478 : True := by trivial", assembled)


if __name__ == "__main__":
    unittest.main()

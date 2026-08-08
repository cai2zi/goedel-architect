from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]

from blueprint import _parse_blueprint, _render_step_grounded_proof  # noqa: E402
from cot_blueprint_refine.formal_steps import (  # noqa: E402
    encode_formal_step_manifest,
    make_formal_step_manifest,
)
from semantic_audit import parse_semantic_audit  # noqa: E402
from semantic_fidelity import validate_blueprint_fidelity  # noqa: E402


def manifest() -> str:
    source = "Let x be one.\nTherefore x equals one."
    return encode_formal_step_manifest(
        make_formal_step_manifest(source, [(0, 14), (14, len(source))])
    )


class StepSemanticFidelityTest(unittest.TestCase):
    def test_render_contains_only_steps(self) -> None:
        rendered = _render_step_grounded_proof(manifest(), include_ir=False)
        self.assertIn("[COT_STEP S001]", rendered)
        self.assertIn("[COT_STEP S002]", rendered)
        self.assertNotIn("COT_CLAIM", rendered)
        self.assertNotIn("COT_SCOPE", rendered)

    def test_multiple_nodes_may_map_to_one_step(self) -> None:
        target = "root"
        code = '''import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S001") (statement := /-- object -/)]
def x : Nat := 1
@[blueprint (title := "COT_STEP:S001") (statement := /-- value -/)]
lemma x_value : x = 1 := by sorry_using [x]
@[blueprint (title := "COT_STEP:S002") (statement := /-- result -/)]
theorem root : x = 1 := by sorry_using [x_value]
'''
        issues = validate_blueprint_fidelity(
            _parse_blueprint(code, target), manifest(), claimed_answer="1",
            require_step_bindings=True,
        )
        self.assertEqual(issues, [])

    def test_every_node_must_reach_root_and_no_auto_repair(self) -> None:
        code = '''import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S001") (statement := /-- orphan -/)]
lemma orphan : (1:Nat) = 1 := by sorry_using []
@[blueprint (title := "COT_STEP:S002") (statement := /-- result -/)]
theorem root : (1:Nat) = 1 := by sorry_using []
'''
        codes = {issue.code for issue in validate_blueprint_fidelity(
            _parse_blueprint(code, "root"), manifest(), claimed_answer="1",
            require_step_bindings=True,
        )}
        self.assertIn("NODE_NOT_ROOT_REACHABLE", codes)
        self.assertIn("STEP_NOT_ROOT_REACHABLE", codes)

    def test_prop_true_is_rejected(self) -> None:
        code = '''import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S001") (statement := /-- setup -/)]
def fake : Prop := True
@[blueprint (title := "COT_STEP:S002") (statement := /-- result -/)]
theorem root : (1:Nat) = 1 := by sorry_using [fake]
'''
        codes = {issue.code for issue in validate_blueprint_fidelity(
            _parse_blueprint(code, "root"), manifest(), claimed_answer="1",
            require_step_bindings=True,
        )}
        self.assertIn("VACUOUS_PROP_DEFINITION", codes)

    def test_semantic_audit_requires_complete_step_inventory(self) -> None:
        parsed = parse_semantic_audit(
            "[[SEMANTIC_AUDIT=PASS]]\n[[STEPS=S001:OK,S002:OK]]",
            expected_step_ids=("S001", "S002"),
        )
        self.assertTrue(parsed.passed)
        self.assertEqual(parsed.step_statuses, (("S001", "OK"), ("S002", "OK")))


if __name__ == "__main__":
    unittest.main()

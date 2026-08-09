from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]

from blueprint import (  # noqa: E402
    _parse_blueprint,
    _phase1a_blocking_semantic_issues,
    _render_step_grounded_proof,
)
from cot_blueprint_refine.formal_steps import (  # noqa: E402
    encode_formal_step_manifest,
    make_formal_step_manifest,
)
from semantic_audit import parse_semantic_audit  # noqa: E402
from semantic_fidelity import (  # noqa: E402
    SemanticIssue,
    format_semantic_issues,
    validate_blueprint_fidelity,
)


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
def x : Nat := 2
@[blueprint (title := "COT_STEP:S001") (statement := /-- value -/)]
lemma x_value : x = 2 := by sorry_using [x]
@[blueprint (title := "COT_STEP:S002") (statement := /-- result -/)]
theorem root : x - 1 = 1 := by sorry_using [x_value]
'''
        issues = validate_blueprint_fidelity(
            _parse_blueprint(code, target), manifest(), claimed_answer="1",
            require_step_bindings=True,
        )
        self.assertEqual(issues, [])

    def test_unreachable_node_and_step_are_warnings(self) -> None:
        code = '''import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S001") (statement := /-- orphan -/)]
lemma orphan : (2:Nat) > 1 := by sorry_using []
@[blueprint (title := "COT_STEP:S002") (statement := /-- support -/)]
lemma support : (1:Nat) < 2 := by sorry_using []
@[blueprint (title := "COT_STEP:S002") (statement := /-- result -/)]
theorem root : (1 + 0:Nat) = 1 := by sorry_using [support]
'''
        issues = validate_blueprint_fidelity(
            _parse_blueprint(code, "root"), manifest(), claimed_answer="1",
            require_step_bindings=True,
        )
        codes = {issue.code for issue in issues}
        self.assertIn("nodeNotRootReachable", codes)
        self.assertIn("stepNotRootReachable", codes)
        self.assertTrue(all(issue.severity == "warning" for issue in issues))

    def test_missing_step_is_deferred_in_phase1a_but_remains_an_error(self) -> None:
        code = '''import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S002") (statement := /-- result -/)]
theorem root : (1 : Nat) = 1 := by sorry_using []
'''
        issues = validate_blueprint_fidelity(
            _parse_blueprint(code, "root"), manifest(), claimed_answer="1",
            require_step_bindings=True,
        )
        missing = next(issue for issue in issues if issue.code == "stepMappingAbsent")
        self.assertEqual(missing.severity, "error")
        self.assertNotIn(missing, _phase1a_blocking_semantic_issues(issues))

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
        self.assertIn("vacuousPropDefinition", codes)
        issue = next(issue for issue in validate_blueprint_fidelity(
            _parse_blueprint(code, "root"), manifest(), claimed_answer="1",
            require_step_bindings=True,
        ) if issue.code == "vacuousPropDefinition")
        self.assertEqual((issue.source_start, issue.source_end), (0, 14))
        self.assertEqual(issue.category, "semanticDegeneration")

    def test_phase1a_pending_claim_is_allowed_but_phase1b_rejects_it(self) -> None:
        code = '''import Mathlib
import Architect
def PendingBlueprintClaim (_nodeId : String) : Prop := True
@[blueprint (title := "COT_STEP:S001") (statement := /-- pending -/)]
lemma setup : PendingBlueprintClaim "setup" := by sorry_using []
@[blueprint (title := "COT_STEP:S002") (statement := /-- result -/)]
theorem root : (1:Nat) = 1 := by sorry_using [setup]
'''
        blueprint = _parse_blueprint(code, "root")
        phase1a_codes = {issue.code for issue in validate_blueprint_fidelity(
            blueprint, manifest(), claimed_answer="1", require_step_bindings=True,
            allow_pending_claims=True,
        )}
        phase1b_issues = validate_blueprint_fidelity(
            blueprint, manifest(), claimed_answer="1", require_step_bindings=True,
            allow_pending_claims=False,
        )
        self.assertNotIn("unresolvedPendingClaim", phase1a_codes)
        pending = next(issue for issue in phase1b_issues if issue.code == "unresolvedPendingClaim")
        self.assertEqual((pending.step_id, pending.node_name), ("S001", "setup"))

    def test_pending_claim_must_match_node_and_cannot_be_root_or_definition(self) -> None:
        code = '''import Mathlib
import Architect
def PendingBlueprintClaim (_nodeId : String) : Prop := True
@[blueprint (title := "COT_STEP:S001") (statement := /-- wrong id -/)]
lemma setup : PendingBlueprintClaim "other" := by sorry_using []
@[blueprint (title := "COT_STEP:S002") (statement := /-- pending root -/)]
theorem root : PendingBlueprintClaim "root" := by sorry_using [setup]
'''
        issues = validate_blueprint_fidelity(
            _parse_blueprint(code, "root"), manifest(), claimed_answer="1",
            require_step_bindings=True, allow_pending_claims=True,
        )
        malformed = [issue for issue in issues if issue.code == "malformedPendingClaim"]
        self.assertEqual({issue.node_name for issue in malformed}, {"setup", "root"})

    def test_duplicate_root_conclusion_is_rejected(self) -> None:
        code = '''import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S001") (statement := /-- early answer -/)]
lemma early : (1:Nat) = 1 := by sorry_using []
@[blueprint (title := "COT_STEP:S002") (statement := /-- result -/)]
theorem root : (1:Nat) = 1 := by sorry_using [early]
'''
        issues = validate_blueprint_fidelity(
            _parse_blueprint(code, "root"), manifest(), claimed_answer="1",
            require_step_bindings=True,
        )
        issue = next(item for item in issues if item.code == "duplicateRootConclusion")
        self.assertEqual(issue.category, "answerGrounding")

    def test_source_definition_matching_claimed_answer_is_not_rejected(self) -> None:
        code = '''import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S001") (statement := /-- The circle has one arc. -/)]
def num_arcs : Nat := 1
@[blueprint (title := "COT_STEP:S002") (statement := /-- result -/)]
theorem root : num_arcs = 1 := by sorry_using [num_arcs]
'''
        codes = {issue.code for issue in validate_blueprint_fidelity(
            _parse_blueprint(code, "root"), manifest(), claimed_answer="1",
            require_step_bindings=True,
        )}
        self.assertNotIn("claimedAnswerInDefinition", codes)
        self.assertNotIn("claimedAnswerInPropDefinition", codes)

    def test_semantic_audit_requires_complete_step_inventory(self) -> None:
        parsed = parse_semantic_audit(
            "[[SEMANTIC_AUDIT=PASS]]\n[[STEPS=S001:OK,S002:OK]]",
            expected_step_ids=("S001", "S002"),
        )
        self.assertTrue(parsed.passed)
        self.assertEqual(parsed.step_statuses, (("S001", "OK"), ("S002", "OK")))

    def test_semantic_feedback_groups_all_locations_without_source_excerpts(self) -> None:
        issues = [SemanticIssue(
            "vacuousTrueStep", "A substantive Step was translated as True.",
            node_name=f"node_{index}", step_id=f"S{index:03d}",
            source_text=f"UNIQUE SOURCE EXCERPT {index}",
        ) for index in range(1, 26)]
        rendered = format_semantic_issues(issues)
        self.assertIn("vacuousTrueStep (25)", rendered)
        self.assertIn("S001/node_1", rendered)
        self.assertIn("S025/node_25", rendered)
        self.assertNotIn("UNIQUE SOURCE EXCERPT", rendered)
        self.assertNotIn("more", rendered)


if __name__ == "__main__":
    unittest.main()

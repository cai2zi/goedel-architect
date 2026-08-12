from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]

from blueprint import (  # noqa: E402
    _parse_blueprint,
    _render_step_grounded_proof,
)
from cot_blueprint_refine.formal_steps import (  # noqa: E402
    encode_formal_step_manifest,
    make_formal_step_manifest,
)
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
    def test_whole_cot_mode_accepts_no_title_and_ignores_accidental_step_title(self) -> None:
        no_title = '''import Mathlib
import Architect
@[blueprint (statement := /-- result -/) (proof := /-- arithmetic -/)]
theorem root : (1 + 1 : Nat) = 2 := by sorry_using []
'''
        self.assertEqual(validate_blueprint_fidelity(
            _parse_blueprint(no_title, "root"), None, claimed_answer="2",
            require_step_bindings=False,
        ), [])
        accidental = no_title.replace(
            "@[blueprint (statement", '@[blueprint (title := "COT_STEP:S999") (statement',
        )
        codes = {issue.code for issue in validate_blueprint_fidelity(
            _parse_blueprint(accidental, "root"), None, claimed_answer="2",
            require_step_bindings=False,
        )}
        self.assertFalse(codes & {
            "unknownStepMapping", "missingStepMapping", "rootNotFinalStep",
            "stepMappingAbsent", "stepNotRootReachable",
        })

    def test_render_contains_only_steps(self) -> None:
        rendered = _render_step_grounded_proof(manifest(), include_ir=False)
        self.assertIn("[COT_STEP S001]", rendered)
        self.assertIn("[COT_STEP S002]", rendered)
        self.assertNotIn("COT_CLAIM", rendered)
        self.assertNotIn("COT_SCOPE", rendered)

    def test_render_includes_final_answer_restatement(self) -> None:
        source = "Derive x = 1.\nTherefore, the answer is \\boxed{1}."
        boundary = source.index("Therefore")
        contract = encode_formal_step_manifest(make_formal_step_manifest(
            source, [(0, boundary), (boundary, len(source))],
        ))
        rendered = _render_step_grounded_proof(contract, include_ir=False)
        self.assertIn("Derive x = 1", rendered)
        self.assertIn("Therefore, the answer", rendered)
        self.assertIn("[COT_STEP S002]", rendered)

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

    def test_missing_step_is_a_hard_error(self) -> None:
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

    def test_false_premise_is_rejected_but_negation_is_not(self) -> None:
        code = '''import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S001") (statement := /-- ex falso -/)]
lemma bad (h : False) : (1 : Nat) = 1 := by sorry_using []
@[blueprint (title := "COT_STEP:S002") (statement := /-- valid negation shape -/)]
theorem root : (0 : Nat) = 1 → False := by sorry_using [bad]
'''
        issues = validate_blueprint_fidelity(
            _parse_blueprint(code, "root"), manifest(), claimed_answer="1",
            require_step_bindings=True,
        )
        false_premises = [issue for issue in issues if issue.code.startswith("falsePremise")]
        self.assertEqual([(issue.code, issue.node_name) for issue in false_premises], [
            ("falsePremiseStep", "bad"),
        ])

    def test_duplicate_root_conclusion_is_allowed(self) -> None:
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
        self.assertNotIn("duplicateRootConclusion", {item.code for item in issues})

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

    def test_leading_let_true_shell_is_rejected(self) -> None:
        code = '''import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S001") (statement := /-- setup -/)]
def setup : Prop := let n := 3; True
@[blueprint (title := "COT_STEP:S002") (statement := /-- result -/)]
theorem root : (1 : Nat) = 1 := by sorry_using [setup]
'''
        codes = {issue.code for issue in validate_blueprint_fidelity(
            _parse_blueprint(code, "root"), manifest(), claimed_answer="1",
            require_step_bindings=True,
        )}
        self.assertTrue({"vacuousPropDefinition", "vacuousTrueShellDefinition"} & codes)

    def test_literal_let_alias_is_reflexive_after_zeta(self) -> None:
        code = '''import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S001") (statement := /-- alias -/)]
lemma alias : let x : Nat := 1; x = 1 := by sorry_using []
@[blueprint (title := "COT_STEP:S002") (statement := /-- result -/)]
theorem root : (1 : Nat) = 1 := by sorry_using [alias]
'''
        codes = {issue.code for issue in validate_blueprint_fidelity(
            _parse_blueprint(code, "root"), manifest(), claimed_answer="1",
            require_step_bindings=True,
        )}
        self.assertIn("reflexiveStep", codes)

    def test_answer_only_existential_is_rejected(self) -> None:
        code = '''import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S001") (statement := /-- witness -/)]
lemma witness : ∃ (x y : Nat), x = 1 ∧ y = 2 := by sorry_using []
@[blueprint (title := "COT_STEP:S002") (statement := /-- result -/)]
theorem root : ∃ x : Nat, x = 1 := by sorry_using [witness]
'''
        codes = {issue.code for issue in validate_blueprint_fidelity(
            _parse_blueprint(code, "root"), manifest(), claimed_answer="1",
            require_step_bindings=True,
        )}
        self.assertIn("unboundAnswerWitnessStep", codes)
        self.assertIn("unboundAnswerWitnessRoot", codes)

    def test_nonvacuous_nullary_prop_definition_is_allowed(self) -> None:
        code = '''import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S001") (statement := /-- setup predicate -/)]
def setup : Prop := ∀ n : Nat, n + 0 = n
@[blueprint (title := "COT_STEP:S002") (statement := /-- result -/)]
theorem root : (1 : Nat) = 1 := by sorry_using [setup]
'''
        codes = {issue.code for issue in validate_blueprint_fidelity(
            _parse_blueprint(code, "root"), manifest(), claimed_answer="1",
            require_step_bindings=True,
        )}
        self.assertNotIn("vacuousPropDefinition", codes)
        self.assertNotIn("vacuousTrueShellDefinition", codes)

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

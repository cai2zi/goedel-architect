from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from blueprint import (  # noqa: E402
    _SEMANTIC_BLUEPRINT_SYSTEM_SUFFIX,
    _compact_scope_targets,
    _parse_blueprint,
    _repair_root_reachability_only,
    _render_step_grounded_proof,
    _semantic_repair_guidance,
)
from semantic_fidelity import (  # noqa: E402
    SemanticIssue,
    _strip_lean_comments,
    check_semantic_freeze,
    effective_blueprint_dependencies,
    parse_cot_manifest,
    semantic_audit_risk_reasons,
    snapshot_blueprint_semantics,
    validate_blueprint_fidelity,
)


def manifest() -> str:
    values = []
    for index, (text, role) in enumerate((
        ("Let x be the problem quantity.", "setup"),
        ("The computation claims x is 45.", "derived_claim"),
        ("Therefore the claimed answer is 45.", "conclusion"),
    ), start=1):
        values.append({
            "step_id": f"S{index:03d}",
            "source_text": text,
            "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "role": role,
            "depends_on": [f"S{index - 1:03d}"] if index > 1 else [],
            "numbers": ["45"] if index > 1 else [],
            "relations": ["eq"] if index > 1 else [],
        })
    return json.dumps(values)


def good_blueprint():
    return _parse_blueprint("""import Mathlib
import Architect

@[blueprint (title := "COT_STEP:S001") (statement := /-- quantity -/)]
def problem_quantity : ℕ := 45

@[blueprint (title := "COT_STEP:S002") (statement := /-- computed -/) (proof := /-- source computation -/)]
lemma computed_value : problem_quantity = 45 := by sorry_using [problem_quantity]

@[blueprint (title := "COT_STEP:S003") (statement := /-- final -/) (proof := /-- by computed value -/)]
theorem target : problem_quantity = 45 := by sorry_using [computed_value]
""", "target")


class SemanticFidelityTest(unittest.TestCase):
    def test_formal_identifier_references_are_root_dependencies(self) -> None:
        rows = json.loads(manifest())
        blueprint = _parse_blueprint("""import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S001") (statement := /-- quantity -/)]
def problem_quantity : ℕ := 44
@[blueprint (title := "COT_STEP:S002") (statement := /-- computed -/) (proof := /-- source -/)]
lemma computed_value : problem_quantity + 1 = 45 := by sorry_using []
@[blueprint (title := "COT_STEP:S003") (statement := /-- final -/) (proof := /-- source -/)]
theorem target : computed_value := by sorry_using []
""", "target")
        target = blueprint.node_by_name("target")
        computed = blueprint.node_by_name("computed_value")
        self.assertEqual(
            effective_blueprint_dependencies(target, blueprint.nodes_by_name()),
            ("computed_value",),
        )
        self.assertEqual(
            effective_blueprint_dependencies(computed, blueprint.nodes_by_name()),
            ("problem_quantity",),
        )
        codes = {
            issue.code for issue in validate_blueprint_fidelity(
                blueprint, rows, claimed_answer="45", require_step_bindings=True,
            )
        }
        self.assertNotIn("STEP_NOT_ROOT_REACHABLE", codes)

    def test_scope_target_ranges_are_compact_only_when_consecutive(self) -> None:
        self.assertEqual(
            _compact_scope_targets(["S004.C002", "S004.C003", "S004.C004"]),
            "S004.C002..C004",
        )
        self.assertEqual(
            _compact_scope_targets(["S004.C002", "S004.C004"]),
            "S004.C002,S004.C004",
        )

    def test_reachability_only_repair_connects_all_claim_nodes_without_semantic_drift(self) -> None:
        rows = json.loads(manifest())
        for row in rows:
            claim_id = f"{row['step_id']}.C001"
            row["claims"] = [{
                "claim_id": claim_id,
                "source_text": row["source_text"],
                "source_sha256": hashlib.sha256(row["source_text"].encode()).hexdigest(),
            }]
        blueprint = _parse_blueprint("""import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S002.C001") (statement := /-- quantity -/)]
def problem_quantity : ℕ := 40
@[blueprint (title := "COT_STEP:S001.C001") (statement := /-- typed quantity -/) (proof := /-- source setup -/)]
lemma quantity_is_nat : problem_quantity ∈ Set.univ := by sorry_using []
@[blueprint (title := "COT_STEP:S002.C001") (statement := /-- computed -/) (proof := /-- source computation -/)]
lemma computed_value : problem_quantity + 5 = 45 := by sorry_using []
@[blueprint (title := "COT_STEP:S003.C001") (statement := /-- final -/) (proof := /-- source -/)]
theorem target : problem_quantity + 5 = 45 := by sorry_using [computed_value]
""", "target")
        original_signatures = {
            node.name: (
                node.full_declaration() if node.kind == "definition" else node.signature()
            )
            for node in blueprint.nodes
        }
        issues = validate_blueprint_fidelity(
            blueprint, rows, claimed_answer="45", require_step_bindings=True,
        )
        self.assertEqual(
            {issue.code for issue in issues},
            {"CLAIM_NOT_ROOT_REACHABLE"},
        )

        repair = _repair_root_reachability_only(blueprint, issues)
        self.assertIsNotNone(repair)
        repaired_code, added = repair
        self.assertEqual(added, ["quantity_is_nat"])
        repaired = _parse_blueprint(repaired_code, "target")
        self.assertEqual(
            repaired.node_by_name("target").dependencies,
            ["computed_value", "quantity_is_nat"],
        )
        self.assertEqual(
            {
                node.name: (
                    node.full_declaration() if node.kind == "definition" else node.signature()
                )
                for node in repaired.nodes
            },
            original_signatures,
        )
        self.assertEqual(
            validate_blueprint_fidelity(
                repaired, rows, claimed_answer="45", require_step_bindings=True,
            ),
            [],
        )

    def test_reachability_repair_refuses_mixed_semantic_or_cyclic_failures(self) -> None:
        rows = json.loads(manifest())
        blueprint = _parse_blueprint("""import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S001") (statement := /-- setup -/) (proof := /-- cyclic -/)]
lemma setup_claim : True := by sorry_using [target]
@[blueprint (title := "COT_STEP:S002") (statement := /-- work -/) (proof := /-- empty -/)]
lemma empty_claim : True := by sorry_using []
@[blueprint (title := "COT_STEP:S003") (statement := /-- final -/) (proof := /-- source -/)]
theorem target : 40 + 5 = 45 := by sorry_using []
""", "target")
        issues = validate_blueprint_fidelity(
            blueprint, rows, claimed_answer="45", require_step_bindings=True,
        )
        self.assertIn("VACUOUS_TRUE_STEP", {issue.code for issue in issues})
        self.assertIsNone(_repair_root_reachability_only(blueprint, issues))

    def test_context_only_step_is_rendered_but_needs_no_blueprint_node(self) -> None:
        rows = json.loads(manifest())
        rows[0]["role"] = "context"
        rows[0]["requires_formalization"] = False
        rendered = _render_step_grounded_proof(json.dumps(rows), include_ir=False)
        self.assertIn("[COT_CONTEXT S001]", rendered)
        self.assertNotIn("[COT_CLAIM S001.", rendered)

        blueprint = _parse_blueprint("""import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S002") (statement := /-- work -/) (proof := /-- gap -/)]
lemma computed_value : 40 + 5 = 45 := by sorry_using []
@[blueprint (title := "COT_STEP:S003") (statement := /-- final -/) (proof := /-- source -/)]
theorem target : 40 + 5 = 45 := by sorry_using [computed_value]
""", "target")
        codes = {
            issue.code for issue in validate_blueprint_fidelity(
                blueprint, rows, claimed_answer="45", require_step_bindings=True,
            )
        }
        self.assertNotIn("STEP_NOT_ROOT_REACHABLE", codes)

    def test_claim_manifest_requires_exact_root_reachable_claim_titles(self) -> None:
        rows = json.loads(manifest())
        for row in rows:
            claim_id = f"{row['step_id']}.C001"
            claim_text = row["source_text"]
            row["claims"] = [{
                "claim_id": claim_id,
                "source_text": claim_text,
                "source_sha256": hashlib.sha256(claim_text.encode()).hexdigest(),
            }]
        blueprint = good_blueprint()
        issues = validate_blueprint_fidelity(
            blueprint, rows, claimed_answer="45", require_step_bindings=True,
        )
        codes = [issue.code for issue in issues]
        self.assertEqual(codes.count("MISSING_CLAIM_MAPPING"), 3)
        self.assertEqual(codes.count("CLAIM_MAPPING_ABSENT"), 3)
        self.assertNotIn("CLAIM_NOT_ROOT_REACHABLE", codes)
        self.assertIsNone(_repair_root_reachability_only(blueprint, issues))

        for node in blueprint.nodes:
            node.source_step_id = f"{node.source_step_id}.C001"
            node.lean_declaration = node.lean_declaration.replace(
                f'COT_STEP:{node.source_step_id.removesuffix(".C001")}',
                f'COT_STEP:{node.source_step_id}',
            )
        repaired_codes = {
            issue.code for issue in validate_blueprint_fidelity(
                blueprint, rows, claimed_answer="45", require_step_bindings=True,
            )
        }
        self.assertNotIn("MISSING_CLAIM_MAPPING", repaired_codes)
        self.assertNotIn("CLAIM_MAPPING_ABSENT", repaired_codes)
        self.assertNotIn("CLAIM_NOT_ROOT_REACHABLE", repaired_codes)

    def test_rendered_steps_expose_atomic_claim_labels_without_changing_step_ids(self) -> None:
        rows = json.loads(manifest())
        rows[1]["source_text"] = (
            "### Step 2: Compute\n\n"
            "The individual criterion is x = 45.\n\n"
            "Therefore the total is 45."
        )
        rendered = _render_step_grounded_proof(json.dumps(rows), include_ir=True)

        self.assertIn("[COT_CONTEXT S002] ### Step 2: Compute", rendered)
        self.assertIn("[COT_CLAIM S002.C001]", rendered)
        self.assertIn("[COT_CLAIM S002.C002]", rendered)
        self.assertIn("[COT_STEP S002 role=derived_claim", rendered)

    def test_duplicate_legacy_claim_text_keeps_distinct_claim_ids(self) -> None:
        text = "So:\n\nSo:"
        duplicate_hash = hashlib.sha256("So:".encode()).hexdigest()
        rows = [{
            "step_id": "S001",
            "source_text": text,
            "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "role": "conclusion",
            "depends_on": [],
            "numbers": [],
            "relations": [],
            "requires_formalization": True,
            "claims": [
                {"claim_id": "S001.C001", "source_text": "So:",
                 "source_sha256": duplicate_hash},
                {"claim_id": "S001.C002", "source_text": "So:",
                 "source_sha256": duplicate_hash},
            ],
        }]

        rendered = _render_step_grounded_proof(json.dumps(rows), include_ir=False)

        self.assertEqual(rendered.count("[COT_CLAIM S001.C001]"), 1)
        self.assertEqual(rendered.count("[COT_CLAIM S001.C002]"), 1)

    def test_explicit_claim_segment_renders_exact_whole_step_slice(self) -> None:
        text = "### Compute\n\nFirst x = 2.\n\nTherefore x + 1 = 3."
        claim_id = "S001.C001"
        rows = [{
            "step_id": "S001",
            "source_text": text,
            "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "role": "conclusion",
            "depends_on": [],
            "numbers": ["1", "2", "3"],
            "relations": ["eq"],
            "requires_formalization": True,
            "claims": [{
                "claim_id": claim_id,
                "source_text": text,
                "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "source_start": 0,
                "source_end": len(text),
            }],
            "segments": [{
                "kind": "claim",
                "claim_id": claim_id,
                "source_start": 0,
                "source_end": len(text),
            }],
        }]

        rendered = _render_step_grounded_proof(json.dumps(rows), include_ir=True)

        self.assertIn(f"[COT_CLAIM {claim_id}]\n{text}\n[/COT_CLAIM {claim_id}]", rendered)
        self.assertNotIn("[COT_CONTEXT S001]", rendered)

    def test_explicit_scope_renders_condition_and_target_claims(self) -> None:
        text = "### Case 1: k = 4\nThen x = 2."
        heading_end = text.index("\n") + 1
        claim_id = "S001.C001"
        rows = [{
            "step_id": "S001",
            "source_text": text,
            "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "role": "case",
            "depends_on": [],
            "numbers": ["2", "4"],
            "relations": ["eq"],
            "requires_formalization": True,
            "claims": [{
                "claim_id": claim_id,
                "source_text": text[heading_end:],
                "source_sha256": hashlib.sha256(text[heading_end:].encode()).hexdigest(),
                "source_start": heading_end,
                "source_end": len(text),
                "scope_ids": ["S001.G001"],
            }],
            "segments": [{
                "kind": "context",
                "context_type": "heading",
                "scope_type": "case_condition",
                "scope_id": "S001.G001",
                "applies_to_claim_ids": [claim_id],
                "source_start": 0,
                "source_end": heading_end,
            }, {
                "kind": "claim",
                "claim_id": claim_id,
                "source_start": heading_end,
                "source_end": len(text),
            }],
        }]

        rendered = _render_step_grounded_proof(json.dumps(rows), include_ir=False)

        self.assertIn(
            "[COT_SCOPE S001.G001 type=case_condition applies_to=S001.C001]",
            rendered,
        )
        self.assertIn("### Case 1: k = 4", rendered)
        self.assertIn("[COT_CLAIM S001.C001]", rendered)

    def test_good_grounded_graph_passes(self) -> None:
        issues = validate_blueprint_fidelity(
            good_blueprint(), parse_cot_manifest(manifest()),
            claimed_answer="45", require_step_bindings=True,
        )
        self.assertEqual(issues, [])

    def test_binding_requires_valid_root_reachable_steps(self) -> None:
        blueprint = good_blueprint()
        blueprint.nodes[1].source_step_id = ""
        blueprint.nodes[1].title = ""
        blueprint.nodes[1].lean_declaration = blueprint.nodes[1].lean_declaration.replace(
            '(title := "COT_STEP:S002") ', "",
        )
        issues = validate_blueprint_fidelity(
            blueprint, manifest(), claimed_answer="45", require_step_bindings=True,
        )
        codes = {issue.code for issue in issues}
        self.assertIn("MISSING_STEP_MAPPING", codes)
        self.assertIn("STEP_MAPPING_ABSENT", codes)
        self.assertNotIn("STEP_NOT_ROOT_REACHABLE", codes)
        self.assertIsNone(_repair_root_reachability_only(blueprint, issues))

    def test_repair_guidance_distinguishes_absent_from_disconnected_mappings(self) -> None:
        absent = _semantic_repair_guidance([
            SemanticIssue(
                code="CLAIM_MAPPING_ABSENT",
                message="missing",
                step_id="S002.C001",
                category="binding",
            ),
        ])
        disconnected = _semantic_repair_guidance([
            SemanticIssue(
                code="CLAIM_NOT_ROOT_REACHABLE",
                message="disconnected",
                step_id="S002.C001",
                category="binding",
            ),
        ])

        self.assertIn("Create a substantive formal node", absent)
        self.assertNotIn("already exist", absent)
        self.assertIn("already exist", disconnected)
        self.assertIn("preserve their declarations", disconnected)
        self.assertNotIn("Create a substantive formal node", disconnected)

    def test_detects_true_reflexive_and_unconstrained_roots(self) -> None:
        cases = {
            "true": "True",
            "reflexive": "(257 : ℚ) / 5 = 257 / 5",
            "exists": "∃ expected : ℕ, expected = 45",
        }
        expected = {
            "true": "VACUOUS_TRUE_ROOT",
            "reflexive": "REFLEXIVE_ROOT",
            "exists": "UNCONSTRAINED_EXISTS_ROOT",
        }
        for label, conclusion in cases.items():
            with self.subTest(label=label):
                blueprint = _parse_blueprint(f"""import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S003") (statement := /-- final -/) (proof := /-- source -/)]
theorem target : {conclusion} := by sorry_using []
""", "target")
                codes = {
                    issue.code for issue in validate_blueprint_fidelity(
                        blueprint, manifest(), claimed_answer="45",
                    )
                }
                self.assertIn(expected[label], codes)

    def test_vacuous_derived_definition_is_hard_rejected_and_role_is_audit_risk(self) -> None:
        blueprint = _parse_blueprint("""import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S002") (statement := /-- claim -/)]
def fake_claim : Prop := True
@[blueprint (title := "COT_STEP:S003") (statement := /-- final -/) (proof := /-- source -/)]
theorem target : 40 + 5 = 45 := by sorry_using [fake_claim]
""", "target")
        codes = {
            issue.code for issue in validate_blueprint_fidelity(
                blueprint, manifest(), claimed_answer="45",
            )
        }
        self.assertIn("VACUOUS_PROP_DEFINITION", codes)
        self.assertNotIn("DERIVED_STEP_AS_DEFINITION", codes)
        reasons = semantic_audit_risk_reasons(
            blueprint, manifest(), claimed_answer="45",
        )
        self.assertTrue(any(
            reason.startswith("DERIVED_STEP_AS_DEFINITION:step=S002:node=fake_claim")
            for reason in reasons
        ))

    def test_rejects_helper_reflexive_and_unconstrained_claims(self) -> None:
        blueprint = _parse_blueprint("""import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S001") (statement := /-- setup -/)]
def quantity : ℕ := 40
@[blueprint (title := "COT_STEP:S002") (statement := /-- claimed work -/) (proof := /-- gap -/)]
lemma fake_equality : quantity = quantity := by sorry_using [quantity]
@[blueprint (title := "COT_STEP:S002.part_b") (statement := /-- claimed witness -/) (proof := /-- gap -/)]
lemma fake_witness : ∃ x : ℕ, x = 45 := by sorry_using [fake_equality]
@[blueprint (title := "COT_STEP:S003") (statement := /-- final -/) (proof := /-- source -/)]
theorem target : quantity + 5 = 45 := by sorry_using [fake_witness]
""", "target")
        codes = {
            issue.code for issue in validate_blueprint_fidelity(
                blueprint, manifest(), claimed_answer="45",
            )
        }
        self.assertIn("REFLEXIVE_STEP", codes)
        self.assertIn("UNCONSTRAINED_EXISTS_STEP", codes)

    def test_rejects_unspecified_prop_and_compound_true_shells(self) -> None:
        blueprint = _parse_blueprint("""import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S001") (statement := /-- setup -/)]
def weakened_setup : Prop := (40 + 5 = 45) → True
@[blueprint (title := "COT_STEP:S002") (statement := /-- missing claim -/) (proof := /-- gap -/)]
lemma unspecified_claim : Prop := by sorry_using [weakened_setup]
@[blueprint (title := "COT_STEP:S002.part_b") (statement := /-- weakened claim -/) (proof := /-- gap -/)]
lemma weakened_claim : 40 + 5 = 45 ∧ True := by sorry_using [unspecified_claim]
@[blueprint (title := "COT_STEP:S003") (statement := /-- final -/) (proof := /-- source -/)]
theorem target : 40 + 5 = 45 := by sorry_using [weakened_claim]
""", "target")
        codes = {
            issue.code for issue in validate_blueprint_fidelity(
                blueprint, manifest(), claimed_answer="45",
            )
        }

        self.assertIn("VACUOUS_TRUE_SHELL_DEFINITION", codes)
        self.assertIn("VACUOUS_PROP_STEP", codes)
        self.assertIn("VACUOUS_TRUE_SHELL_STEP", codes)

    def test_rejects_parenthesized_unconstrained_existential_binder(self) -> None:
        blueprint = _parse_blueprint("""import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S003") (statement := /-- final -/) (proof := /-- source -/)]
theorem target : ∃ (expected : ℕ), expected = 45 := by sorry_using []
""", "target")
        codes = {
            issue.code for issue in validate_blueprint_fidelity(
                blueprint, manifest(), claimed_answer="45",
            )
        }
        self.assertIn("UNCONSTRAINED_EXISTS_ROOT", codes)

    def test_rejects_multiple_binders_with_true_existential_body(self) -> None:
        blueprint = _parse_blueprint("""import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S003") (statement := /-- final -/) (proof := /-- source -/)]
theorem target : ∃ (Torus Sphere : Type), True := by sorry_using []
""", "target")
        codes = {
            issue.code for issue in validate_blueprint_fidelity(
                blueprint, manifest(), claimed_answer="45",
            )
        }
        self.assertIn("UNCONSTRAINED_EXISTS_ROOT", codes)

    def test_rejects_existential_true_shell_inside_prop_definition(self) -> None:
        blueprint = _parse_blueprint("""import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S001") (statement := /-- objects -/)]
def object_shell : Prop := ∃ (Torus Sphere : Type), True
@[blueprint (title := "COT_STEP:S003") (statement := /-- final -/) (proof := /-- source -/)]
theorem target : 40 + 5 = 45 := by sorry_using [object_shell]
""", "target")
        codes = {
            issue.code for issue in validate_blueprint_fidelity(
                blueprint, manifest(), claimed_answer="45",
            )
        }
        self.assertIn("UNCONSTRAINED_EXISTS_DEFINITION", codes)

    def test_comments_cannot_hide_vacuous_formal_shapes(self) -> None:
        blueprint = _parse_blueprint("""import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S001") (statement := /-- setup -/)]
def commented_truth : Prop :=
  -- The prose claims that this contains the complete problem setup.
  /- An outer explanation with a /- nested explanation -/ inside. -/
  True
@[blueprint (title := "COT_STEP:S003") (statement := /-- final -/) (proof := /-- source -/)]
theorem target :
  -- The prose claims that this is the uniquely determined answer.
  ∃ (expected : ℕ), expected = 45 := by sorry_using [commented_truth]
""", "target")
        codes = {
            issue.code for issue in validate_blueprint_fidelity(
                blueprint, manifest(), claimed_answer="45",
            )
        }
        self.assertIn("VACUOUS_PROP_DEFINITION", codes)
        self.assertIn("UNCONSTRAINED_EXISTS_ROOT", codes)

    def test_comment_stripping_preserves_string_literals(self) -> None:
        source = (
            'def labels : List String := '
            '["-- string data", "/- block-looking data -/", '
            '"escaped \\"-- still data\\""] '
            '-- actual line comment\n'
            '/- actual outer /- nested -/ block comment -/\n'
        )
        stripped = _strip_lean_comments(source)

        self.assertEqual(len(stripped), len(source))
        self.assertEqual(stripped.count("\n"), source.count("\n"))
        self.assertIn('"-- string data"', stripped)
        self.assertIn('"/- block-looking data -/"', stripped)
        self.assertIn('"escaped \\"-- still data\\""', stripped)
        self.assertNotIn("actual line comment", stripped)
        self.assertNotIn("actual outer", stripped)
        self.assertNotIn("nested", stripped)

    def test_computation_role_may_define_a_computed_object(self) -> None:
        rows = json.loads(manifest())
        rows[1]["role"] = "computation"
        blueprint = _parse_blueprint("""import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S002") (statement := /-- computed object -/)]
def computed_object : ℕ := 40 + 5
@[blueprint (title := "COT_STEP:S003") (statement := /-- final -/) (proof := /-- source -/)]
theorem target : computed_object = 45 := by sorry_using [computed_object]
""", "target")
        codes = {
            issue.code for issue in validate_blueprint_fidelity(
                blueprint, rows, claimed_answer="45",
            )
        }
        self.assertNotIn("DERIVED_STEP_AS_DEFINITION", codes)

    def test_hard_rejects_nullary_definition_equal_to_claimed_answer_for_claim_roles(self) -> None:
        for role in (
            "derived_claim", "verification", "conclusion", "final", "final_claim",
        ):
            with self.subTest(role=role):
                rows = json.loads(manifest())
                rows[1]["role"] = role
                blueprint = _parse_blueprint("""import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S002") (statement := /-- claimed value -/)]
def hardcoded_claim : ℕ := (45 : ℕ)
@[blueprint (title := "COT_STEP:S003") (statement := /-- final -/) (proof := /-- source -/)]
theorem target : hardcoded_claim = 45 := by sorry_using [hardcoded_claim]
""", "target")
                issues = validate_blueprint_fidelity(
                    blueprint, rows, claimed_answer="45",
                )
                matches = [
                    issue for issue in issues
                    if issue.code == "HARDCODED_CLAIMED_ANSWER_DEFINITION"
                ]

                self.assertEqual(len(matches), 1)
                self.assertEqual(matches[0].node_name, "hardcoded_claim")
                self.assertEqual(matches[0].step_id, "S002")
                self.assertEqual(matches[0].category, "static")
                self.assertIsNone(_repair_root_reachability_only(blueprint, matches))

    def test_hardcoded_answer_gate_exempts_nonclaim_roles(self) -> None:
        for role in ("setup", "given", "object_definition", "computation"):
            with self.subTest(role=role):
                rows = json.loads(manifest())
                rows[1]["role"] = role
                blueprint = _parse_blueprint("""import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S002") (statement := /-- source object -/)]
def source_object : ℕ := 45
@[blueprint (title := "COT_STEP:S003") (statement := /-- final -/) (proof := /-- source -/)]
theorem target : source_object = 45 := by sorry_using [source_object]
""", "target")
                codes = {
                    issue.code for issue in validate_blueprint_fidelity(
                        blueprint, rows, claimed_answer="45",
                    )
                }

                self.assertNotIn("HARDCODED_CLAIMED_ANSWER_DEFINITION", codes)

    def test_hardcoded_answer_gate_exempts_parameters_and_expressions(self) -> None:
        blueprint = _parse_blueprint("""import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S002.part_a") (statement := /-- function -/)]
def parameterized (x : ℕ) : ℕ := 45
@[blueprint (title := "COT_STEP:S002.part_b") (statement := /-- calculation -/)]
def calculated : ℕ := 40 + 5
@[blueprint (title := "COT_STEP:S002.part_c") (statement := /-- answer-bearing expression -/)]
def contains_answer : ℕ := 45 + 0
@[blueprint (title := "COT_STEP:S003") (statement := /-- final -/) (proof := /-- source -/)]
theorem target : calculated = 45 := by sorry_using [parameterized, calculated, contains_answer]
""", "target")
        codes = {
            issue.code for issue in validate_blueprint_fidelity(
                blueprint, manifest(), claimed_answer="45",
            )
        }

        self.assertNotIn("HARDCODED_CLAIMED_ANSWER_DEFINITION", codes)

    def test_freeze_allows_proof_metadata_but_rejects_root_change(self) -> None:
        baseline_blueprint = good_blueprint()
        snapshot = snapshot_blueprint_semantics(baseline_blueprint, manifest())
        metadata_only = _parse_blueprint(
            baseline_blueprint.lean_file.replace(
                "source computation", "a different Lean proof plan",
            ),
            "target",
        )
        self.assertEqual(check_semantic_freeze(snapshot, metadata_only, manifest()), [])

        changed = _parse_blueprint(
            baseline_blueprint.lean_file.replace(
                "theorem target : problem_quantity = 45",
                "theorem target : problem_quantity = 46",
            ),
            "target",
        )
        codes = {issue.code for issue in check_semantic_freeze(snapshot, changed, manifest())}
        self.assertIn("ROOT_SIGNATURE_DRIFT", codes)
        self.assertIn("STEP_SEMANTIC_DRIFT", codes)

    def test_manifest_rejects_source_hash_mutation(self) -> None:
        rows = json.loads(manifest())
        rows[0]["source_text"] = "silently changed"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            parse_cot_manifest(rows)

    def test_risk_router_uses_minimal_ir_without_an_llm(self) -> None:
        reasons = semantic_audit_risk_reasons(
            good_blueprint(), manifest(), claimed_answer="45",
        )
        # S002 says 45 and equality and both remain visible; the answer also
        # appears in a setup definition, which is legal but audit-worthy.
        self.assertIn("ANSWER_IN_DEFINITION:problem_quantity", reasons)
        changed_rows = json.loads(manifest())
        changed_rows[1]["numbers"] = ["999"]
        reasons = semantic_audit_risk_reasons(
            good_blueprint(), changed_rows, claimed_answer="45",
        )
        self.assertIn("NUMBER_NOT_VISIBLE:S002:999", reasons)

    def test_future_numbered_dependency_is_audit_risk_not_hard_reject(self) -> None:
        blueprint = _parse_blueprint("""import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S002") (statement := /-- detailed setup -/) (proof := /-- source -/)]
lemma detailed_setup : 40 + 5 = 45 := by sorry_using []
@[blueprint (title := "COT_STEP:S001") (statement := /-- opening overview -/) (proof := /-- source -/)]
lemma opening_overview : 40 + 5 = 45 := by sorry_using [detailed_setup]
@[blueprint (title := "COT_STEP:S003") (statement := /-- conclusion -/) (proof := /-- source -/)]
theorem target : 40 + 5 = 45 := by sorry_using [opening_overview]
""", "target")

        issues = validate_blueprint_fidelity(
            blueprint, manifest(), claimed_answer="45", require_step_bindings=True,
        )
        self.assertNotIn("FUTURE_STEP_DEPENDENCY", {issue.code for issue in issues})

        reasons = semantic_audit_risk_reasons(
            blueprint, manifest(), claimed_answer="45",
        )
        self.assertIn(
            "FUTURE_STEP_DEPENDENCY:node=opening_overview:source=S001:"
            "dependency=detailed_setup:dependency_source=S002",
            reasons,
        )

    def test_semantic_prompt_treats_step_ids_as_provenance_not_chronology(self) -> None:
        self.assertNotIn(
            "Dependencies must follow source-step order",
            _SEMANTIC_BLUEPRINT_SYSTEM_SUFFIX,
        )
        self.assertIn(
            "Step identifiers record provenance, not a forced\nchronology",
            _SEMANTIC_BLUEPRINT_SYSTEM_SUFFIX,
        )
        self.assertIn("actual logical dependencies", _SEMANTIC_BLUEPRINT_SYSTEM_SUFFIX)

    def test_semantic_prompt_requires_constraints_in_formal_lean(self) -> None:
        self.assertIn(
            "Only the formal Lean type and definition body count",
            _SEMANTIC_BLUEPRINT_SYSTEM_SUFFIX,
        )
        self.assertIn(
            "Comments, docstrings, and natural-language `statement` fields do not encode",
            _SEMANTIC_BLUEPRINT_SYSTEM_SUFFIX,
        )
        self.assertIn("quantifier and its polarity", _SEMANTIC_BLUEPRINT_SYSTEM_SUFFIX)
        self.assertIn("A derived equation must be a\nconclusion", _SEMANTIC_BLUEPRINT_SYSTEM_SUFFIX)
        self.assertIn("same formal quantity connected", _SEMANTIC_BLUEPRINT_SYSTEM_SUFFIX)
        self.assertIn("coordinate choice or normalization", _SEMANTIC_BLUEPRINT_SYSTEM_SUFFIX)
        self.assertIn("typed abstract relational model", _SEMANTIC_BLUEPRINT_SYSTEM_SUFFIX)
        self.assertIn("relation or function binders", _SEMANTIC_BLUEPRINT_SYSTEM_SUFFIX)
        self.assertIn("Reuse those same formal objects", _SEMANTIC_BLUEPRINT_SYSTEM_SUFFIX)
        self.assertIn("Coverage is clause-level", _SEMANTIC_BLUEPRINT_SYSTEM_SUFFIX)
        self.assertIn("lemma cot_total_jump : N = K", _SEMANTIC_BLUEPRINT_SYSTEM_SUFFIX)
        self.assertIn("restrictedCount = K", _SEMANTIC_BLUEPRINT_SYSTEM_SUFFIX)


if __name__ == "__main__":
    unittest.main()

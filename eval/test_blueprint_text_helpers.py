"""Regression tests for blueprint attribute and signature text handling."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from blueprint import (  # noqa: E402
    BlueprintNode,
    _extract_lean_code,
    _parse_blueprint,
    _unannotated_local_declaration_errors,
    canonicalize_blueprint,
    extract_blueprint_signature,
    format_phase2_standalone_issues,
    phase2_contract_error_counts,
    phase2_contract_errors,
    phase2_standalone_contract_errors,
    phase2_standalone_contract_report,
    render_solved_declaration,
    strip_blueprint_attr,
)
from blueprint_text import extract_current_node_decl  # noqa: E402
from kimina_lean_compiler import CompilerResult, assemble_node_attempt  # noqa: E402


ZMOD_NODE = """@[blueprint
  (statement := /-- 2003 is congruent to 1 modulo 7. -/)
  (proof := /-- Compute 2003 = 1 [ZMOD 7]. -/)]
lemma mod7_2003_eq_1 :
    (2003 : Int) = (1 : Int) [ZMOD (7 : Int)] := by
  sorry_using []
"""


class BlueprintTextHelpersTest(unittest.TestCase):
    def test_canonical_header_adds_bigoperators_once_and_moves_late_open(self) -> None:
        lean_code = """import Mathlib
import Architect
@[blueprint] def d : Nat := 1
@[blueprint] theorem root : d = 1 := by sorry_using [d]
open scoped BigOperators
"""
        parsed = _parse_blueprint(lean_code, "root")
        canonical = canonicalize_blueprint(parsed, list(parsed.nodes))

        self.assertEqual(canonical.lean_file.count("open scoped BigOperators"), 1)
        self.assertEqual(canonical.lean_file.count("set_option autoImplicit false"), 1)
        self.assertLess(
            canonical.lean_file.index("open scoped BigOperators"),
            canonical.lean_file.index("@[blueprint]"),
        )
        self.assertEqual(canonical.phase2_header, parsed.phase2_header)
        self.assertEqual(_unannotated_local_declaration_errors(parsed), [])

    def test_canonical_header_replaces_autoimplicit_and_deduplicates_scope(self) -> None:
        lean_code = """import Architect
set_option autoImplicit true
open scoped BigOperators
open scoped BigOperators
@[blueprint] theorem root : True := by sorry_using []
"""
        blueprint = _parse_blueprint(lean_code, "root")
        self.assertTrue(blueprint.phase2_header.startswith(
            "import Mathlib\nimport Architect\nset_option autoImplicit false\n"
            "open scoped BigOperators\n"
        ))
        self.assertNotIn("autoImplicit true", blueprint.phase2_header)
        self.assertEqual(blueprint.phase2_header.count("BigOperators"), 1)

    def test_standalone_contract_forwards_batch_concurrency(self) -> None:
        declarations = []
        for index in range(17):
            declarations.append(f"""@[blueprint
  (statement := /-- Node {index}. -/)
  (proof := /-- Trivial. -/)]
lemma node_{index} : True := by
  sorry_using []
""")
        declarations.append("""@[blueprint
  (statement := /-- Root. -/)
  (proof := /-- Trivial. -/)]
theorem root : True := by
  sorry_using []
""")
        blueprint = _parse_blueprint("\n".join(declarations), "root")

        class RecordingCompiler:
            batch_concurrency = 0
            request_count = 0

            def check_many(self, requests, *, batch_concurrency=1):
                self.batch_concurrency = batch_concurrency
                self.request_count = len(requests)
                return [CompilerResult(True) for _ in requests]

        compiler = RecordingCompiler()
        errors = phase2_standalone_contract_errors(
            blueprint, compiler, concurrency=8,
        )

        self.assertEqual(errors, [])
        self.assertEqual(compiler.batch_concurrency, 8)
        self.assertEqual(compiler.request_count, 18)

    def test_standalone_report_maps_ambient_identifier_to_target_node(self) -> None:
        blueprint = _parse_blueprint('''import Mathlib
variable (x : Nat)
@[blueprint
  (statement := /-- Preserve the ambient value. -/)
  (proof := /-- Reflexive only for the contract fixture. -/)]
theorem root : x = x := by sorry_using []
''', "root")

        class Compiler:
            def check_many(self, requests, **_kwargs):
                self.request_code = requests[0].lean_code
                return [CompilerResult(False, errors=[
                    '{"severity":"error","pos":{"line":5,"column":15},'
                    '"data":"Unknown identifier `x`"}'
                ])]

        compiler = Compiler()
        report = phase2_standalone_contract_report(blueprint, compiler)
        self.assertNotIn("variable (x", compiler.request_code)
        self.assertEqual(len(report.issues), 1)
        issue = report.issues[0]
        self.assertEqual(issue.code, "phase2StandaloneFailed")
        self.assertEqual(issue.error_kind, "unknownIdentifier")
        self.assertEqual(issue.identifiers, ("x",))
        self.assertEqual(issue.node_name, "root")

    def test_standalone_cache_reuses_unchanged_code_and_invalidates_edit(self) -> None:
        blueprint = _parse_blueprint('''import Mathlib
@[blueprint (title := "COT_STEP:S001")
  (statement := /-- One equals one. -/)
  (proof := /-- Deferred. -/)]
theorem root : (1 : Nat) = 1 := by sorry_using []
''', "root")

        class Compiler:
            request_count = 0

            def check_many(self, requests, **_kwargs):
                self.request_count += len(requests)
                return [CompilerResult(True) for _ in requests]

        compiler = Compiler()
        cache = {}
        first = phase2_standalone_contract_report(blueprint, compiler, cache=cache)
        second = phase2_standalone_contract_report(blueprint, compiler, cache=cache)
        self.assertEqual((first.cached_node_count, second.cached_node_count), (0, 1))
        self.assertEqual(compiler.request_count, 1)

        replacement = '''@[blueprint (title := "COT_STEP:S001")
  (statement := /-- Two minus one equals one. -/)
  (proof := /-- Deferred. -/)]
theorem root : (2 : Nat) - 1 = 1 := by sorry_using []'''
        revised = _parse_blueprint("import Mathlib\n" + replacement, "root")
        third = phase2_standalone_contract_report(revised, compiler, cache=cache)
        self.assertEqual(third.cached_node_count, 0)
        self.assertEqual(compiler.request_count, 2)

    def test_standalone_feedback_groups_repeated_root_cause_without_dropping_nodes(self) -> None:
        declarations = []
        for index in range(13):
            kind = "theorem" if index == 12 else "lemma"
            name = "root" if index == 12 else f"node_{index}"
            declarations.append(f'''@[blueprint
  (statement := /-- Node {index}. -/)
  (proof := /-- Deferred. -/)]
{kind} {name} : missingObject = missingObject := by sorry_using []''')
        blueprint = _parse_blueprint("\n".join(declarations), "root")

        class Compiler:
            def check_many(self, requests, **_kwargs):
                return [CompilerResult(False, errors=[
                    '{"severity":"error","pos":{"line":5,"column":1},'
                    '"data":"Unknown identifier `missingObject`"}'
                ]) for _ in requests]

        report = phase2_standalone_contract_report(blueprint, Compiler())
        rendered = format_phase2_standalone_issues(report.issues)
        self.assertEqual(len(report.issues), 13)
        self.assertIn("node_0", rendered)
        self.assertIn("root", rendered)
        self.assertEqual(rendered.count("Representative diagnostic"), 1)

    def test_extract_lean_code_treats_none_content_as_empty(self) -> None:
        self.assertEqual(_extract_lean_code(None), "")

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

    def test_full_declaration_preserves_definition_rhs_and_nested_assignment(self) -> None:
        declarations = {
            "regular": """@[blueprint
  (statement := /-- A pair built through a local binding. -/)]
def regular : Nat × Nat :=
  let left := 1
  (left, 2)
""",
            "choice": """@[blueprint (statement := /-- A chosen value. -/)]
noncomputable def choice : Nat := Classical.choose (show Nonempty Nat from ⟨1⟩)
""",
            "short": """@[blueprint (statement := /-- A short alias. -/)]
abbrev short : Nat := 3
""",
        }

        for name, declaration in declarations.items():
            with self.subTest(name=name):
                node = BlueprintNode(
                    name=name,
                    kind="definition",
                    statement="",
                    proof_sketch="",
                    lean_declaration=declaration,
                )
                full = node.full_declaration()
                self.assertNotIn("@[blueprint", full)
                self.assertIn(":=", full)

        regular = BlueprintNode(
            name="regular",
            kind="definition",
            statement="",
            proof_sketch="",
            lean_declaration=declarations["regular"],
        )
        self.assertIn("let left := 1", regular.full_declaration())
        self.assertEqual(regular.full_declaration().count(":="), 2)

    def test_definition_renderer_ignores_polluted_cached_body(self) -> None:
        declaration = """@[blueprint (statement := /-- Base value. -/)]
def base_val : ℝ := (30 : ℝ)
"""
        node = BlueprintNode(
            name="base_val",
            kind="definition",
            statement="",
            proof_sketch="",
            lean_declaration=declaration,
        )
        polluted = "@[blueprint] def base_val : ℝ := (999 : ℝ)"
        rendered = render_solved_declaration(node, polluted)

        self.assertEqual(rendered, "def base_val : ℝ := (30 : ℝ)")
        self.assertNotIn("999", rendered)

    def test_definition_cache_key_includes_rhs(self) -> None:
        def node(rhs: str) -> BlueprintNode:
            return BlueprintNode(
                name="value",
                kind="definition",
                statement="",
                proof_sketch="",
                lean_declaration=f"def value : Nat := {rhs}",
            )

        self.assertNotEqual(node("1").cache_key(), node("2").cache_key())

    def test_node_decl_ignores_declaration_words_inside_blueprint_comments(self) -> None:
        declaration = """@[blueprint
  (statement := /-- The target theorem. -/)
  (proof := /-- The hypotheses are part of the original theorem statement. -/)]
theorem mathd_algebra_478 : True := by
  sorry_using []
"""
        expected = "theorem mathd_algebra_478 : True := by\n  sorry_using []"

        self.assertEqual(extract_current_node_decl(declaration), expected)

        assembled = assemble_node_attempt(declaration, "", "by trivial", "import Mathlib")
        self.assertNotIn("@[blueprint", assembled)
        self.assertNotIn("theorem statement", assembled)
        self.assertIn("theorem mathd_algebra_478 : True := by trivial", assembled)

    def test_proof_node_without_placeholder_fails_before_backend(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one"):
            assemble_node_attempt(
                "theorem already_complete : True := by trivial",
                "",
                "by trivial",
                "import Mathlib",
            )

    def test_phase2_contract_accepts_definitions_and_sorry_using_proof_nodes(self) -> None:
        lean_code = """@[blueprint (statement := /-- Helper value. -/)]
def helper : Nat := 1

@[blueprint
  (statement := /-- The target. -/)
  (proof := /-- Use `helper`. -/)]
theorem root : helper = 1 := by
  sorry_using [helper]
"""
        blueprint = _parse_blueprint(lean_code, "root")

        self.assertEqual(phase2_contract_errors(blueprint), [])

    def test_phase2_contract_rejects_completed_proof_node(self) -> None:
        lean_code = """@[blueprint
  (statement := /-- Already proved, but not phase2-spliceable. -/)
  (proof := /-- Trivial. -/)]
theorem root : True := by
  trivial
"""
        blueprint = _parse_blueprint(lean_code, "root")
        errors = phase2_contract_errors(blueprint)

        self.assertEqual(
            phase2_contract_error_counts(errors),
            {"missing_sorry_using_placeholder": 1},
        )
        self.assertIn("root", errors[0])

    def test_phase2_contract_rejects_plain_sorry_proof_node(self) -> None:
        lean_code = """@[blueprint
  (statement := /-- Plain sorry is accepted by Lean but unusable by phase2. -/)
  (proof := /-- Trivial. -/)]
theorem root : True := by
  sorry
"""
        blueprint = _parse_blueprint(lean_code, "root")
        errors = phase2_contract_errors(blueprint)

        self.assertEqual(
            phase2_contract_error_counts(errors),
            {"missing_sorry_using_placeholder": 1},
        )

    def test_dependency_order_rejects_cycles(self) -> None:
        lean_code = """@[blueprint
  (statement := /-- First. -/)
  (proof := /-- Uses b. -/)]
lemma a : True := by
  sorry_using [b]

@[blueprint
  (statement := /-- Second. -/)
  (proof := /-- Uses a. -/)]
theorem b : True := by
  sorry_using [a]
"""
        blueprint = _parse_blueprint(lean_code, "b")

        with self.assertRaisesRegex(ValueError, "a -> b -> a"):
            blueprint.dependency_order()

        self.assertEqual(
            phase2_contract_error_counts(phase2_contract_errors(blueprint)),
            {"dependency_cycle": 1},
        )

    def test_mathd_numbertheory_227_style_blueprint_is_not_phase2_ready(self) -> None:
        lean_code = """@[blueprint (statement := /-- Coffee contribution. -/)]
noncomputable def coffee_share (x : ℕ+) : ℝ := (x : ℝ) / 4

@[blueprint (statement := /-- Milk contribution. -/)]
noncomputable def milk_share (y : ℕ+) : ℝ := (y : ℝ) / 6

@[blueprint (statement := /-- Balance relation. -/)]
noncomputable def family_balance (x y n : ℕ+) : Prop :=
  coffee_share x + milk_share y = ((x + y : ℕ+) : ℝ) / n

@[blueprint
  (statement := /-- Rewrites the hypothesis. -/)
  (proof := /-- Direct unfolding. -/)]
lemma h0_is_family_balance (x y n : ℕ+)
    (h₀ : ↑x / (4 : ℝ) + y / 6 = (x + y) / n) :
    family_balance x y n := by
  dsimp [family_balance, coffee_share, milk_share] at *
  simpa using h₀

@[blueprint
  (statement := /-- Core arithmetic. -/)
  (proof := /-- Deferred. -/)]
lemma family_balance_implies_five (x y n : ℕ+)
    (h : family_balance x y n) : n = 5 := by
  sorry

@[blueprint
  (statement := /-- Root. -/)
  (proof := /-- Apply helpers. -/)]
theorem mathd_numbertheory_227 (x y n : ℕ+)
    (h₀ : ↑x / (4 : ℝ) + y / 6 = (x + y) / n) : n = 5 := by
  have h : family_balance x y n := h0_is_family_balance x y n h₀
  exact family_balance_implies_five x y n h
"""
        blueprint = _parse_blueprint(lean_code, "mathd_numbertheory_227")
        errors = phase2_contract_errors(blueprint)

        self.assertEqual(
            phase2_contract_error_counts(errors),
            {"missing_sorry_using_placeholder": 3},
        )


if __name__ == "__main__":
    unittest.main()

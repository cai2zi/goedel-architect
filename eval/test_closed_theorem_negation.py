from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from prover import _build_negation_node_decl  # noqa: E402


class ClosedTheoremNegationTest(unittest.TestCase):
    def test_closed_proposition_without_outer_binders(self) -> None:
        decl = "theorem root : ∀ n : Nat, n = 0 := by sorry_using []"
        self.assertEqual(
            _build_negation_node_decl(decl, "root"),
            "theorem neg_root : ¬ (∀ n : Nat, n = 0) := by sorry_using []",
        )

    def test_explicit_parameters_and_hypotheses_move_under_negation(self) -> None:
        decl = (
            "lemma step (n : Nat) (h : 5 < n) : n = 7 := "
            "by sorry_using []"
        )
        self.assertEqual(
            _build_negation_node_decl(decl, "step"),
            "theorem neg_step : ¬ (∀ (n : Nat) (h : 5 < n), n = 7) "
            ":= by sorry_using []",
        )

    def test_implicit_and_instance_binders_move_under_negation(self) -> None:
        decl = (
            "theorem impossible {α : Type} [Inhabited α] (x : α) : False := "
            "by sorry_using []"
        )
        self.assertEqual(
            _build_negation_node_decl(decl, "impossible"),
            "theorem neg_impossible : ¬ (∀ {α : Type} [Inhabited α] (x : α), False) "
            ":= by sorry_using []",
        )

    def test_blueprint_metadata_is_removed_and_name_is_sanitized(self) -> None:
        decl = (
            "@[blueprint (statement := /-- text -/) (proof := /-- sketch -/)]\n"
            "lemma step-name (x : Nat) : x = 0 := by sorry_using [parent]"
        )
        self.assertEqual(
            _build_negation_node_decl(decl, "step-name"),
            "theorem neg_step_name : ¬ (∀ (x : Nat), x = 0) := by sorry_using []",
        )

    def test_rejects_empty_conclusion(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty theorem conclusion"):
            _build_negation_node_decl(
                "theorem bad (x : Nat) : := by sorry_using []", "bad",
            )


if __name__ == "__main__":
    unittest.main()

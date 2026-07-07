import Mathlib
import Architect
import Frap.Exercises.Equiv

namespace Imp
open AExp
open BExp
open Com
open CEval

attribute [local simp]
  aeval beval aequiv bequiv cequiv

@[blueprint
  (statement := /-- Folding an arithmetic expression preserves its evaluation in every state, in the orientation useful for rewriting from the folded expression back to the original expression. -/)
  (proof := /-- This is just `fold_constants_aexp_sound` specialized to `a` and `st`, with symmetry if needed. -/)]
lemma fold_constants_aexp_eval_folded_eq (a : AExp) (st) :
    aeval st (fold_constants_aexp a) = aeval st a := := by
  exact (fold_constants_aexp_sound a st).symm

@[blueprint
  (statement := /-- If two folded arithmetic expressions are exposed by equations `fold_constants_aexp a₁ = a₁'` and `fold_constants_aexp a₂ = a₂'`, then evaluating `a₁'` and `a₂'` agrees with evaluating the original expressions. -/)
  (proof := /-- Rewrite `a₁'` and `a₂'` using the two defining equations and apply arithmetic constant-folding soundness to both subexpressions. -/)]
lemma fold_constants_aexp_eval_of_generalized
    (a₁ a₂ a₁' a₂' : AExp) (st)
    (h₁ : fold_constants_aexp a₁ = a₁')
    (h₂ : fold_constants_aexp a₂ = a₂') :
    aeval st a₁' = aeval st a₁ ∧ aeval st a₂' = aeval st a₂ := := by
  constructor
  · rw [← h₁]
    exact fold_constants_aexp_eval_folded_eq a₁ st
  · rw [← h₂]
    exact fold_constants_aexp_eval_folded_eq a₂ st

@[blueprint
  (statement := /-- A single generalized folded arithmetic expression evaluates like its original expression. -/)
  (proof := /-- Rewrite the generalized folded expression using `h` and use arithmetic constant-folding soundness in the folded-to-original orientation. -/)]
lemma fold_constants_aexp_eval_one_of_generalized
    (a a' : AExp) (st)
    (h : fold_constants_aexp a = a') :
    aeval st a' = aeval st a := := by
  subst a'
  exact fold_constants_aexp_eval_folded_eq a st

@[blueprint
  (statement := /-- If a folded arithmetic expression is a numeral, then the original expression evaluates to that numeral. -/)
  (proof := /-- Combine the generalized evaluation lemma with the fact that `aeval st (a_num n) = n`. -/)]
lemma fold_constants_aexp_eval_eq_num_of_fold_eq_num
    (a : AExp) (st) (n : Nat)
    (h : fold_constants_aexp a = a_num n) :
    aeval st a = n := := by
  have hgen : aeval st (a_num n) = aeval st a := fold_constants_aexp_eval_one_of_generalized a (a_num n) st h
  simpa using hgen.symm

@[blueprint
  (statement := /-- If two arithmetic expressions are folded, the equality test over the originals evaluates like the equality test over their folded forms. -/)
  (proof := /-- Evaluate both equality tests and use arithmetic folding soundness on both operands. -/)]
lemma beval_b_eq_of_folded
    (a₁ a₂ : AExp) (st) :
    beval st (b_eq a₁ a₂) =
    beval st (b_eq (fold_constants_aexp a₁) (fold_constants_aexp a₂)) := := by
  unfold beval
  rw [fold_constants_aexp_sound a₁ st]
  rw [fold_constants_aexp_sound a₂ st]

@[blueprint
  (statement := /-- If two arithmetic expressions are folded, the disequality test over the originals evaluates like the disequality test over their folded forms. -/)
  (proof := /-- Evaluate both disequality tests and use arithmetic folding soundness on both operands. -/)]
lemma beval_b_neq_of_folded
    (a₁ a₂ : AExp) (st) :
    beval st (b_neq a₁ a₂) =
    beval st (b_neq (fold_constants_aexp a₁) (fold_constants_aexp a₂)) := := by
  simp [beval]
  rw [fold_constants_aexp_sound a₁ st, fold_constants_aexp_sound a₂ st]

@[blueprint
  (statement := /-- If two arithmetic expressions are folded, the less-or-equal test over the originals evaluates like the less-or-equal test over their folded forms. -/)
  (proof := /-- Evaluate both less-or-equal tests and use arithmetic folding soundness on both operands. -/)]
lemma beval_b_le_of_folded
    (a₁ a₂ : AExp) (st) :
    beval st (b_le a₁ a₂) =
    beval st (b_le (fold_constants_aexp a₁) (fold_constants_aexp a₂)) := := by
  simp [beval]
  rw [fold_constants_aexp_sound a₁ st, fold_constants_aexp_sound a₂ st]

@[blueprint
  (statement := /-- If two folded arithmetic subexpressions have been generalized to `a₁'` and `a₂'`, then the equality test over the originals has the same value as the equality test over the generalized folded expressions. -/)
  (proof := /-- Use `fold_constants_aexp_eval_of_generalized` to rewrite the evaluations of `a₁'` and `a₂'` to the evaluations of `a₁` and `a₂`, then compute `beval` for equality tests. -/)]
lemma beval_b_eq_of_folded_generalized
    (a₁ a₂ a₁' a₂' : AExp) (st)
    (h₁ : fold_constants_aexp a₁ = a₁')
    (h₂ : fold_constants_aexp a₂ = a₂') :
    beval st (b_eq a₁ a₂) = beval st (b_eq a₁' a₂') := := by
  unfold beval
  rw [fold_constants_aexp_sound a₁ st]
  rw [fold_constants_aexp_sound a₂ st]

@[blueprint
  (statement := /-- If two folded arithmetic subexpressions have been generalized to `a₁'` and `a₂'`, then the disequality test over the originals has the same value as the disequality test over the generalized folded expressions. -/)
  (proof := /-- Use the arithmetic generalized-evaluation helper to identify both arithmetic evaluations, then simplify `beval` for disequality. -/)]
lemma beval_b_neq_of_folded_generalized
    (a₁ a₂ a₁' a₂' : AExp) (st)
    (h₁ : fold_constants_aexp a₁ = a₁')
    (h₂ : fold_constants_aexp a₂ = a₂') :
    beval st (b_neq a₁ a₂) = beval st (b_neq a₁' a₂') := := by
  simp [beval]
  rw [fold_constants_aexp_sound a₁ st, fold_constants_aexp_sound a₂ st]

@[blueprint
  (statement := /-- If two folded arithmetic subexpressions have been generalized to `a₁'` and `a₂'`, then the less-or-equal test over the originals has the same value as the less-or-equal test over the generalized folded expressions. -/)
  (proof := /-- Use the arithmetic generalized-evaluation helper to identify both arithmetic evaluations, then simplify `beval` for less-or-equal. -/)]
lemma beval_b_le_of_folded_generalized
    (a₁ a₂ a₁' a₂' : AExp) (st)
    (h₁ : fold_constants_aexp a₁ = a₁')
    (h₂ : fold_constants_aexp a₂ = a₂') :
    beval st (b_le a₁ a₂) = beval st (b_le a₁' a₂') := := by
  simp [beval]
  rw [fold_constants_aexp_sound a₁ st, fold_constants_aexp_sound a₂ st]

@[blueprint
  (statement := /-- In the numeric equality-folding case, evaluating the computed boolean constant agrees with evaluating the equality test on the two numerals. -/)
  (proof := /-- Split on the decidable boolean equality `n₁ == n₂`; in each branch both sides compute by `beval` and `aeval`. -/)]
lemma beval_fold_constants_bexp_eq_num_case
    (n₁ n₂ : Nat) (st) :
    beval st (if n₁ == n₂ then b_true else b_false) =
    beval st (b_eq (a_num n₁) (a_num n₂)) := := by
  have h := fold_constants_aexp_eval_of_generalized a₁ a₂ a₁' a₂' st h₁ h₂
  unfold beval
  rw [h.1, h.2]

@[blueprint
  (statement := /-- The constant-folding match for equality tests agrees, after evaluation, with the residual equality test on the already-folded arithmetic expressions. -/)
  (proof := /-- Split on the forms of the two already-folded arithmetic expressions. In the numeral/numeral case, use the dedicated numeric equality helper. In all residual cases, the match returns the residual equality test directly. -/)]
lemma beval_fold_constants_bexp_eq_match
    (a₁' a₂' : AExp) (st) :
    beval st
      (match a₁', a₂' with
       | a_num n₁, a_num n₂ => if n₁ == n₂ then b_true else b_false
       | a₁'', a₂'' => b_eq a₁'' a₂'')
    =
    beval st (b_eq a₁' a₂') := := by
  have h₁eval := fold_constants_aexp_eval_one_of_generalized a₁ a₁' st h₁
  have h₂eval := fold_constants_aexp_eval_one_of_generalized a₂ a₂' st h₂
  simp [beval, h₁eval, h₂eval]

@[blueprint
  (statement := /-- The root equality case of boolean constant folding unfolds definitionally to the equality-folding match over the two folded arithmetic operands. -/)
  (proof := /-- Unfold `fold_constants_bexp` at the root constructor `b_eq`; the result is exactly the displayed match. -/)]
lemma fold_constants_bexp_eq_root_unfold (a₁ a₂ : AExp) :
    fold_constants_bexp (b_eq a₁ a₂) =
      (match fold_constants_aexp a₁, fold_constants_aexp a₂ with
       | a_num n₁, a_num n₂ => if n₁ == n₂ then b_true else b_false
       | a₁'', a₂'' => b_eq a₁'' a₂'') := := by
  have h := fold_constants_aexp_eval_of_generalized a₁ a₂ a₁' a₂' st h₁ h₂
  simp [beval]
  rw [h.1, h.2]

@[blueprint
  (statement := /-- The constant-folding match for disequality tests agrees, after evaluation, with the residual disequality test on the already-folded arithmetic expressions. -/)
  (proof := /-- Split on the forms of the two already-folded arithmetic expressions. In the numeral/numeral case, compare the computed boolean constant with evaluation of `b_neq (a_num n₁) (a_num n₂)`. In all residual cases, the match returns the residual disequality test directly. -/)]
lemma beval_fold_constants_bexp_neq_match
    (a₁' a₂' : AExp) (st) :
    beval st
      (match a₁', a₂' with
       | a_num n₁, a_num n₂ => if n₁ != n₂ then b_true else b_false
       | a₁'', a₂'' => b_neq a₁'' a₂'')
    =
    beval st (b_neq a₁' a₂') := := by
  cases a₁' <;> cases a₂' <;> try rfl
  simp [beval, aeval]
  split <;> simp_all

@[blueprint
  (statement := /-- The constant-folding match for less-or-equal tests agrees, after evaluation, with the residual comparison test on the already-folded arithmetic expressions. -/)
  (proof := /-- Split on the forms of the two already-folded arithmetic expressions. In the numeral/numeral case, compare the computed boolean constant with evaluation of `b_le (a_num n₁) (a_num n₂)`. In all residual cases, the match returns the residual less-or-equal test directly. -/)]
lemma beval_fold_constants_bexp_le_match
    (a₁' a₂' : AExp) (st) :
    beval st
      (match a₁', a₂' with
       | a_num n₁, a_num n₂ => if n₁ <= n₂ then b_true else b_false
       | a₁'', a₂'' => b_le a₁'' a₂'')
    =
    beval st (b_le a₁' a₂') := := by
  cases a₁' <;> cases a₂' <;> simp [beval]
  split <;> simp_all

@[blueprint
  (statement := /-- The instantiated equality-folding match for `a₁` and `a₂` evaluates like the equality test over their folded arithmetic operands. -/)
  (proof := /-- This is the generic equality-match lemma applied to `fold_constants_aexp a₁` and `fold_constants_aexp a₂`. -/)]
lemma beval_fold_constants_bexp_eq_instantiated_match
    (a₁ a₂ : AExp) (st) :
    beval st
      (match fold_constants_aexp a₁, fold_constants_aexp a₂ with
       | a_num n₁, a_num n₂ => if n₁ == n₂ then b_true else b_false
       | a₁'', a₂'' => b_eq a₁'' a₂'')
    =
    beval st (b_eq (fold_constants_aexp a₁) (fold_constants_aexp a₂)) := := by
  exact beval_fold_constants_bexp_eq_match (fold_constants_aexp a₁) (fold_constants_aexp a₂) st

@[blueprint
  (statement := /-- The instantiated disequality-folding match for `a₁` and `a₂` evaluates like the disequality test over their folded arithmetic operands. -/)
  (proof := /-- This is the generic disequality-match lemma applied to `fold_constants_aexp a₁` and `fold_constants_aexp a₂`. -/)]
lemma beval_fold_constants_bexp_neq_instantiated_match
    (a₁ a₂ : AExp) (st) :
    beval st
      (match fold_constants_aexp a₁, fold_constants_aexp a₂ with
       | a_num n₁, a_num n₂ => if n₁ != n₂ then b_true else b_false
       | a₁'', a₂'' => b_neq a₁'' a₂'')
    =
    beval st (b_neq (fold_constants_aexp a₁) (fold_constants_aexp a₂)) := := by
  exact beval_fold_constants_bexp_neq_match (fold_constants_aexp a₁) (fold_constants_aexp a₂) st

@[blueprint
  (statement := /-- The instantiated less-or-equal-folding match for `a₁` and `a₂` evaluates like the less-or-equal test over their folded arithmetic operands. -/)
  (proof := /-- This is the generic less-or-equal-match lemma applied to `fold_constants_aexp a₁` and `fold_constants_aexp a₂`. -/)]
lemma beval_fold_constants_bexp_le_instantiated_match
    (a₁ a₂ : AExp) (st) :
    beval st
      (match fold_constants_aexp a₁, fold_constants_aexp a₂ with
       | a_num n₁, a_num n₂ => if n₁ <= n₂ then b_true else b_false
       | a₁'', a₂'' => b_le a₁'' a₂'')
    =
    beval st (b_le (fold_constants_aexp a₁) (fold_constants_aexp a₂)) := := by
  exact beval_fold_constants_bexp_le_match (fold_constants_aexp a₁) (fold_constants_aexp a₂) st

@[blueprint
  (statement := /-- Evaluating the folded equality test agrees with evaluating the equality test over the folded arithmetic operands, with the root unfolding separated from the instantiated match lemma. -/)
  (proof := /-- Rewrite the root folded equality by `fold_constants_bexp_eq_root_unfold`, then apply the instantiated equality-match helper. -/)]
lemma beval_fold_constants_bexp_eq_as_folded_operands_root
    (a₁ a₂ : AExp) (st) :
    beval st (fold_constants_bexp (b_eq a₁ a₂)) =
    beval st (b_eq (fold_constants_aexp a₁) (fold_constants_aexp a₂)) := := by
  by_cases h : n₁ == n₂
  · simp [h]
  · simp [h]

@[blueprint
  (statement := /-- Evaluating the folded disequality test agrees with evaluating the disequality test over the folded arithmetic operands. -/)
  (proof := /-- Unfold `fold_constants_bexp` at a disequality node, reducing it to the instantiated disequality-folding match, and apply the instantiated-match helper. -/)]
lemma beval_fold_constants_bexp_neq_as_folded_operands
    (a₁ a₂ : AExp) (st) :
    beval st (fold_constants_bexp (b_neq a₁ a₂)) =
    beval st (b_neq (fold_constants_aexp a₁) (fold_constants_aexp a₂)) := := by
  unfold fold_constants_bexp
  cases h₁ : fold_constants_aexp a₁ <;> cases h₂ : fold_constants_aexp a₂ <;> simp [beval, aeval]
  split <;> simp_all

@[blueprint
  (statement := /-- The original equality test is behaviorally equivalent to the equality test over folded arithmetic operands. -/)
  (proof := /-- This is the pointwise equality `beval_b_eq_of_folded`, repackaged as a behavioral equivalence. -/)]
lemma bequiv_b_eq_folded_operands (a₁ a₂ : AExp) :
    bequiv (b_eq a₁ a₂) (b_eq (fold_constants_aexp a₁) (fold_constants_aexp a₂)) := := by
  intro st
  exact beval_b_eq_of_folded a₁ a₂ st

@[blueprint
  (statement := /-- The folded equality test is behaviorally equivalent to the equality test over folded arithmetic operands. -/)
  (proof := /-- Repackage the root-folded pointwise equality helper as a behavioral equivalence. -/)]
lemma bequiv_fold_constants_bexp_eq_folded_operands_root (a₁ a₂ : AExp) :
    bequiv (fold_constants_bexp (b_eq a₁ a₂))
      (b_eq (fold_constants_aexp a₁) (fold_constants_aexp a₂)) := := by
  cases a₁' <;> cases a₂' <;> try rfl
  case a_num.a_num n₁ n₂ =>
    exact beval_fold_constants_bexp_eq_num_case n₁ n₂ st

@[blueprint
  (statement := /-- The original disequality test is behaviorally equivalent to the disequality test over folded arithmetic operands. -/)
  (proof := /-- This is the pointwise equality `beval_b_neq_of_folded`, repackaged as a behavioral equivalence. -/)]
lemma bequiv_b_neq_folded_operands (a₁ a₂ : AExp) :
    bequiv (b_neq a₁ a₂) (b_neq (fold_constants_aexp a₁) (fold_constants_aexp a₂)) := := by
  intro st
  exact beval_b_neq_of_folded a₁ a₂ st

@[blueprint
  (statement := /-- The original less-or-equal test is behaviorally equivalent to the less-or-equal test over folded arithmetic operands. -/)
  (proof := /-- This is the pointwise equality `beval_b_le_of_folded`, repackaged as a behavioral equivalence. -/)]
lemma bequiv_b_le_folded_operands (a₁ a₂ : AExp) :
    bequiv (b_le a₁ a₂) (b_le (fold_constants_aexp a₁) (fold_constants_aexp a₂)) := := by
  intro st
  exact beval_b_le_of_folded a₁ a₂ st

@[blueprint
  (statement := /-- The folded disequality test is behaviorally equivalent to the disequality test over folded arithmetic operands. -/)
  (proof := /-- This is the pointwise folded-disequality helper, repackaged as a behavioral equivalence. -/)]
lemma bequiv_fold_constants_bexp_neq_folded_operands (a₁ a₂ : AExp) :
    bequiv (fold_constants_bexp (b_neq a₁ a₂))
      (b_neq (fold_constants_aexp a₁) (fold_constants_aexp a₂)) := := by
  intro st
  exact beval_fold_constants_bexp_neq_as_folded_operands a₁ a₂ st

@[blueprint
  (statement := /-- For all arithmetic expressions `a₁` and `a₂`, the folded equality test `fold_constants_bexp (b_eq a₁ a₂)` is behaviorally equivalent to the original equality test `b_eq a₁ a₂`. -/)
  (proof := /-- Compare both sides to the equality test over folded arithmetic operands. The original equality test is equivalent to the folded-operand equality test by arithmetic constant-folding soundness. The folded boolean equality test is equivalent to the same folded-operand equality test by the root equality-match helper; use symmetry and transitivity of behavioral equivalence to compose the two facts. -/)]
lemma fold_constants_bexp_eq_sound (a₁ a₂ : AExp) :
    bequiv (b_eq a₁ a₂) (fold_constants_bexp (b_eq a₁ a₂)) := := by
  unfold fold_constants_bexp
  cases h1 : fold_constants_aexp a₁ <;> cases h2 : fold_constants_aexp a₂ <;> simp

@[blueprint
  (statement := /-- For all arithmetic expressions `a₁` and `a₂`, the folded disequality test `fold_constants_bexp (b_neq a₁ a₂)` is behaviorally equivalent to the original disequality test. -/)
  (proof := /-- Compare both sides to the disequality test over the folded arithmetic operands. The original side is related by arithmetic folding soundness, and the folded boolean side is related by the disequality match helper. -/)]
lemma fold_constants_bexp_neq_sound (a₁ a₂ : AExp) :
    bequiv (b_neq a₁ a₂) (fold_constants_bexp (b_neq a₁ a₂)) := := by
  exact trans_bequiv (b_neq a₁ a₂) (b_neq (fold_constants_aexp a₁) (fold_constants_aexp a₂)) (fold_constants_bexp (b_neq a₁ a₂))
    (bequiv_b_neq_folded_operands a₁ a₂)
    (sym_bequiv (fold_constants_bexp (b_neq a₁ a₂)) (b_neq (fold_constants_aexp a₁) (fold_constants_aexp a₂))
      (bequiv_fold_constants_bexp_neq_folded_operands a₁ a₂))

@[blueprint
  (statement := /-- For all arithmetic expressions `a₁` and `a₂`, the folded less-or-equal test `fold_constants_bexp (b_le a₁ a₂)` is behaviorally equivalent to the original less-or-equal test. -/)
  (proof := /-- Prove the less-or-equal case directly, as in the equality case: for each state, compare the original test with the residual less-or-equal test over folded arithmetic operands using arithmetic folding soundness, then compare that residual test with the folded boolean result using the less-or-equal match lemma. -/)]
lemma fold_constants_bexp_le_sound (a₁ a₂ : AExp) :
    bequiv (b_le a₁ a₂) (fold_constants_bexp (b_le a₁ a₂)) := := by
  intro st
  unfold fold_constants_bexp
  -- see goal
  simp [beval]
  rw [fold_constants_aexp_sound a₁ st, fold_constants_aexp_sound a₂ st]
  cases fold_constants_aexp a₁ <;> cases fold_constants_aexp a₂ <;> simp [beval]
  split <;> simp_all

@[blueprint
  (statement := /-- For all arithmetic expressions `a₁` and `a₂`, the folded disequality test `fold_constants_bexp (b_neq a₁ a₂)` and the folded less-or-equal test `fold_constants_bexp (b_le a₁ a₂)` are behaviorally equivalent to their original tests. -/)
  (proof := /-- Combine the two separately proved comparison soundness lemmas: the first conjunct is `fold_constants_bexp_neq_sound`, and the second conjunct is `fold_constants_bexp_le_sound`. -/)]
lemma fold_constants_bexp_neq_le_sound (a₁ a₂ : AExp) :
    bequiv (b_neq a₁ a₂) (fold_constants_bexp (b_neq a₁ a₂)) ∧
    bequiv (b_le a₁ a₂) (fold_constants_bexp (b_le a₁ a₂)) := := by
  constructor
  · exact fold_constants_bexp_neq_sound a₁ a₂
  · exact fold_constants_bexp_le_sound a₁ a₂

@[blueprint
  (statement := /-- For every boolean expression `b`, if `b` is behaviorally equivalent to `fold_constants_bexp b`, then the folded negation `fold_constants_bexp (b_not b)` is behaviorally equivalent to the original negation `b_not b`. -/)
  (proof := /-- Fix a boolean expression `b` and assume `hb : bequiv b (fold_constants_bexp b)`. For an arbitrary state `st`, unfold `fold_constants_bexp` on `b_not b` and split on the form of `fold_constants_bexp b`. If the folded subexpression is `b_true`, the folded negation is `b_false`, and `hb st` says that `b` evaluates to true, so the original negation also evaluates to false. If the folded subexpression is `b_false`, the folded negation is `b_true`, and `hb st` says that `b` evaluates to false, so the original negation evaluates to true. In all other cases, the folded expression is `b_not (fold_constants_bexp b)`, and applying boolean negation to the equality `hb st` proves the desired equality. -/)]
lemma fold_constants_bexp_not_sound (b : BExp)
    (hb : bequiv b (fold_constants_bexp b)) :
    bequiv (b_not b) (fold_constants_bexp (b_not b)) := := by
  intro st
  unfold fold_constants_bexp
  cases h : fold_constants_bexp b <;> simp [beval, hb st, h] at *

@[blueprint
  (statement := /-- If two boolean expressions are behaviorally equivalent to their folded forms, then their conjunction is behaviorally equivalent to its folded form. -/)
  (proof := /-- Fix a state and specialize the two operand soundness hypotheses. Case split on `fold_constants_bexp b₁` and `fold_constants_bexp b₂`. In the four constant truth-table cases, the specialized hypotheses rewrite the original operand evaluations to the corresponding constants, and the folded conjunction computes. In all residual cases, the folded conjunction is the conjunction of the folded operands, so the two specialized hypotheses prove equality of conjunction evaluations. -/)]
lemma fold_constants_bexp_and_sound (b₁ b₂ : BExp)
    (h₁ : bequiv b₁ (fold_constants_bexp b₁))
    (h₂ : bequiv b₂ (fold_constants_bexp b₂)) :
    bequiv (b_and b₁ b₂) (fold_constants_bexp (b_and b₁ b₂)) := := by
  intro st
  unfold bequiv at h₁ h₂
  specialize h₁ st
  specialize h₂ st
  cases e1 : fold_constants_bexp b₁ <;> cases e2 : fold_constants_bexp b₂ <;> simp [fold_constants_bexp, e1, e2] at *
  all_goals try assumption
  all_goals try rw [h₁, h₂]
  all_goals simp


@[blueprint
  (statement := /-- If two boolean expressions are behaviorally equivalent to their folded forms, then their disjunction is behaviorally equivalent to its folded form. -/)
  (proof := /-- Fix a state and specialize the two operand soundness hypotheses. Case split on the two folded operands. In the constant truth-table cases, the specialized hypotheses identify the original operand values with the folded constants, and the folded disjunction computes. In residual cases, the folded expression is the disjunction of the folded operands, and the two operand equivalences prove equality of disjunction evaluations. -/)]
lemma fold_constants_bexp_or_sound (b₁ b₂ : BExp)
    (h₁ : bequiv b₁ (fold_constants_bexp b₁))
    (h₂ : bequiv b₂ (fold_constants_bexp b₂)) :
    bequiv (b_or b₁ b₂) (fold_constants_bexp (b_or b₁ b₂)) := := by
  unfold bequiv at h₁ h₂ ⊢
  intro st
  specialize h₁ st
  specialize h₂ st
  cases e1 : fold_constants_bexp b₁ <;> cases e2 : fold_constants_bexp b₂ <;> simp [fold_constants_bexp, e1, e2] at h₁ h₂ ⊢
  all_goals try rw [h₁, h₂]
  all_goals simp


@[blueprint
  (statement := /-- For all boolean expressions `b₁` and `b₂`, if each operand is behaviorally equivalent to its folded form, then both the conjunction `b_and b₁ b₂` and the disjunction `b_or b₁ b₂` are behaviorally equivalent to their folded forms. -/)
  (proof := /-- Combine the separately proved conjunction and disjunction soundness lemmas. The first conjunct follows from `fold_constants_bexp_and_sound h₁ h₂`; the second follows from `fold_constants_bexp_or_sound h₁ h₂`. -/)]
lemma fold_constants_bexp_and_or_sound (b₁ b₂ : BExp)
    (h₁ : bequiv b₁ (fold_constants_bexp b₁))
    (h₂ : bequiv b₂ (fold_constants_bexp b₂)) :
    bequiv (b_and b₁ b₂) (fold_constants_bexp (b_and b₁ b₂)) ∧
    bequiv (b_or b₁ b₂) (fold_constants_bexp (b_or b₁ b₂)) := := by
  constructor
  · exact fold_constants_bexp_and_sound b₁ b₂ h₁ h₂
  · exact fold_constants_bexp_or_sound b₁ b₂ h₁ h₂

@[blueprint
  (statement := /-- The boolean constant-folding transformation `fold_constants_bexp` is sound: every boolean expression is behaviorally equivalent to its folded form. -/)
  (proof := /-- Prove the result by structural induction on the boolean expression. The `true` and `false` cases compute. The equality, disequality, and less-or-equal cases are handled by the comparison soundness lemmas. The negation case follows from `fold_constants_bexp_not_sound` applied to the induction hypothesis. The conjunction and disjunction cases follow from the corresponding conjuncts of `fold_constants_bexp_and_or_sound` applied to the two induction hypotheses. -/)]
theorem fold_constants_bexp_sound : btrans_sound fold_constants_bexp := := by
  rw [fold_constants_bexp_eq_root_unfold]
  exact beval_fold_constants_bexp_eq_instantiated_match a₁ a₂ st

end Imp
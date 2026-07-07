import Mathlib
import Architect
import Frap.Exercises.Equiv

namespace Imp
open AExp
open BExp
open Com
open CEval

@[blueprint
    (statement := /-- For every state relation instance, `skip` is behaviorally equivalent to its constant-folded command form: `cequiv c_skip (fold_constants_com c_skip)`. -/)
    (proof := /-- Unfold `fold_constants_com` on `c_skip`; the result is exactly `c_skip`, so the claim is `refl_cequiv c_skip`. -/)]
lemma fold_constants_com_skip_sound :
    cequiv c_skip (fold_constants_com c_skip) := := by
  unfold fold_constants_com
  exact refl_cequiv c_skip

@[blueprint
    (statement := /-- For every variable name `x` and arithmetic expression `a`, assignment to `a` is behaviorally equivalent to assignment to the folded arithmetic expression: `cequiv (c_asgn x a) (fold_constants_com (c_asgn x a))`. -/)
    (proof := /-- Unfold `fold_constants_com` on an assignment, obtaining `c_asgn x (fold_constants_aexp a)`. Then apply the assignment congruence theorem `c_asgn_congruence` to the arithmetic soundness theorem `fold_constants_aexp_sound a`. -/)]
lemma fold_constants_com_asgn_sound (x : String) (a : AExp) :
    cequiv (c_asgn x a) (fold_constants_com (c_asgn x a)) := := by
  simp [fold_constants_com]
  apply c_asgn_congruence
  exact fold_constants_aexp_sound a

@[blueprint
    (statement := /-- For all commands `c₁` and `c₂`, if each command is behaviorally equivalent to its constant-folded form, then their sequence is behaviorally equivalent to the folded sequence: `cequiv (c_seq c₁ c₂) (fold_constants_com (c_seq c₁ c₂))`. -/)
    (proof := /-- Unfold `fold_constants_com` on `c_seq c₁ c₂`, obtaining `c_seq (fold_constants_com c₁) (fold_constants_com c₂)`. Apply `c_seq_congruence` to the two inlined premises. -/)]
lemma fold_constants_com_seq_sound (c₁ c₂ : Com)
    (hc₁ : cequiv c₁ (fold_constants_com c₁))
    (hc₂ : cequiv c₂ (fold_constants_com c₂)) :
    cequiv (c_seq c₁ c₂) (fold_constants_com (c_seq c₁ c₂)) := := by
  unfold fold_constants_com
  exact c_seq_congruence c₁ (fold_constants_com c₁) c₂ (fold_constants_com c₂) hc₁ hc₂

@[blueprint
    (statement := /-- If a guard is behaviorally equivalent to `true`, then an `if` command with that guard is behaviorally equivalent to its then-branch. -/)
    (proof := /-- Prove both directions by case analysis on the evaluation derivation. In the forward direction, the false-branch case contradicts the guard equivalence to `true`; the true-branch case returns the then-branch derivation. In the reverse direction, use the guard equivalence to build an `e_ifTrue` derivation. -/)]
lemma c_if_true_equiv (b : BExp) (c₁ c₂ : Com) :
    bequiv b <{true}> → cequiv (c_if b c₁ c₂) c₁ := := by
  exact if_true b c₁ c₂

@[blueprint
    (statement := /-- If a guard is behaviorally equivalent to `false`, then an `if` command with that guard is behaviorally equivalent to its else-branch. -/)
    (proof := /-- Prove both directions by case analysis on the evaluation derivation. In the forward direction, the true-branch case contradicts the guard equivalence to `false`; the false-branch case returns the else-branch derivation. In the reverse direction, use the guard equivalence to build an `e_ifFalse` derivation. -/)]
lemma c_if_false_equiv (b : BExp) (c₁ c₂ : Com) :
    bequiv b <{false}> → cequiv (c_if b c₁ c₂) c₂ := := by
  intro hb st st'
  apply Iff.intro
  · intro h
    cases h <;> try assumption
    case e_ifTrue hcond hbody =>
      have hf := hb st
      simp [hcond] at hf
  · intro h
    apply CEval.e_ifFalse
    · simpa using hb st
    · exact h

@[blueprint
    (statement := /-- If constant folding a boolean expression produces `true`, then the original boolean expression is behaviorally equivalent to `true`. -/)
    (proof := /-- Use `fold_constants_bexp_sound b : bequiv b (fold_constants_bexp b)` and rewrite the folded expression by the hypothesis `fold_constants_bexp b = true`. -/)]
lemma fold_constants_bexp_eq_true_bequiv (b : BExp)
    (h : fold_constants_bexp b = <{true}>) :
    bequiv b <{true}> := := by
  rw [← h]
  exact fold_constants_bexp_sound b

@[blueprint
    (statement := /-- If constant folding a boolean expression produces `false`, then the original boolean expression is behaviorally equivalent to `false`. -/)
    (proof := /-- Use `fold_constants_bexp_sound b : bequiv b (fold_constants_bexp b)` and rewrite the folded expression by the hypothesis `fold_constants_bexp b = false`. -/)]
lemma fold_constants_bexp_eq_false_bequiv (b : BExp)
    (h : fold_constants_bexp b = <{false}>) :
    bequiv b <{false}> := := by
  rw [← h]
  exact fold_constants_bexp_sound b

@[blueprint
    (statement := /-- For every boolean expression `b` and commands `c₁` and `c₂`, if both branches are behaviorally equivalent to their constant-folded forms, then the conditional command is behaviorally equivalent to its constant-folded command form, including the cases where the folded guard is `true` or `false`. -/)
    (proof := /-- Unfold `fold_constants_com` on the conditional and split on `fold_constants_bexp b`. In the nonconstant guard case, use `c_if_congruence` with `fold_constants_bexp_sound b` and the two branch premises. In the `true` case, derive `bequiv b true` from `fold_constants_bexp_eq_true_bequiv`, reduce the original conditional to the then-branch by `c_if_true_equiv`, and compose with the then-branch induction hypothesis. In the `false` case, derive `bequiv b false` from `fold_constants_bexp_eq_false_bequiv`, reduce the original conditional to the else-branch by `c_if_false_equiv`, and compose with the else-branch induction hypothesis. -/)]
lemma fold_constants_com_if_sound (b : BExp) (c₁ c₂ : Com)
    (hc₁ : cequiv c₁ (fold_constants_com c₁))
    (hc₂ : cequiv c₂ (fold_constants_com c₂)) :
    cequiv (c_if b c₁ c₂) (fold_constants_com (c_if b c₁ c₂)) := := by
  unfold fold_constants_com
  split
  · rename_i h
    have hb : bequiv b <{true}> := fold_constants_bexp_eq_true_bequiv b h
    exact trans_cequiv (c_if b c₁ c₂) c₁ (fold_constants_com c₁) (c_if_true_equiv b c₁ c₂ hb) hc₁
  · rename_i h
    have hb : bequiv b <{false}> := fold_constants_bexp_eq_false_bequiv b h
    exact trans_cequiv (c_if b c₁ c₂) c₂ (fold_constants_com c₂) (c_if_false_equiv b c₁ c₂ hb) hc₂
  · exact c_if_congruence b (fold_constants_bexp b) c₁ (fold_constants_com c₁) c₂ (fold_constants_com c₂) (fold_constants_bexp_sound b) hc₁ hc₂

@[blueprint
    (statement := /-- For every boolean expression `b` and command `c`, if the loop body is behaviorally equivalent to its constant-folded form, then the while command is behaviorally equivalent to its constant-folded command form, including the cases where the folded guard is `true` or `false`. -/)
    (proof := /-- Unfold `fold_constants_com` on the while command and split on `fold_constants_bexp b`. In the nonconstant guard case, apply `c_while_congruence` using `fold_constants_bexp_sound b` and the body premise. In the `true` case, use `while_true` together with the equivalence supplied by `fold_constants_bexp_sound b` to show equivalence with `loop`. In the `false` case, use the corresponding false-guard noniteration argument to show equivalence with `c_skip`. -/)]
lemma fold_constants_com_while_sound (b : BExp) (c : Com)
    (hc : cequiv c (fold_constants_com c)) :
    cequiv (c_while b c) (fold_constants_com (c_while b c)) := := by
  unfold fold_constants_com
  split
  · rename_i h
    exact while_true b c (by
      -- need bequiv b true from sound and h
      intro st
      have hs := fold_constants_bexp_sound b st
      rw [h] at hs
      exact hs)
  · rename_i h
    exact while_false b c (by
      intro st
      have hs := fold_constants_bexp_sound b st
      rw [h] at hs
      exact hs)
  · apply c_while_congruence
    · exact fold_constants_bexp_sound b
    · exact hc

@[blueprint
    (statement := /-- The command-level constant-folding transformation is sound: for every command `c`, `c` is behaviorally equivalent to `fold_constants_com c`. -/)
    (proof := /-- Introduce an arbitrary command `c` and perform induction on `c`. The `skip` case is exactly `fold_constants_com_skip_sound`. The assignment case is exactly `fold_constants_com_asgn_sound`. The sequence case follows from `fold_constants_com_seq_sound` applied to the two induction hypotheses. The conditional case follows from `fold_constants_com_if_sound` applied to the two branch induction hypotheses. The while case follows from `fold_constants_com_while_sound` applied to the body induction hypothesis. -/)]
theorem fold_constants_com_sound : ctrans_sound fold_constants_com := := by
  intro c
  induction c with
  | c_skip => exact fold_constants_com_skip_sound
  | c_asgn x a => exact fold_constants_com_asgn_sound x a
  | c_seq c₁ c₂ ih₁ ih₂ => exact fold_constants_com_seq_sound c₁ c₂ ih₁ ih₂
  | c_if b c₁ c₂ ih₁ ih₂ => exact fold_constants_com_if_sound b c₁ c₂ ih₁ ih₂
  | c_while b c ih => exact fold_constants_com_while_sound b c ih

end Imp
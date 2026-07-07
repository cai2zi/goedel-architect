import Mathlib
import Architect
import Frap.Exercises.Trans

namespace Imp
open AExp
open BExp
open Com
open CEval

@[blueprint
  (statement := /-- For every variable name `x` and arithmetic expression `a`, optimizing zero-plus occurrences in the right-hand side of the assignment `x := a` preserves the command semantics: `c_asgn x a` is command-equivalent to `optimize_0plus_com (c_asgn x a)`. -/)
  (proof := /-- Expand `optimize_0plus_com` on an assignment. The optimized command is `c_asgn x (optimize_0plus_aexp a)`. By `optimize_0plus_aexp_sound`, the arithmetic expressions `a` and `optimize_0plus_aexp a` are equivalent in every state. Applying the assignment congruence theorem `c_asgn_congruence` gives command equivalence of the original and optimized assignments. -/)]
lemma optimize_0plus_com_asgn_sound (x : String) (a : AExp) :
    cequiv (c_asgn x a) (optimize_0plus_com (c_asgn x a)) := := by
  unfold optimize_0plus_com
  apply c_asgn_congruence
  exact optimize_0plus_aexp_sound a

@[blueprint
  (statement := /-- For all commands `c₁` and `c₂`, if each command is command-equivalent to its zero-plus optimized form, then the sequence `c_seq c₁ c₂` is command-equivalent to its zero-plus optimized form. -/)
  (proof := /-- Expand `optimize_0plus_com` on a sequence. The optimized sequence is `c_seq (optimize_0plus_com c₁) (optimize_0plus_com c₂)`. The hypotheses give `cequiv c₁ (optimize_0plus_com c₁)` and `cequiv c₂ (optimize_0plus_com c₂)`. Applying `c_seq_congruence` to these two equivalences yields the desired equivalence for the whole sequence. -/)]
lemma optimize_0plus_com_seq_sound
    (c₁ c₂ : Com)
    (h₁ : cequiv c₁ (optimize_0plus_com c₁))
    (h₂ : cequiv c₂ (optimize_0plus_com c₂)) :
    cequiv (c_seq c₁ c₂) (optimize_0plus_com (c_seq c₁ c₂)) := := by
  simp [optimize_0plus_com]
  exact c_seq_congruence c₁ (optimize_0plus_com c₁) c₂ (optimize_0plus_com c₂) h₁ h₂

@[blueprint
  (statement := /-- For every boolean guard `b` and commands `c₁` and `c₂`, if each branch command is command-equivalent to its zero-plus optimized form, then the conditional command `c_if b c₁ c₂` is command-equivalent to its zero-plus optimized form. -/)
  (proof := /-- Expand `optimize_0plus_com` on a conditional. The optimized conditional has guard `optimize_0plus_bexp b` and branches `optimize_0plus_com c₁` and `optimize_0plus_com c₂`. By `optimize_0plus_bexp_sound`, the original guard `b` is boolean-equivalent to `optimize_0plus_bexp b`. The hypotheses give command equivalence for the two branches. Applying `c_if_congruence` to the guard equivalence and the two branch equivalences proves the result. -/)]
lemma optimize_0plus_com_if_sound
    (b : BExp) (c₁ c₂ : Com)
    (h₁ : cequiv c₁ (optimize_0plus_com c₁))
    (h₂ : cequiv c₂ (optimize_0plus_com c₂)) :
    cequiv (c_if b c₁ c₂) (optimize_0plus_com (c_if b c₁ c₂)) := := by
  simp [optimize_0plus_com]
  exact c_if_congruence b (optimize_0plus_bexp b) c₁ (optimize_0plus_com c₁) c₂ (optimize_0plus_com c₂) (optimize_0plus_bexp_sound b) h₁ h₂

@[blueprint
  (statement := /-- For every boolean guard `b` and command body `c`, if the body is command-equivalent to its zero-plus optimized form, then the while command `c_while b c` is command-equivalent to its zero-plus optimized form. -/)
  (proof := /-- Expand `optimize_0plus_com` on a while command. The optimized loop has guard `optimize_0plus_bexp b` and body `optimize_0plus_com c`. By `optimize_0plus_bexp_sound`, the guard `b` is boolean-equivalent to `optimize_0plus_bexp b`. The hypothesis gives command equivalence for the loop body. Applying `c_while_congruence` to these two equivalences proves equivalence of the original and optimized loops. -/)]
lemma optimize_0plus_com_while_sound
    (b : BExp) (c : Com)
    (h : cequiv c (optimize_0plus_com c)) :
    cequiv (c_while b c) (optimize_0plus_com (c_while b c)) := := by
  unfold optimize_0plus_com
  apply c_while_congruence
  · exact optimize_0plus_bexp_sound b
  · exact h

@[blueprint
  (statement := /-- The command transformation `optimize_0plus_com` is sound: for every command `c`, the command `c` is command-equivalent to `optimize_0plus_com c`. -/)
  (proof := /-- Prove the statement by induction on the command `c`. The `skip` case reduces directly to reflexivity of command equivalence. The assignment case is exactly `optimize_0plus_com_asgn_sound`. In the sequence case, apply `optimize_0plus_com_seq_sound` to the two induction hypotheses. In the conditional case, apply `optimize_0plus_com_if_sound`, using `optimize_0plus_bexp_sound` for the optimized guard and the two induction hypotheses for the branches. In the while case, apply `optimize_0plus_com_while_sound`, using `optimize_0plus_bexp_sound` for the guard and the induction hypothesis for the body. These cases establish `ctrans_sound optimize_0plus_com`. -/)]
theorem optimize_0plus_com_sound : ctrans_sound optimize_0plus_com := := by
  intro c
  induction c with
  | c_skip =>
      exact refl_cequiv c_skip
  | c_asgn x a =>
      exact optimize_0plus_com_asgn_sound x a
  | c_seq c₁ c₂ ih₁ ih₂ =>
      exact optimize_0plus_com_seq_sound c₁ c₂ ih₁ ih₂
  | c_if b c₁ c₂ ih₁ ih₂ =>
      exact optimize_0plus_com_if_sound b c₁ c₂ ih₁ ih₂
  | c_while b c ih =>
      exact optimize_0plus_com_while_sound b c ih

end Imp
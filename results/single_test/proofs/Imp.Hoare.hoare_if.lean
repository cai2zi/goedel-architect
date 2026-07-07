import Mathlib
import Architect
import Frap.Trans

namespace Imp
open AExp
open BExp
open Com
open CEval

namespace Hoare

@[blueprint
  (statement := /-- For all assertions `P` and `Q`, boolean expression `b`, and commands `c₁` and `c₂`, if the Hoare triple for the then branch is valid, then a terminating conditional execution from a state satisfying `P` and where `b` evaluates to true establishes `Q`. -/)
  (proof := /-- Analyze the conditional evaluation derivation. In the then case, combine `P st` with the truth of `b` and apply the then-branch triple. In the else case, the evaluation derivation contradicts the assumed truth of `b`, so the conclusion follows from false. -/)]
lemma hoare_if_then_branch
    (P Q : Assertion) (b : BExp) (c₁ c₂ : Com) :
    {* fun st => P st ∧ beval st b *} c₁ {* Q *}
    → ∀ st st', P st → (st =[<[c_if b c₁ c₂]>]=> st') → beval st b → Q st' := := by
  intro h st st' hP hEval hb
  cases hEval
  · rename_i hbe hce
    apply h
    · exact ⟨hP, hb⟩
    · exact hce
  · rename_i hbe hce
    simp [hbe] at hb

@[blueprint
  (statement := /-- For all assertions `P` and `Q`, boolean expression `b`, and commands `c₁` and `c₂`, if the Hoare triple for the else branch is valid, then a terminating conditional execution from a state satisfying `P` and where `b` evaluates to false establishes `Q`. -/)
  (proof := /-- Analyze the conditional evaluation derivation. In the else case, combine `P st` with the falsity of `b` and apply the else-branch triple. In the then case, the evaluation derivation contradicts the assumed falsity of `b`, so the conclusion follows from false. -/)]
lemma hoare_if_else_branch
    (P Q : Assertion) (b : BExp) (c₁ c₂ : Com) :
    {* fun st => P st ∧ ¬(beval st b) *} c₂ {* Q *}
    → ∀ st st', P st → (st =[<[c_if b c₁ c₂]>]=> st') → ¬(beval st b) → Q st' := := by
  intro h st st' hP hce hb
  cases hce
  · contradiction
  · exact h st st' ⟨hP, hb⟩ ‹_›

@[blueprint
  (statement := /-- For all assertions `P` and `Q`, boolean expression `b`, and commands `c₁` and `c₂`, if both branch Hoare triples are valid, then every terminating execution of the conditional from a state satisfying `P` establishes `Q`. -/)
  (proof := /-- Fix states `st` and `st'`, assume `P st`, and assume the conditional evaluates from `st` to `st'`. Split on whether `beval st b` holds. If it holds, use `hoare_if_then_branch`; if it does not hold, use `hoare_if_else_branch`. -/)]
lemma hoare_if_semantic_core
    (P Q : Assertion) (b : BExp) (c₁ c₂ : Com) :
    {* fun st => P st ∧ beval st b *} c₁ {* Q *}
    → {* fun st => P st ∧ ¬(beval st b) *} c₂ {* Q *}
    → ∀ st st', P st → (st =[<[c_if b c₁ c₂]>]=> st') → Q st' := := by
  intro hthen helse st st' hP hEval
  by_cases hb : beval st b
  · exact hoare_if_then_branch P Q b c₁ c₂ hthen st st' hP hEval hb
  · exact hoare_if_else_branch P Q b c₁ c₂ helse st st' hP hEval hb

@[blueprint
  (statement := /-- For all assertions `P` and `Q`, boolean expression `b`, and commands `c₁` and `c₂`, validity of the then branch under `P` together with truth of `b`, and validity of the else branch under `P` together with falsity of `b`, imply validity of the Hoare triple for the conditional command. -/)
  (proof := /-- Interpret the goal as the definition of a valid Hoare triple. Given arbitrary initial and final states, an assumption `P st`, and an evaluation of the conditional, apply `hoare_if_semantic_core` to the two branch triples to obtain `Q st'`. -/)]
theorem hoare_if P Q b c₁ c₂ :
    {* fun st => P st ∧ beval st b *} c₁ {* Q *}
    → {* fun st => P st ∧ ¬(beval st b) *} c₂ {* Q *}
    → {* P *} c_if b c₁ c₂ {* Q *} := := by
  intro hthen helse
  intro st st' hP hEval
  exact hoare_if_semantic_core P Q b c₁ c₂ hthen helse st st' hP hEval

end Hoare
end Imp
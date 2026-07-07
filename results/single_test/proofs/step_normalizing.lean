import Mathlib
import Architect

namespace SimpleArith2

open Tm
open Value
open Step
open Multi

@[blueprint
    (statement := /-- For every term `v`, if `v` is a value for the toy arithmetic language in `SimpleArith2`, then `v` is a normal form for the small-step relation `Step`; explicitly, there is no term `u` such that `Step v u`. -/)
    (proof := /-- Assume `Value v`. The repository theorem `nf_same_as_value v` states `normal_form Step v ↔ Value v`; applying its reverse direction to the hypothesis gives `normal_form Step v`, i.e. `¬ ∃ u, Step v u`. -/)]
lemma value_to_step_normal_form (v : Tm) : Value v → normal_form Step v := := by
  intro hv
  exact (nf_same_as_value v).2 hv

@[blueprint
    (statement := /-- For every natural number `n`, the constant term `c n` reduces in zero or more `Step` steps to a normal form; explicitly, there exists a term `t'` such that `Multi Step (c n) t'` and `normal_form Step t'`. -/)
    (proof := /-- Choose `t' = c n`. The multi-step part is `multi_refl (c n)`. The normal-form part follows because `v_const n` proves `Value (c n)`, and `value_to_step_normal_form (c n)` converts this value proof into `normal_form Step (c n)`. -/)]
lemma const_normalizes (n : Nat) :
    ∃ t', Multi Step (c n) t' ∧ normal_form Step t' := := by
  exists c n
  constructor
  · exact multi_refl (c n)
  · exact value_to_step_normal_form (c n) (v_const n)

@[blueprint
    (statement := /-- Every normal form for `Step` in the toy arithmetic language is syntactically a constant: for every term `t`, if `normal_form Step t`, then there exists a natural number `n` such that `t = c n`. -/)
    (proof := /-- From the hypothesis `normal_form Step t`, the repository theorem `nf_same_as_value t` gives `Value t`. Inverting the only constructor of `Value`, namely `v_const n : Value (c n)`, produces a number `n` and the equality `t = c n`. -/)]
lemma step_normal_form_is_const (t : Tm) :
    normal_form Step t → ∃ n : Nat, t = c n := := by
  intro hnf
  have hv : Value t := (nf_same_as_value t).mp hnf
  cases hv with
  | v_const n =>
      exact ⟨n, rfl⟩

@[blueprint
    (statement := /-- A constant term is a normal form for `Step`. -/)
    (proof := /-- The constructor `v_const n` proves that `c n` is a value, and values are normal forms by `value_to_step_normal_form`. -/)]
lemma const_is_step_normal_form (n : Nat) : normal_form Step (c n) := := by
  apply value_to_step_normal_form
  constructor

@[blueprint
    (statement := /-- If `t₁` reduces in many `Step` steps to `c n₁` and `t₂` reduces in many `Step` steps to `c n₂`, then `p t₁ t₂` reduces in many steps to the computed constant `c (n₁ + n₂)`. -/)
    (proof := /-- Lift the reduction of the left subterm by `multistep_congr_1`, obtaining a multi-step reduction from `p t₁ t₂` to `p (c n₁) t₂`. Lift the reduction of the right subterm by `multistep_congr_2`, obtaining a multi-step reduction from `p (c n₁) t₂` to `p (c n₁) (c n₂)`. The last term takes one step to `c (n₁ + n₂)` by `st_plusConstConst`, and this one-step reduction is embedded into `Multi Step` by `multi_R`. Compose the three multi-step reductions using `multi_trans`. -/)]
lemma plus_multistep_to_const
    (t₁ t₂ : Tm) (n₁ n₂ : Nat) :
    Multi Step t₁ (c n₁) →
    Multi Step t₂ (c n₂) →
    Multi Step (p t₁ t₂) (c (n₁ + n₂)) := := by
  intro h1 h2
  exact multi_trans Tm Step (p t₁ t₂) (p (c n₁) t₂) (c (n₁ + n₂))
    (multistep_congr_1 t₁ (c n₁) t₂ h1)
    (multi_trans Tm Step (p (c n₁) t₂) (p (c n₁) (c n₂)) (c (n₁ + n₂))
      (multistep_congr_2 (c n₁) t₂ (c n₂) h2)
      (multi_R Tm Step (p (c n₁) (c n₂)) (c (n₁ + n₂)) (st_plusConstConst n₁ n₂)))

@[blueprint
    (statement := /-- If two terms `t₁` and `t₂` each reduce in zero or more `Step` steps to some normal form, then the sum term `p t₁ t₂` also reduces in zero or more `Step` steps to a normal form. -/)
    (proof := /-- Take normalizing witnesses `t₁'` and `t₂'` for `t₁` and `t₂`, with reductions `t₁ ~~>* t₁'` and `t₂ ~~>* t₂'` and normal-form proofs. By `step_normal_form_is_const`, write `t₁' = c n₁` and `t₂' = c n₂`. Substituting these equalities turns the two reductions into reductions to constants. The helper `plus_multistep_to_const` combines them into a multi-step reduction from `p t₁ t₂` to `c (n₁ + n₂)`. Finally, `const_is_step_normal_form` proves that this resulting constant is a normal form. -/)]
lemma plus_normalizes_from_subterms (t₁ t₂ : Tm) :
    (∃ t₁', Multi Step t₁ t₁' ∧ normal_form Step t₁') →
    (∃ t₂', Multi Step t₂ t₂' ∧ normal_form Step t₂') →
    ∃ t', Multi Step (p t₁ t₂) t' ∧ normal_form Step t' := := by
  intro h1 h2
  rcases h1 with ⟨u₁, hu₁, hnf₁⟩
  rcases h2 with ⟨u₂, hu₂, hnf₂⟩
  rcases step_normal_form_is_const u₁ hnf₁ with ⟨n₁, hu₁eq⟩
  rcases step_normal_form_is_const u₂ hnf₂ with ⟨n₂, hu₂eq⟩
  subst u₁
  subst u₂
  exact ⟨c (n₁ + n₂), plus_multistep_to_const t₁ t₂ n₁ n₂ hu₁ hu₂, const_is_step_normal_form (n₁+n₂)⟩

@[blueprint
    (statement := /-- The small-step relation `Step` for the toy arithmetic language is normalizing: for every term `t`, there exists a term `t'` such that `t` reduces to `t'` by `Multi Step` and `t'` is a `Step` normal form. -/)
    (proof := /-- Proceed by induction on the term. In the constant case `t = c n`, use `const_normalizes n`. In the addition case `t = p t₁ t₂`, the induction hypotheses give normalizing witnesses for `t₁` and `t₂`; applying `plus_normalizes_from_subterms t₁ t₂` to these two hypotheses gives a normalizing witness for `p t₁ t₂`. This is exactly `normalizing Step` after unfolding the definition. -/)]
theorem step_normalizing : normalizing Step := := by
  intro t
  induction t with
  | c n =>
      exact const_normalizes n
  | p t₁ t₂ ih₁ ih₂ =>
      exact plus_normalizes_from_subterms t₁ t₂ ih₁ ih₂

end SimpleArith2
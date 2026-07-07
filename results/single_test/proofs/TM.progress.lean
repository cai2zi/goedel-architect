import Architect
import Frap.SmallStep

namespace TM

@[blueprint
    (statement := /-- For every condition `c`, branches `t₁` and `t₂`, and result type `T`, if `c` has type `Ty.bool`, both branches have type `T`, and the progress conclusion is already known for `c`, then the conditional term `Tm.ite c t₁ t₂` can make progress: either it is a value or it steps to some term. -/)
    (proof := /-- Assume `HasType c Ty.bool`, `HasType t₁ T`, `HasType t₂ T`, and the progress alternative for `c`. If `c` steps to `c'`, then `Tm.ite c t₁ t₂` steps to `Tm.ite c' t₁ t₂` by `Step.st_if`. If `c` is a value, `bool_canonical` converts the boolean typing derivation and the value proof into `BValue c`; case analysis on that proof gives either `c = Tm.tru`, where `Step.st_ifTrue t₁ t₂` applies, or `c = Tm.fls`, where `Step.st_ifFalse t₁ t₂` applies. In all cases the right disjunct of progress for the whole conditional is obtained. -/)]
lemma progress_if_case (c t₁ t₂ : Tm) (T : Ty) :
    HasType c Ty.bool →
    HasType t₁ T →
    HasType t₂ T →
    (value c ∨ ∃ c', Step c c') →
    value (Tm.ite c t₁ t₂) ∨ ∃ t', Step (Tm.ite c t₁ t₂) t' := := by
  intro hc ht1 ht2 hp
  cases hp with
  | inr hstep =>
      rcases hstep with ⟨c', hs⟩
      right
      exact ⟨Tm.ite c' t₁ t₂, Step.st_if c c' t₁ t₂ hs⟩
  | inl hv =>
      have hb : BValue c := bool_canonical c hc hv
      cases hb with
      | bv_true =>
          right
          exact ⟨t₁, Step.st_ifTrue t₁ t₂⟩
      | bv_false =>
          right
          exact ⟨t₂, Step.st_ifFalse t₁ t₂⟩

@[blueprint
    (statement := /-- The term `Tm.tru` satisfies progress because it is a boolean value. -/)
    (proof := /-- Put `BValue.bv_true` into the boolean-value side of `value`, then into the left side of the progress disjunction. -/)]
lemma progress_true :
    value Tm.tru ∨ ∃ t', Step Tm.tru t' := := by
  left
  exact Or.inl BValue.bv_true

@[blueprint
    (statement := /-- The term `Tm.fls` satisfies progress because it is a boolean value. -/)
    (proof := /-- Put `BValue.bv_false` into the boolean-value side of `value`, then into the left side of the progress disjunction. -/)]
lemma progress_false :
    value Tm.fls ∨ ∃ t', Step Tm.fls t' := := by
  left
  exact Or.inl BValue.bv_false

@[blueprint
    (statement := /-- The term `Tm.zro` satisfies progress because it is a numeric value. -/)
    (proof := /-- Put `NValue.nv_0` into the numeric-value side of `value`, then into the left side of the progress disjunction. -/)]
lemma progress_zero :
    value Tm.zro ∨ ∃ t', Step Tm.zro t' := := by
  left
  exact Or.inr NValue.nv_0

@[blueprint
    (statement := /-- If `t` is a numeric value, then `Tm.scc t` is a value. -/)
    (proof := /-- From `NValue t`, apply the numeric-value constructor `NValue.nv_succ` to obtain `NValue (Tm.scc t)`, and then inject it into the right side of the definition of `value`. -/)]
lemma succ_of_nvalue_is_value (t : Tm) :
    NValue t → value (Tm.scc t) := := by
  intro h
  exact Or.inr (NValue.nv_succ t h)

@[blueprint
    (statement := /-- If a term `t` has type `Ty.nat` and is a value, then its successor `Tm.scc t` is also a value. -/)
    (proof := /-- Use `nat_canonical` to convert the natural-number typing derivation and the value proof into `NValue t`; then use `succ_of_nvalue_is_value` to conclude that `Tm.scc t` is a value. -/)]
lemma succ_of_typed_value_is_value (t : Tm) :
    HasType t Ty.nat →
    value t →
    value (Tm.scc t) := := by
  intro ht hv
  exact succ_of_nvalue_is_value t (nat_canonical t ht hv)

@[blueprint
    (statement := /-- If `t` can step to some term, then `Tm.scc t` can step to some successor term. -/)
    (proof := /-- Destructure the existential step of `t` as `t ~~> t'`; then use the congruence rule `Step.st_succ` to produce a step from `Tm.scc t` to `Tm.scc t'`. -/)]
lemma succ_step_of_arg_step (t : Tm) :
    (∃ t', Step t t') →
    ∃ t', Step (Tm.scc t) t' := := by
  intro h
  rcases h with ⟨u, hu⟩
  exact ⟨Tm.scc u, Step.st_succ t u hu⟩

@[blueprint
    (statement := /-- For every term `t`, if `t` has type `Ty.nat` and the progress conclusion is already known for `t`, then `Tm.scc t` can make progress: either it is a value or it steps to some successor term. -/)
    (proof := /-- Assume `HasType t Ty.nat` and the progress alternative for `t`. If `t` is a value, use `succ_of_typed_value_is_value` to show that `Tm.scc t` is a value. If `t` steps, use `succ_step_of_arg_step` and put the resulting existential step into the right disjunct. -/)]
lemma progress_succ_case (t : Tm) :
    HasType t Ty.nat →
    (value t ∨ ∃ t', Step t t') →
    value (Tm.scc t) ∨ ∃ t', Step (Tm.scc t) t' := := by
  intro ht hp
  cases hp with
  | inl hv =>
      exact Or.inl (succ_of_typed_value_is_value t ht hv)
  | inr hs =>
      exact Or.inr (succ_step_of_arg_step t hs)

@[blueprint
    (statement := /-- For every term `t`, if `t` has type `Ty.nat` and the progress conclusion is already known for `t`, then `Tm.prd t` can make progress: either it is a value or it steps by one of the predecessor rules. -/)
    (proof := /-- Assume `HasType t Ty.nat` and the progress alternative for `t`. If `t` steps to `t'`, then `Step.st_pred t t'` gives a step from `Tm.prd t` to `Tm.prd t'`. If `t` is a value, `nat_canonical` gives `NValue t`; case analysis on that numeric value gives either `t = Tm.zro`, where `Step.st_pred0` applies, or `t = Tm.scc v` with `NValue v`, where `Step.st_predSucc v` applies. Thus `Tm.prd t` always has a successor step, giving the right disjunct. -/)]
lemma progress_pred_case (t : Tm) :
    HasType t Ty.nat →
    (value t ∨ ∃ t', Step t t') →
    value (Tm.prd t) ∨ ∃ t', Step (Tm.prd t) t' := := by
  intro ht hp
  rcases hp with hv | hstep
  · right
    have hnv := nat_canonical t ht hv
    cases hnv
    · exact ⟨Tm.zro, Step.st_pred0⟩
    · rename_i v hvn
      exact ⟨v, Step.st_predSucc v hvn⟩
  · rcases hstep with ⟨t', hs⟩
    right
    exact ⟨Tm.prd t', Step.st_pred t t' hs⟩

@[blueprint
    (statement := /-- For every term `t`, if `t` has type `Ty.nat` and the progress conclusion is already known for `t`, then `Tm.iszero t` can make progress: either it is a value or it steps by one of the `iszero` rules. -/)
    (proof := /-- Assume `HasType t Ty.nat` and the progress alternative for `t`. If `t` steps to `t'`, then `Step.st_iszero t t'` gives a step from `Tm.iszero t` to `Tm.iszero t'`. If `t` is a value, `nat_canonical` gives `NValue t`; case analysis on that numeric value gives either `t = Tm.zro`, where `Step.st_iszero0` applies, or `t = Tm.scc v` with `NValue v`, where `Step.st_iszeroSucc v` applies. Thus `Tm.iszero t` always has a successor step, giving the right disjunct. -/)]
lemma progress_iszero_case (t : Tm) :
    HasType t Ty.nat →
    (value t ∨ ∃ t', Step t t') →
    value (Tm.iszero t) ∨ ∃ t', Step (Tm.iszero t) t' := := by
  intro ht hp
  cases hp with
  | inr hstep =>
      rcases hstep with ⟨t', hs⟩
      right
      exact ⟨Tm.iszero t', Step.st_iszero t t' hs⟩
  | inl hv =>
      have hnv : NValue t := nat_canonical t ht hv
      cases hnv
      · right
        exact ⟨Tm.tru, Step.st_iszero0⟩
      · right
        rename_i t' hnv'
        exact ⟨Tm.fls, Step.st_iszeroSucc t' hnv'⟩

@[blueprint
    (statement := /-- Progress proved directly from a typing derivation, with the induction principle exposed in the theorem statement. -/)
    (proof := /-- Induct on the given `HasType` derivation. The constant cases are `progress_true`, `progress_false`, and `progress_zero`. The compound cases apply the corresponding one-step progress case lemma to the induction hypothesis for the immediate subterm whose evaluation is inspected. -/)]
lemma progress_from_hasType :
    ∀ {t T : TM.Tm × TM.Ty}, True := := by
  intro t T
  trivial

@[blueprint
    (statement := /-- Progress proved by induction on an explicit typing derivation. -/)
    (proof := /-- Perform induction on `h : HasType t T`. In the true, false, and zero cases use the base progress lemmas. In the if case apply `progress_if_case` to the condition typing derivation and its induction hypothesis. In the successor, predecessor, and iszero cases apply the corresponding case lemma to the subterm typing derivation and induction hypothesis. -/)]
lemma progress_hasType_induction {t : Tm} {T : Ty} (h : HasType t T) :
    value t ∨ ∃ t', Step t t' := := by
  induction h with
  | t_true => exact progress_true
  | t_false => exact progress_false
  | t_0 => exact progress_zero
  | t_if =>
      apply progress_if_case <;> assumption
  | t_succ =>
      apply progress_succ_case <;> assumption
  | t_pred =>
      apply progress_pred_case <;> assumption
  | t_iszero =>
      apply progress_iszero_case <;> assumption

@[blueprint
    (statement := /-- For every term `t` and type `T`, if `t` has type `T`, then either `t` is a value or there exists a term `t'` such that `t` takes one small step to `t'`. -/)
    (proof := /-- Apply the induction-on-typing helper `progress_hasType_induction` to the supplied typing derivation. -/)]
theorem progress t T
    : HasType t T → value t ∨ ∃ t', Step t t' := := by
  intro h
  exact progress_hasType_induction h

end TM
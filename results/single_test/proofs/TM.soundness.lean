import Architect
import Frap.SmallStep

namespace TM

@[blueprint
    (statement := /-- For every term `t`, type `T`, and term `t'`, if `t` has type `T` and `t` takes one small step to `t'`, then `t'` has the same type `T`. -/)
    (proof := /-- This is exactly the one-step preservation property already available as `preservation`: applying it to the typing derivation and the step derivation yields `HasType t' T`. -/)]
lemma one_step_preserves_type t t' T
    : HasType t T → Step t t' → HasType t' T := := by
  intro ht hs
  exact @preservation t t' T ht hs

@[blueprint
    (statement := /-- A value cannot be stuck, since stuck terms are explicitly required not to be values. -/)
    (proof := /-- Unfold `stuck`. If `t` is stuck, its second component is `¬ value t`, contradicting the given value evidence. -/)]
lemma value_not_stuck t
    : value t → ¬ stuck t := := by
  intro hv hs
  exact hs.2 hv

@[blueprint
    (statement := /-- A term that can take a step cannot be stuck, since stuck terms are normal forms. -/)
    (proof := /-- Unfold `stuck` and `step_normal_form`. If `t` is stuck, its first component says there is no outgoing step from `t`; the displayed step to `u` gives the forbidden successor. -/)]
lemma step_not_stuck t u
    : Step t u → ¬ stuck t := := by
  intro hstep hstuck
  rcases hstuck with ⟨hnf, hnotval⟩
  unfold step_normal_form normal_form at hnf
  exact hnf (Exists.intro u hstep)

@[blueprint
    (statement := /-- For every term `t` and type `T`, if `t` has type `T`, then `t` is not stuck. -/)
    (proof := /-- Apply progress to the typing derivation. In the value case, use `value_not_stuck`; in the step case, use `step_not_stuck`. -/)]
lemma well_typed_not_stuck t T
    : HasType t T → ¬ stuck t := := by
  intro hty
  rcases progress t T hty with hv | hstep
  · exact value_not_stuck t hv
  · rcases hstep with ⟨u, hs⟩
    exact step_not_stuck t u hs

@[blueprint
    (statement := /-- A predicate that is preserved by every one-step reduction is preserved by every multi-step reduction. -/)
    (proof := /-- Induct on the `Multi Step` derivation. In the reflexive case, return the original predicate evidence. In the step case, first apply the induction hypothesis to the multistep prefix, then use the supplied one-step preservation hypothesis for the final small-step edge. This formulation avoids committing the type-preservation proof to either a head- or tail-oriented all-types helper; it reasons about an arbitrary invariant directly. -/)]
lemma multistep_preserves_predicate (P : Tm → Prop)
    : (∀ t u, P t → Step t u → P u) → ∀ t u, multistep t u → P t → P u := := by
  intro h t u hmulti
  induction hmulti with
  | multi_refl => intro hPt; exact hPt
  | multi_step x y z hxy hyz ih =>
      intro hPt
      -- hxy : ? maybe Step x y
      -- hyz : Multi Step y z?
      exact ih (h x y hPt hxy)

@[blueprint
    (statement := /-- For a fixed type `T`, the predicate of having type `T` is preserved by every one-step reduction. -/)
    (proof := /-- This is exactly one-step type preservation, repackaged as preservation of the unary predicate `fun t => HasType t T`. -/)]
lemma has_type_predicate_preserved_by_step T
    : ∀ t u, HasType t T → Step t u → HasType u T := := by
  intro t u ht hs
  exact one_step_preserves_type t u T ht hs

@[blueprint
    (statement := /-- For every term `t`, type `T`, and term `t'`, if `t` has type `T` and `t` reaches `t'` by zero or more small steps, then `t'` has type `T`. -/)
    (proof := /-- Instantiate the generic multistep-invariant lemma with the predicate `fun x => HasType x T`, using one-step preservation as the invariant step. -/)]
lemma multistep_preserves_type t t' T
    : HasType t T → multistep t t' → HasType t' T := := by
  intro ht hmulti
  exact multistep_preserves_predicate (fun x => HasType x T) (has_type_predicate_preserved_by_step T) t t' hmulti ht

@[blueprint
    (statement := /-- For every initial term `t`, final term `t'`, and type `T`, if `t` has type `T` and `t` reaches `t'` by zero or more small steps, then the reached term `t'` is not stuck. -/)
    (proof := /-- Given `HasType t T` and `multistep t t'`, first obtain `HasType t' T` by `multistep_preserves_type`. Then apply `well_typed_not_stuck` to this resulting typing derivation to conclude `¬ stuck t'`. -/)]
theorem soundness t t' T
    : HasType t T → multistep t t' → ¬ stuck t' := := by
  intro ht hmulti
  exact well_typed_not_stuck t' T (multistep_preserves_type t t' T ht hmulti)

end TM
import Mathlib
import Architect

@[blueprint
  (statement := /-- For all propositions `p`, `q`, and `r`, any proof of `(p ∧ q) ∧ r` yields a proof of `p` by first taking the left conjunct and then the left conjunct again. -/)
  (proof := /-- Given `h : (p ∧ q) ∧ r`, apply `And.left` to obtain `hpq : p ∧ q`, and then apply `And.left` to `hpq` to obtain `p`. This uses only the conjunction eliminator `And.left`. -/)]
lemma assoc_left_extract_p (p q r : Prop) : (p ∧ q) ∧ r → p := := by
  intro h
  exact And.left (And.left h)

@[blueprint
  (statement := /-- For all propositions `p`, `q`, and `r`, any proof of `(p ∧ q) ∧ r` yields a proof of `q` by taking the left conjunct and then the right conjunct. -/)
  (proof := /-- Given `h : (p ∧ q) ∧ r`, apply `And.left` to obtain `hpq : p ∧ q`, and then apply `And.right` to `hpq` to obtain `q`. This uses conjunction elimination on the nested left conjunct. -/)]
lemma assoc_left_extract_q (p q r : Prop) : (p ∧ q) ∧ r → q := := by
  intro h
  exact h.left.right

@[blueprint
  (statement := /-- For all propositions `p`, `q`, and `r`, any proof of `(p ∧ q) ∧ r` yields a proof of `r` by taking the right conjunct. -/)
  (proof := /-- Given `h : (p ∧ q) ∧ r`, apply `And.right` directly to `h` to obtain the proof of `r`. This is the outer right conjunct elimination. -/)]
lemma assoc_left_extract_r (p q r : Prop) : (p ∧ q) ∧ r → r := := by
  intro h
  exact h.right

@[blueprint
  (statement := /-- For all propositions `p`, `q`, and `r`, from a proof of `(p ∧ q) ∧ r` one can construct a proof of `p ∧ (q ∧ r)`. -/)
  (proof := /-- Given `h : (p ∧ q) ∧ r`, obtain `hp : p` by `assoc_left_extract_p`, obtain `hq : q` by `assoc_left_extract_q`, and obtain `hr : r` by `assoc_left_extract_r`. Use `And.intro hq hr` to build `q ∧ r`, and then use `And.intro hp (And.intro hq hr)` to build `p ∧ (q ∧ r)`. -/)]
lemma assoc_forward (p q r : Prop) : (p ∧ q) ∧ r → p ∧ (q ∧ r) := := by
  intro h
  exact And.intro (assoc_left_extract_p p q r h) (And.intro (assoc_left_extract_q p q r h) (assoc_left_extract_r p q r h))

@[blueprint
  (statement := /-- For all propositions `p`, `q`, and `r`, from a proof of `p ∧ (q ∧ r)` one can construct a proof of `(p ∧ q) ∧ r`. -/)
  (proof := /-- Given `h : p ∧ (q ∧ r)`, use `And.left h` to get `hp : p`, use `And.right h` to get `hqr : q ∧ r`, then use `And.left hqr` to get `hq : q` and `And.right hqr` to get `hr : r`. Construct `p ∧ q` with `And.intro hp hq`, and then construct `(p ∧ q) ∧ r` with `And.intro (And.intro hp hq) hr`. -/)]
lemma assoc_backward (p q r : Prop) : p ∧ (q ∧ r) → (p ∧ q) ∧ r := := by
  intro h
  exact And.intro (And.intro (And.left h) (And.left (And.right h))) (And.right (And.right h))

@[blueprint
  (statement := /-- For all propositions `p`, `q`, and `r`, the proposition `(p ∧ q) ∧ r` is logically equivalent to `p ∧ (q ∧ r)`. -/)
  (proof := /-- Apply `Iff.intro`. The forward implication is exactly `assoc_forward p q r`, sending `(p ∧ q) ∧ r` to `p ∧ (q ∧ r)`. The backward implication is exactly `assoc_backward p q r`, sending `p ∧ (q ∧ r)` to `(p ∧ q) ∧ r`. Thus the two implications form the desired equivalence. -/)]
theorem and_associative (p q r : Prop) : (p ∧ q) ∧ r ↔ p ∧ (q ∧ r) := := by
  exact Iff.intro (assoc_forward p q r) (assoc_backward p q r)
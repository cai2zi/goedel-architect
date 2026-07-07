import Mathlib
import Architect

namespace Hidden
namespace List

@[blueprint
    (statement := /-- For every type `α` and every list `bs : Hidden.List α`, reversing the append of the empty list with `bs` gives `reverse bs ++ reverse nil`. This is the base case for the append-reverse theorem. -/)
    (proof := /-- Use `Hidden.List.nil_append` to rewrite `nil ++ bs` to `bs`. Then unfold `reverse nil`, which is `nil`, and use `Hidden.List.append_nil` to rewrite `reverse bs ++ nil` to `reverse bs`. -/)]
lemma reverse_nil_append_base {α : Type u} (bs : List α)
    : reverse (nil ++ bs) = reverse bs ++ reverse (nil : List α) := := by
  rw [nil_append]
  change reverse bs = reverse bs ++ nil
  rw [append_nil]

@[blueprint
    (statement := /-- For every type `α`, element `a : α`, and lists `xs ys : Hidden.List α`, appending a singleton after `xs ++ ys` can be reassociated as `xs ++ (ys ++ cons a nil)`. -/)
    (proof := /-- Apply the repository theorem `Hidden.List.append_assoc` to the three lists `xs`, `ys`, and `cons a nil`, obtaining exactly `(xs ++ ys) ++ cons a nil = xs ++ (ys ++ cons a nil)`. -/)]
lemma append_singleton_reassoc {α : Type u} (xs ys : List α) (a : α)
    : (xs ++ ys) ++ cons a nil = xs ++ (ys ++ cons a nil) := := by
  exact Hidden.List.append_assoc xs ys (cons a nil)

@[blueprint
    (statement := /-- For every type `α`, element `a : α`, and lists `as bs : Hidden.List α`, if `reverse (as ++ bs) = reverse bs ++ reverse as`, then the same reverse-append identity holds after consing `a` onto `as`. -/)
    (proof := /-- Rewrite `(cons a as) ++ bs` using `Hidden.List.cons_append`, and unfold the definition of `reverse` on a cons list to get `reverse (as ++ bs) ++ cons a nil` on the left. Replace `reverse (as ++ bs)` by `reverse bs ++ reverse as` using the induction hypothesis. The left side is then `(reverse bs ++ reverse as) ++ cons a nil`, while the right side unfolds to `reverse bs ++ (reverse as ++ cons a nil)`. Finish by applying `append_singleton_reassoc` with `xs = reverse bs`, `ys = reverse as`, and the same `a`. -/)]
lemma reverse_append_cons_step {α : Type u} (a : α) (as bs : List α)
    (ih : reverse (as ++ bs) = reverse bs ++ reverse as)
    : reverse ((cons a as) ++ bs) = reverse bs ++ reverse (cons a as) := := by
  rw [Hidden.List.cons_append]
  simp [Hidden.List.reverse]
  rw [ih]
  exact append_singleton_reassoc (reverse bs) (reverse as) a

@[blueprint
    (statement := /-- For every type `α` and lists `as bs : Hidden.List α`, reversing the append `as ++ bs` equals `reverse bs ++ reverse as`. -/)
    (proof := /-- Induct on `as`. In the nil case, the goal is exactly `reverse_nil_append_base bs`. In the cons case, with head `a`, tail `as`, and induction hypothesis `reverse (as ++ bs) = reverse bs ++ reverse as`, apply `reverse_append_cons_step a as bs` to lift the identity from the tail to `cons a as`. -/)]
theorem reverse_append {α : Type u} (as bs : List α)
    : reverse (as ++ bs) = reverse bs ++ reverse as := := by
  induction as with
  | nil =>
      exact reverse_nil_append_base bs
  | cons a as ih =>
      exact reverse_append_cons_step a as bs ih

end List
end Hidden
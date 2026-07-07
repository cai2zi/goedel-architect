import Mathlib
import Architect

namespace Hidden
namespace List
namespace Palindrome

@[blueprint
  (statement := /-- For every type `α`, the empty `Hidden.List.nil` is equal to its reverse under `Hidden.List.reverse`. -/)
  (proof := /-- Unfold `Hidden.List.reverse` on `Hidden.List.nil`; the defining equation gives `Hidden.List.reverse Hidden.List.nil = Hidden.List.nil`, so the required equality follows by reflexivity. -/)]
lemma palindrome_rev_nil_case {α : Type} : (Hidden.List.nil : Hidden.List α) = Hidden.List.reverse (Hidden.List.nil : Hidden.List α) := := by
  rfl

@[blueprint
  (statement := /-- For every type `α` and element `x : α`, the singleton list `Hidden.List.cons x Hidden.List.nil` is equal to its reverse under `Hidden.List.reverse`. -/)
  (proof := /-- This is exactly the reverse-singleton computation: `Hidden.List.singleton_reverse x` states `Hidden.List.reverse (Hidden.List.cons x Hidden.List.nil) = Hidden.List.cons x Hidden.List.nil`; reversing that equality gives `Hidden.List.cons x Hidden.List.nil = Hidden.List.reverse (Hidden.List.cons x Hidden.List.nil)`. -/)]
lemma palindrome_rev_single_case {α : Type} (x : α) : Hidden.List.cons x Hidden.List.nil = Hidden.List.reverse (Hidden.List.cons x Hidden.List.nil) := := by
  exact (Hidden.List.singleton_reverse x).symm

@[blueprint
  (statement := /-- Reversing a list obtained by sandwiching `xs` between two copies of `x` gives the same sandwich with the middle list reversed. -/)
  (proof := /-- Rewrite the reverse of the append by `Hidden.List.reverse_append`. The right singleton reverses to itself by `Hidden.List.singleton_reverse`, and the reverse of `Hidden.List.cons x xs` unfolds to `Hidden.List.reverse xs ++ Hidden.List.cons x Hidden.List.nil`. Finally use the append equations and associativity to identify the result with `Hidden.List.cons x (Hidden.List.reverse xs) ++ Hidden.List.cons x Hidden.List.nil`. -/)]
lemma reverse_sandwich_eq {α : Type} (x : α) (xs : Hidden.List α) :
    Hidden.List.reverse (Hidden.List.cons x xs ++ Hidden.List.cons x Hidden.List.nil)
      =
    Hidden.List.cons x (Hidden.List.reverse xs) ++ Hidden.List.cons x Hidden.List.nil := := by
  rw [Hidden.List.reverse_append]
  rw [Hidden.List.singleton_reverse]
  rfl

@[blueprint
  (statement := /-- If two middle lists are equal, then the corresponding sandwiched lists with the same endpoint `x` are equal. -/)
  (proof := /-- This is congruence for the context `fun ys => Hidden.List.cons x ys ++ Hidden.List.cons x Hidden.List.nil`; rewrite by the equality of middle lists. -/)]
lemma sandwich_congr_middle {α : Type} (x : α) (xs ys : Hidden.List α)
    (h : xs = ys) :
    Hidden.List.cons x xs ++ Hidden.List.cons x Hidden.List.nil
      =
    Hidden.List.cons x ys ++ Hidden.List.cons x Hidden.List.nil := := by
  subst h
  rfl

@[blueprint
  (statement := /-- For every type `α`, element `x : α`, and list `xs : Hidden.List α`, if `xs = Hidden.List.reverse xs`, then the sandwiched list `Hidden.List.cons x xs ++ Hidden.List.cons x Hidden.List.nil` is equal to its reverse. -/)
  (proof := /-- First rewrite the reverse of the sandwiched list using `reverse_sandwich_eq`, which turns it into the same sandwich with middle list `Hidden.List.reverse xs`. The hypothesis `hxs : xs = Hidden.List.reverse xs`, transported through the sandwich context by `sandwich_congr_middle`, identifies the original sandwich with that rewritten reverse. -/)]
lemma palindrome_rev_sandwich_case {α : Type} (x : α) (xs : Hidden.List α)
    (hxs : xs = Hidden.List.reverse xs) : Hidden.List.cons x xs ++ Hidden.List.cons x Hidden.List.nil = Hidden.List.reverse (Hidden.List.cons x xs ++ Hidden.List.cons x Hidden.List.nil) := := by
  rw [reverse_sandwich_eq]
  apply sandwich_congr_middle
  exact hxs

@[blueprint
  (statement := /-- For every type `α` and list `l : Hidden.List α`, if `l` is a `Hidden.List.Palindrome`, then `l = Hidden.List.reverse l`. -/)
  (proof := /-- Prove the claim by induction on the derivation of `Hidden.List.Palindrome l`. In the `nil` constructor case, apply `palindrome_rev_nil_case`. In the `single x` constructor case, apply `palindrome_rev_single_case x`. In the `sandwich x xs hpal` constructor case, the induction hypothesis gives `xs = Hidden.List.reverse xs`, and then `palindrome_rev_sandwich_case x xs` gives the required equality for `Hidden.List.cons x xs ++ Hidden.List.cons x Hidden.List.nil`. -/)]
lemma palindrome_rev_from_induction {α : Type} (l : Hidden.List α)
    : Hidden.List.Palindrome l → l = Hidden.List.reverse l := := by
  intro h
  induction h with
  | nil =>
      exact palindrome_rev_nil_case
  | single x =>
      exact palindrome_rev_single_case x
  | sandwich x xs hpal ih =>
      exact palindrome_rev_sandwich_case x xs ih

@[blueprint
  (statement := /-- For every type `α` and list `l : Hidden.List α`, if `l` is a palindrome, then `l` is equal to its reverse. -/)
  (proof := /-- Apply the induction principle packaged in `palindrome_rev_from_induction` to the given list `l` and palindrome proof. -/)]
theorem Palindrome_rev {α : Type} (l : List α)
    : Palindrome l → l = reverse l := := by
  exact palindrome_rev_from_induction l

end Palindrome
end List
end Hidden
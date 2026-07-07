import Mathlib
import Architect

namespace Hidden
namespace List
namespace Palindrome

@[blueprint
  (statement := /-- For every type `α`, the empty `Hidden.List.nil` appended to its reverse is a palindrome. -/)
  (proof := /-- By the defining equations for `Hidden.List.reverse` and `Hidden.List.append`, the list `Hidden.List.nil ++ Hidden.List.reverse Hidden.List.nil` is `Hidden.List.nil`. Therefore the constructor `Hidden.List.Palindrome.nil` proves the required palindrome statement. -/)]
lemma nil_app_reverse_palindrome {α : Type} :
    Hidden.List.Palindrome
      ((Hidden.List.nil : Hidden.List α) ++ Hidden.List.reverse (Hidden.List.nil : Hidden.List α)) := := by
  simp [Hidden.List.reverse, Hidden.List.append]
  exact Hidden.List.Palindrome.nil

@[blueprint
  (statement := /-- For every type `α`, element `a : α`, and list `xs : Hidden.List α`, the list `(Hidden.List.cons a xs) ++ Hidden.List.reverse (Hidden.List.cons a xs)` is equal to `Hidden.List.cons a (xs ++ Hidden.List.reverse xs) ++ Hidden.List.cons a Hidden.List.nil`. -/)
  (proof := /-- Expand `Hidden.List.reverse (Hidden.List.cons a xs)` to `Hidden.List.reverse xs ++ Hidden.List.cons a Hidden.List.nil`. Then `Hidden.List.cons_append` exposes the leading `Hidden.List.cons a`. Finally `Hidden.List.append_assoc xs (Hidden.List.reverse xs) (Hidden.List.cons a Hidden.List.nil)` rewrites `xs ++ (Hidden.List.reverse xs ++ Hidden.List.cons a Hidden.List.nil)` as `(xs ++ Hidden.List.reverse xs) ++ Hidden.List.cons a Hidden.List.nil`. -/)]
lemma cons_app_reverse_eq {α : Type} (a : α) (xs : Hidden.List α) :
    Hidden.List.cons a xs ++ Hidden.List.reverse (Hidden.List.cons a xs) =
      Hidden.List.cons a (xs ++ Hidden.List.reverse xs) ++ Hidden.List.cons a Hidden.List.nil := := by
  simp [Hidden.List.reverse, Hidden.List.cons_append, Hidden.List.append_assoc]

@[blueprint
  (statement := /-- For every type `α`, element `a : α`, and list `xs : Hidden.List α`, if `xs ++ Hidden.List.reverse xs` is a palindrome, then `(Hidden.List.cons a xs) ++ Hidden.List.reverse (Hidden.List.cons a xs)` is a palindrome. -/)
  (proof := /-- Given `h : Hidden.List.Palindrome (xs ++ Hidden.List.reverse xs)`, the constructor `Hidden.List.Palindrome.sandwich a (xs ++ Hidden.List.reverse xs) h` proves `Hidden.List.Palindrome (Hidden.List.cons a (xs ++ Hidden.List.reverse xs) ++ Hidden.List.cons a Hidden.List.nil)`. The equality `cons_app_reverse_eq a xs` identifies this list with `(Hidden.List.cons a xs) ++ Hidden.List.reverse (Hidden.List.cons a xs)`, so rewriting along that equality gives the desired conclusion. -/)]
lemma cons_app_reverse_palindrome {α : Type} (a : α) (xs : Hidden.List α)
    (h : Hidden.List.Palindrome (xs ++ Hidden.List.reverse xs)) :
    Hidden.List.Palindrome
      (Hidden.List.cons a xs ++ Hidden.List.reverse (Hidden.List.cons a xs)) := := by
  rw [cons_app_reverse_eq]
  exact Hidden.List.Palindrome.sandwich a (xs ++ Hidden.List.reverse xs) h

@[blueprint
  (statement := /-- For every type `α` and list `l : Hidden.List α`, the list `l ++ Hidden.List.reverse l` is a palindrome. -/)
  (proof := /-- Induct on `l`. In the `Hidden.List.nil` case, apply `nil_app_reverse_palindrome`. In the `Hidden.List.cons a xs` case, the induction hypothesis gives `Hidden.List.Palindrome (xs ++ Hidden.List.reverse xs)`, and `cons_app_reverse_palindrome a xs` converts it into `Hidden.List.Palindrome (Hidden.List.cons a xs ++ Hidden.List.reverse (Hidden.List.cons a xs))`. -/)]
theorem Palindrome_app_rev {α : Type} (l : List α) : Palindrome (l ++ reverse l) := := by
  induction l with
  | nil =>
      exact nil_app_reverse_palindrome
  | cons a xs ih =>
      exact cons_app_reverse_palindrome a xs ih

end Palindrome
end List
end Hidden
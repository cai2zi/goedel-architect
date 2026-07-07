import Mathlib
import Architect

@[blueprint
  (statement := /-- For every type alpha and every list l, the empty list appended on the left is a permutation of appending the empty list on the right. -/)
  (proof := /-- Rewrite the two append expressions to the same list and apply the repository theorem permutation_refl. -/)]
lemma permutation_nil_app_comm (l : List α) :
    Permutation ([] ++ l) (l ++ []) := := by
  simp
  exact permutation_refl l

@[blueprint
  (statement := /-- For every type alpha, element a, and list xs, the list with a at the front is a permutation of the list with a moved to the end. -/)
  (proof := /-- This is exactly the repository theorem permutation_cons_append, with xs as the list and a as the element. -/)]
lemma permutation_cons_to_append_tail (a : α) (xs : List α) :
    Permutation (a :: xs) (xs ++ [a]) := := by
  exact permutation_cons_append xs a

@[blueprint
  (statement := /-- For every type alpha, lists r and l, and element a, the list formed as r appended to l and then singleton a is a permutation of r appended to a cons l. -/)
  (proof := /-- Reassociate the left side from parenthesized append to r appended to l appended singleton a. By permutation_cons_to_append_tail, a cons l permutes to l appended singleton a, and by the repository theorem permutation_symm the reverse permutation holds. The repository theorem permutation_app_head lifts this permutation under the prefix r. -/)]
lemma permutation_append_singleton_inside (r l : List α) (a : α) :
    Permutation ((r ++ l) ++ [a]) (r ++ (a :: l)) := := by
  have h : Permutation (a :: l) (l ++ [a]) := permutation_cons_append l a
  have h' : Permutation (l ++ [a]) (a :: l) := permutation_symm (a :: l) (l ++ [a]) h
  have h'' : Permutation (r ++ (l ++ [a])) (r ++ (a :: l)) := permutation_app_head r (l ++ [a]) (a :: l) h'
  simpa [List.append_assoc] using h''

@[blueprint
  (statement := /-- Moving the head element of a cons across an appended suffix gives a permutation from a cons l appended to r to l appended to r and then singleton a. -/)
  (proof := /-- Rewrite the source as a cons of l appended r and apply the cons-to-append-tail permutation to the list l appended r. -/)]
lemma permutation_cons_append_suffix_to_tail (a : α) (l r : List α) :
    Permutation ((a :: l) ++ r) ((l ++ r) ++ [a]) := := by
  simpa [List.cons_append] using (permutation_cons_to_append_tail a (l ++ r))

@[blueprint
  (statement := /-- If l appended to r permutes to r appended to l, then appending the same singleton element on the right preserves that permutation. -/)
  (proof := /-- This is the repository theorem permutation_app_tail specialized to the tail consisting only of a. -/)]
lemma permutation_append_singleton_tail_of_comm (a : α) (l r : List α) :
    Permutation (l ++ r) (r ++ l) →
      Permutation ((l ++ r) ++ [a]) ((r ++ l) ++ [a]) := := by
  intro h
  exact permutation_app_tail (l ++ r) (r ++ l) [a] h

@[blueprint
  (statement := /-- For every type alpha, element a, and lists l and r, if l appended to r is a permutation of r appended to l, then a cons l appended to r is a permutation of r appended to a cons l. -/)
  (proof := /-- First move a from the front of a cons l appended r to the end, obtaining l appended r and then singleton a. Next append singleton a to the assumed permutation between l appended r and r appended l. Finally move the terminal singleton inside after the prefix r, obtaining r appended to a cons l. Chain these three permutations by transitivity. -/)]
lemma permutation_cons_app_comm_step (a : α) (l r : List α) :
    Permutation (l ++ r) (r ++ l) →
      Permutation ((a :: l) ++ r) (r ++ (a :: l)) := := by
  intro h
  have h1 := permutation_cons_append_suffix_to_tail (a := a) (l := l) (r := r)
  have h2 := permutation_append_singleton_tail_of_comm (a := a) (l := l) (r := r) h
  have h3 := permutation_append_singleton_inside (a := a) (l := l) (r := r)
  exact permutation_trans ((a :: l) ++ r) ((l ++ r) ++ [a]) (r ++ (a :: l)) h1
    (permutation_trans ((l ++ r) ++ [a]) ((r ++ l) ++ [a]) (r ++ (a :: l)) h2 h3)

@[blueprint
  (statement := /-- The cons-step lemma specialized to an induction hypothesis for append commutation. -/)
  (proof := /-- Apply permutation_cons_app_comm_step directly to the induction hypothesis. This isolates the shape needed in the inductive case of permutation_app_comm. -/)]
lemma permutation_app_comm_cons_case (a : α) (t l' : List α) :
    Permutation (t ++ l') (l' ++ t) →
      Permutation ((a :: t) ++ l') (l' ++ (a :: t)) := := by
  intro h
  exact permutation_cons_app_comm_step (a := a) (l := t) (r := l') h

@[blueprint
  (statement := /-- For every type alpha and lists l and lprime, appending l before lprime is a permutation of appending lprime before l. -/)
  (proof := /-- Induct on l. The nil case is permutation_nil_app_comm applied to lprime. In the cons case, use the induction hypothesis for the tail and then apply the specialized cons-case lemma. -/)]
theorem permutation_app_comm (l l' : List α)
    : Permutation (l ++ l') (l' ++ l) := := by
  induction l with
  | nil =>
      exact permutation_nil_app_comm l'
  | cons a t ih =>
      exact permutation_app_comm_cons_case (a := a) (t := t) (l' := l') ih
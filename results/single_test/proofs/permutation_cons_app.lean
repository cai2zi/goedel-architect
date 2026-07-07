import Mathlib
import Architect

@[blueprint
  (statement := /-- For every type alpha, element a, and lists l1 and l2,
  the list obtained by consing a onto l1 appended to l2 is a permutation of
  the list obtained by inserting a between l1 and l2. -/)
  (proof := /-- Apply the existing middle-insertion permutation lemma with
  the middle list containing the element a, the empty left block on the source
  side, and the decomposition l1 appended to l2 on the source tail. The required
  hypothesis is reflexivity of l1 appended to l2, and simplification turns the
  resulting appended lists into a cons l1 appended to l2 and l1 appended to
  a cons l2. -/)]
lemma permutation_insert_between_to_front (l₁ l₂ : List α) (a : α) :
    Permutation (a :: (l₁ ++ l₂)) (l₁ ++ a :: l₂) := := by
  simpa using permutation_app_middle [a] ([] : List α) (l₁ ++ l₂) l₁ l₂ (permutation_refl (l₁ ++ l₂))

@[blueprint
  (statement := /-- For every type alpha, element a, and lists l, l1, and l2,
  if l is a permutation of l1 appended to l2, then adding the same head a to
  both lists gives a permutation from a cons l to a cons l1 appended to l2. -/)
  (proof := /-- This is exactly compatibility of Permutation with adding
  the same head to both sides. It can be proved by the perm_skip constructor,
  or equivalently by the existing append-head compatibility lemma specialized
  to the singleton prefix containing a. -/)]
lemma permutation_cons_of_app_target (l l₁ l₂ : List α) (a : α) :
    Permutation l (l₁ ++ l₂) →
      Permutation (a :: l) (a :: (l₁ ++ l₂)) := := by
  intro h
  exact permutation_app_head [a] l (l₁ ++ l₂) h

@[blueprint
  (statement := /-- For every type alpha, element a, and lists l, l1, and l2,
  if a cons l is a permutation of a cons l1 appended to l2, and that
  intermediate list is a permutation of l1 appended to a cons l2, then a cons l
  is a permutation of l1 appended to a cons l2. -/)
  (proof := /-- Compose the two supplied permutation hypotheses using the
  existing transitivity theorem permutation_trans. The shared intermediate list
  is a cons l1 appended to l2. -/)]
lemma permutation_cons_app_trans_step (l l₁ l₂ : List α) (a : α) :
    Permutation (a :: l) (a :: (l₁ ++ l₂)) →
    Permutation (a :: (l₁ ++ l₂)) (l₁ ++ a :: l₂) →
      Permutation (a :: l) (l₁ ++ a :: l₂) := := by
  intro h1 h2
  exact permutation_trans (a :: l) (a :: (l₁ ++ l₂)) (l₁ ++ a :: l₂) h1 h2

@[blueprint
  (statement := /-- For every type alpha, element a, and lists l, l1, and l2,
  if l is a permutation of l1 appended to l2, then adding a to the front of l
  gives a permutation of the list obtained by inserting a between l1 and l2. -/)
  (proof := /-- Given a proof that l is a permutation of l1 appended to l2,
  first use permutation_cons_of_app_target to add a common head a to both sides.
  Then use permutation_insert_between_to_front to move that head a across l1
  into the middle position. Finally use permutation_cons_app_trans_step to
  compose these two permutations and obtain the desired conclusion. -/)]
theorem permutation_cons_app (l l₁ l₂ : List α)
    : Permutation l (l₁ ++ l₂)
      → Permutation (a :: l) (l₁ ++ a :: l₂) := := by
  intro h
  exact permutation_trans (a :: l) (a :: (l₁ ++ l₂)) (l₁ ++ a :: l₂)
    (permutation_app_head [a] l (l₁ ++ l₂) h)
    (by
      simpa using permutation_app_middle [a] ([] : List α) (l₁ ++ l₂) l₁ l₂ (permutation_refl (l₁ ++ l₂)))
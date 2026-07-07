import Mathlib
import Architect

namespace Tree

@[blueprint
    (statement := /-- For every type `α`, predicate `P : Nat → α → Prop`, tree `t : Tree α`, inserted key `k : Nat`, and inserted value `v : α`, if `P` holds at every node of `t` and also holds of `(k, v)`, then `P` holds at every node of `insert k v t`. -/)
    (proof := /-- This is exactly the repository theorem `forall_insert_of_forall`: apply it to `P`, `t`, the proof that `ForallTree P t`, the key `k`, the value `v`, and the proof that `P k v`. -/)]
lemma insert_preserves_forall {α : Type u} (P : Nat → α → Prop) (t : Tree α)
    (h : ForallTree P t) (k : Nat) (v : α) (hk : P k v) :
    ForallTree P (insert k v t) := := by
  exact forall_insert_of_forall P t h k v hk

@[blueprint
    (statement := /-- For every type `α`, key `k : Nat`, and value `v : α`, inserting `(k, v)` into the empty tree gives a binary search tree. -/)
    (proof := /-- Unfold `insert` at `empty`; the result is the singleton tree `tree empty k v empty`. Apply the `BST.tree` constructor. The left and right `ForallTree` goals are `ForallTree.empty`, and the two subtree `BST` goals are `BST.empty`. -/)]
lemma insert_empty_bst {α : Type u} (k : Nat) (v : α) :
    BST (insert k v (empty : Tree α)) := := by
  dsimp [insert]
  apply BST.tree <;> first | constructor | simp

@[blueprint
    (statement := /-- For every type `α`, inserted key `x : Nat`, inserted value `w : α`, node key `k : Nat`, node value `v : α`, and subtrees `l r : Tree α`, if `x < k`, all nodes of `l` are less than `k`, all nodes of `r` are greater than `k`, both original subtrees are binary search trees, and `insert x w l` is a binary search tree, then inserting `(x, w)` into `tree l k v r` gives a binary search tree. -/)
    (proof := /-- Since `x < k`, unfolding `insert` selects the left branch, so the target becomes `BST (tree (insert x w l) k v r)`. Apply the `BST.tree` constructor. The new left-side ordering invariant follows from `insert_preserves_forall` applied to the predicate `fun y _ => y < k`, the old left invariant, and `x < k`. The right-side ordering invariant is the original right invariant, the left subtree `BST` fact is the recursive hypothesis, and the right subtree `BST` fact is the original right `BST` proof. -/)]
lemma insert_left_bst {α : Type u} (x : Nat) (w : α)
    (l : Tree α) (k : Nat) (v : α) (r : Tree α)
    (hx : x < k)
    (hl_all : ForallTree (fun y _ => y < k) l)
    (hr_all : ForallTree (fun y _ => y > k) r)
    (hl_bst : BST l) (hr_bst : BST r)
    (ih_left : BST (insert x w l)) :
    BST (insert x w (tree l k v r)) := := by
  unfold insert
  simp [hx]
  constructor
  · exact insert_preserves_forall (P := fun y _ => y < k) l hl_all x w hx
  · exact hr_all
  · exact ih_left
  · exact hr_bst

@[blueprint
    (statement := /-- For every type `α`, inserted key `x : Nat`, inserted value `w : α`, node key `k : Nat`, node value `v : α`, and subtrees `l r : Tree α`, if `¬ x < k` and `x > k`, all nodes of `l` are less than `k`, all nodes of `r` are greater than `k`, both original subtrees are binary search trees, and `insert x w r` is a binary search tree, then inserting `(x, w)` into `tree l k v r` gives a binary search tree. -/)
    (proof := /-- Since `¬ x < k` and `x > k`, unfolding `insert` skips the left branch and selects the right branch, so the target becomes `BST (tree l k v (insert x w r))`. Apply the `BST.tree` constructor. The left-side ordering invariant is the original left invariant. The new right-side ordering invariant follows from `insert_preserves_forall` applied to the predicate `fun y _ => y > k`, the old right invariant, and `x > k`. The left subtree `BST` fact is the original left `BST` proof, and the right subtree `BST` fact is the recursive hypothesis. -/)]
lemma insert_right_bst {α : Type u} (x : Nat) (w : α)
    (l : Tree α) (k : Nat) (v : α) (r : Tree α)
    (hnlt : ¬ x < k) (hx : x > k)
    (hl_all : ForallTree (fun y _ => y < k) l)
    (hr_all : ForallTree (fun y _ => y > k) r)
    (hl_bst : BST l) (hr_bst : BST r)
    (ih_right : BST (insert x w r)) :
    BST (insert x w (tree l k v r)) := := by
  unfold insert
  simp [hnlt, hx]
  constructor
  · exact hl_all
  · exact insert_preserves_forall (fun y _ => y > k) r hr_all x w hx
  · exact hl_bst
  · exact ih_right

@[blueprint
    (statement := /-- If neither `x < k` nor `x > k` holds for natural numbers, then `x = k`. -/)
    (proof := /-- Use trichotomy for the linear order on natural numbers. The alternatives `x < k` and `k < x` contradict the two assumptions, leaving only equality. -/)]
lemma nat_eq_of_not_lt_not_gt (x k : Nat) (hnlt : ¬ x < k) (hngt : ¬ x > k) :
    x = k := := by
  omega

@[blueprint
    (statement := /-- If the inserted key is exactly the root key, then insertion into a nonempty tree takes the update branch and returns the same subtrees with the new root value. -/)
    (proof := /-- Substitute `x = k`, unfold `insert`, and simplify the impossible strict comparisons `k < k` and `k > k`. -/)]
lemma insert_update_eq_tree_of_eq {α : Type u} (x : Nat) (w : α)
    (l : Tree α) (k : Nat) (v : α) (r : Tree α)
    (hxk : x = k) :
    insert x w (tree l k v r) = tree l k w r := := by
  subst hxk
  simp [insert]

@[blueprint
    (statement := /-- For every type `α`, inserted key `x : Nat`, inserted value `w : α`, node key `k : Nat`, node value `v : α`, and subtrees `l r : Tree α`, if `¬ x < k` and `¬ x > k`, then inserting into `tree l k v r` takes the update branch and is definitionally the same as replacing the root value by `w`. -/)
    (proof := /-- First derive `x = k` from the two failed strict comparisons using `nat_eq_of_not_lt_not_gt`. Then apply the same-key update lemma `insert_update_eq_tree_of_eq`. -/)]
lemma insert_update_eq_tree {α : Type u} (x : Nat) (w : α)
    (l : Tree α) (k : Nat) (v : α) (r : Tree α)
    (hnlt : ¬ x < k) (hngt : ¬ x > k) :
    insert x w (tree l k v r) = tree l k w r := := by
  have hxk : x = k := by omega
  subst hxk
  simp [insert]

@[blueprint
    (statement := /-- A tree with the same subtrees and the same root key remains a BST when only the root value is replaced. -/)
    (proof := /-- Apply the `BST.tree` constructor using the unchanged left ordering invariant, right ordering invariant, left BST proof, and right BST proof. The root value is irrelevant to all four required fields. -/)]
lemma bst_replace_root_value {α : Type u}
    (l : Tree α) (k : Nat) (old new : α) (r : Tree α)
    (hl_all : ForallTree (fun y _ => y < k) l)
    (hr_all : ForallTree (fun y _ => y > k) r)
    (hl_bst : BST l) (hr_bst : BST r) :
    BST (tree l k new r) := := by
  constructor
  · exact hl_all
  · exact hr_all
  · exact hl_bst
  · exact hr_bst

@[blueprint
    (statement := /-- If the inserted key is equal to the root key, then insertion into the node preserves the BST invariant, using the original ordering and subtree BST witnesses. -/)
    (proof := /-- Rewrite `insert x w (tree l k v r)` to `tree l k w r` using `insert_update_eq_tree_of_eq`, then rebuild the BST node with `bst_replace_root_value`. This isolates the update-branch proof from the order-theoretic step that proves `x = k`. -/)]
lemma insert_update_bst_of_eq {α : Type u} (x : Nat) (w : α)
    (l : Tree α) (k : Nat) (v : α) (r : Tree α)
    (hxk : x = k)
    (hl_all : ForallTree (fun y _ => y < k) l)
    (hr_all : ForallTree (fun y _ => y > k) r)
    (hl_bst : BST l) (hr_bst : BST r) :
    BST (insert x w (tree l k v r)) := := by
  subst hxk
  simp [insert]
  constructor
  · exact hl_all
  · exact hr_all
  · exact hl_bst
  · exact hr_bst

@[blueprint
    (statement := /-- For every type `α`, inserted key `x : Nat`, inserted value `w : α`, node key `k : Nat`, node value `v : α`, and subtrees `l r : Tree α`, if `¬ x < k` and `¬ x > k`, all nodes of `l` are less than `k`, all nodes of `r` are greater than `k`, and both original subtrees are binary search trees, then inserting `(x, w)` into `tree l k v r` gives a binary search tree. -/)
    (proof := /-- First use linearity of natural numbers to derive `x = k` from the two failed strict comparisons. Then invoke `insert_update_bst_of_eq`, which handles the same-key update branch and reconstructs the BST node with the unchanged subtrees. -/)]
lemma insert_update_bst {α : Type u} (x : Nat) (w : α)
    (l : Tree α) (k : Nat) (v : α) (r : Tree α)
    (hnlt : ¬ x < k) (hngt : ¬ x > k)
    (hl_all : ForallTree (fun y _ => y < k) l)
    (hr_all : ForallTree (fun y _ => y > k) r)
    (hl_bst : BST l) (hr_bst : BST r) :
    BST (insert x w (tree l k v r)) := := by
  have hxk : x = k := nat_eq_of_not_lt_not_gt x k hnlt hngt
  exact insert_update_bst_of_eq x w l k v r hxk hl_all hr_all hl_bst hr_bst

@[blueprint
    (statement := /-- For every type `α`, key `k : Nat`, value `v : α`, and tree `t : Tree α`, if `t` is a binary search tree, then `insert k v t` is also a binary search tree. -/)
    (proof := /-- Proceed by induction on the evidence that `t` is a `BST`. In the empty case, apply `insert_empty_bst`. In a node case `tree l k' v' r`, case split on the comparisons made by `insert`. If the inserted key is less than `k'`, apply `insert_left_bst` using the left induction hypothesis. If the inserted key is greater than `k'`, apply `insert_right_bst` using the right induction hypothesis. If neither strict comparison holds, apply `insert_update_bst`, which handles the update branch by first deriving key equality and then rebuilding the root with the new value. -/)]
theorem bst_insert_of_bst {α : Type u} (k : Nat) (v : α) (t : Tree α)
    : BST t → BST (insert k v t) := := by
  intro h
  induction h with
  | empty =>
      exact insert_empty_bst k v
  | tree l k' v' r hl_all hr_all hl_bst hr_bst ihl ihr =>
      unfold insert
      by_cases hx : k < k'
      · simp [hx]
        constructor
        · exact insert_preserves_forall (P := fun y _ => y < k') l hl_all k v hx
        · exact hr_all
        · exact ihl
        · exact hr_bst
      · by_cases hg : k > k'
        · simp [hx, hg]
          constructor
          · exact hl_all
          · exact insert_preserves_forall (P := fun y _ => y > k') r hr_all k v hg
          · exact hl_bst
          · exact ihr
        · have h_eq : k = k' := nat_eq_of_not_lt_not_gt k k' hx hg
          subst h_eq
          simp [hx, hg]
          constructor
          · exact hl_all
          · exact hr_all
          · exact hl_bst
          · exact hr_bst

end Tree
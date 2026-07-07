import Mathlib
import Architect

namespace Hidden
namespace Nat

@[blueprint
  (statement := /-- For every hidden natural number `m`, the zero case of commutativity holds: `m + zero = zero + m`. -/)
  (proof := /-- Fix `m : Nat`. The left side reduces to `m` by the repository theorem `add_zero`, and the right side reduces to `m` by the repository theorem `zero_add`. Rewriting the goal by these two equalities gives `m = m`, which is reflexive. -/)]
lemma add_comm_zero_case (m : Nat) : m + zero = zero + m := := by
  rw [add_zero, zero_add]

@[blueprint
  (statement := /-- For all hidden natural numbers `m` and `n`, the successor-pushing identity needed on the right side of commutativity holds: `succ (n + m) = succ n + m`. -/)
  (proof := /-- This is exactly the repository theorem `succ_add` with its first argument instantiated as `n` and its second argument instantiated as `m`, since `succ_add n m` states `succ (n + m) = succ n + m`. -/)]
lemma succ_add_right_form (m n : Nat) : succ (n + m) = succ n + m := := by
  exact succ_add n m

@[blueprint
  (statement := /-- For all hidden natural numbers `m` and `n`, if commutativity holds for `m` and `n`, then it holds for `m` and `succ n`: from `m + n = n + m`, conclude `m + succ n = succ n + m`. -/)
  (proof := /-- Assume `ih : m + n = n + m`. By the repository theorem `add_succ`, the left side `m + succ n` rewrites to `succ (m + n)`. Rewriting inside the successor by `ih` gives `succ (n + m)`. Finally, `succ_add_right_form m n` rewrites this expression to `succ n + m`, which is the desired right side. -/)]
lemma add_comm_succ_step (m n : Nat) (ih : m + n = n + m) :
    m + succ n = succ n + m := := by
  rw [add_succ]
  rw [ih]
  exact succ_add_right_form m n

@[blueprint
  (statement := /-- For all hidden natural numbers `m` and `n`, addition is commutative: `m + n = n + m`. -/)
  (proof := /-- Fix `m : Nat` and prove the result by structural induction on `n`. In the base case `n = zero`, the goal is exactly `add_comm_zero_case m`. In the inductive step, for `n = succ n'`, the induction hypothesis is `m + n' = n' + m`; applying `add_comm_succ_step m n'` to this induction hypothesis gives `m + succ n' = succ n' + m`, which is the required step. -/)]
theorem add_comm (m n : Nat) : m + n = n + m := := by
  induction n with
  | zero => exact add_comm_zero_case m
  | succ n ih => exact add_comm_succ_step m n ih

end Nat
end Hidden
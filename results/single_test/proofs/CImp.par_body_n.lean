import Mathlib
import Architect
import Frap.SmallStep

open Imp
open AExp
open BExp
open Multi

namespace CImp

open Com
open CStep

@[blueprint
    (statement := /-- If a state has `x = n` and `y = 0`, then one scheduled right-thread loop iteration of `par_loop` reaches another copy of `par_loop` in a state with `x = n + 1` and `y = 0`. -/)
    (proof := /-- Apply the repository theorem `par_body_n__Sn`, which performs exactly one iteration of the right-hand while thread. Its resulting state is `x !-> n + 1; st`; use the update facts to show this state has the required readings of `x` and `y`. -/)]
lemma par_loop_one_right_iteration (n : Nat) (st : State)
    : st x = n ∧ st y = 0
      → ∃ st_next, Multi CStep (par_loop, st) (par_loop, st_next)
          ∧ st_next x = n + 1 ∧ st_next y = 0 := := by
  intro h
  refine ⟨(x !-> n + 1; st), ?_, ?_, ?_⟩
  · exact par_body_n__Sn n st h
  · exact lookup_update_eq st x (n + 1)
  · rw [lookup_update_neq]
    · exact h.2
    · decide

@[blueprint
    (statement := /-- For every natural number `n` and state `st`, if `st x = 0` and `st y = 0`, then after scheduling the right-hand while-thread for `n` iterations, `par_loop` can return to the same command `par_loop` in some state whose `x` value is `n` and whose `y` value is still `0`. -/)
    (proof := /-- Prove by induction on `n`. For `0`, use `multi_refl` and the initial hypotheses, taking the original state as witness. For the successor case, use the induction hypothesis to reach an intermediate state with `x = n` and `y = 0`; then apply `par_loop_one_right_iteration` to perform one more right-thread loop iteration. Compose the two multi-step executions by transitivity and use the state facts supplied by the one-iteration lemma. -/)]
lemma par_loop_repeats_to_x_update (n : Nat) (st : State)
    : st x = 0 ∧ st y = 0
      → ∃ st_mid, Multi CStep (par_loop, st) (par_loop, st_mid)
          ∧ st_mid x = n ∧ st_mid y = 0 := := by
  intro h
  induction n generalizing st with
  | zero =>
      refine ⟨st, ?_, ?_, ?_⟩
      · apply multi_refl
      · exact h.1
      · exact h.2
  | succ n ih =>
      rcases ih st h with ⟨st_mid, hmulti, hx, hy⟩
      rcases par_loop_one_right_iteration n st_mid ⟨hx, hy⟩ with ⟨st_next, hstep, hxnext, hynext⟩
      refine ⟨st_next, ?_, hxnext, hynext⟩
      exact multi_trans _ _ _ _ _ hmulti hstep

@[blueprint
    (statement := /-- For every natural number `n` and state `st`, if `st x = 0` and `st y = 0`, then `par_loop` can take zero or more steps to a copy of `par_loop` in some intermediate state whose `x` value is `n` and whose `y` value is still `0`. -/)
    (proof := /-- This is exactly `par_loop_repeats_to_x_update`. -/)]
lemma par_loop_reaches_state_with_x_n (n : Nat) (st : State)
    : st x = 0 ∧ st y = 0
      → ∃ st_mid, Multi CStep (par_loop, st) (par_loop, st_mid)
          ∧ st_mid x = n ∧ st_mid y = 0 := := by
  exact par_loop_repeats_to_x_update n st

@[blueprint
    (statement := /-- For every natural number `n` and state `st`, if `st x = 0` and `st y = 0`, then there exists a final state `st'` such that `par_loop` can take zero or more small steps from `(par_loop, st)` to `(par_loop, st')`, and in that state `x` has value `n` while `y` still has value `0`. -/)
    (proof := /-- Apply `par_loop_reaches_state_with_x_n` to the initial assumptions. It supplies a witness state `st_mid`, a multi-step execution from `(par_loop, st)` to `(par_loop, st_mid)`, and the facts `st_mid x = n` and `st_mid y = 0`. Choose `st' = st_mid`; the supplied execution is exactly the required execution, and the supplied state facts are exactly the required final conjuncts. -/)]
theorem par_body_n n st
    : st x = 0 ∧ st y = 0
      → ∃ st', Multi CStep (par_loop, st) (par_loop, st')
          ∧ st' x = n ∧ st' y = 0 := := by
  intro h
  exact par_loop_reaches_state_with_x_n n st h

end CImp
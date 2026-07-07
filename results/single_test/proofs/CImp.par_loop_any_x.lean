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

@[blueprint (statement := /-- `loop_guard` is the boolean test used by the right-hand thread of `par_loop`, namely `y == 0`. -/)]
def loop_guard : BExp :=
  b_eq (a_id y) (a_num 0)

@[blueprint (statement := /-- `loop_body` is the command executed by the right-hand thread of `par_loop`, namely `x := x + 1`. -/)]
def loop_body : Com :=
  c_asgn x (a_plus (a_id x) (a_num 1))

@[blueprint (statement := /-- `right_loop` is the while-loop thread of `par_loop`, namely `while y == 0 do x := x + 1 end`. -/)]
def right_loop : Com :=
  c_while loop_guard loop_body

@[blueprint (statement := /-- `after_left_thread` is the parallel command obtained from `par_loop` after the left assignment thread has reduced to `skip` while the right loop remains. -/)]
def after_left_thread : Com :=
  c_par c_skip right_loop

@[blueprint
  (statement := /-- For every natural number `n`, starting from the empty state, the right thread of `par_loop` can execute exactly enough loop-body iterations to reach some state `st` with `x = n` and `y = 0`, while the whole parallel command is still `par_loop`. -/)
  (proof := /-- Apply the repository theorem `par_body_n` to `n` and `empty`.  Its precondition is `empty x = 0 ∧ empty y = 0`, which follows from the definition of `empty`.  The conclusion is precisely the existence of a state reachable from `(par_loop, empty)` to `(par_loop, st)` with `st x = n` and `st y = 0`. -/)]
lemma run_right_loop_to_x_n (n : Nat) :
    ∃ st, Multi CStep (par_loop, empty) (par_loop, st)
      ∧ st x = n ∧ st y = 0 := := by
  exact par_body_n n empty (by simp [empty])

@[blueprint
  (statement := /-- For every natural number `n` and state `st`, if `st x = n`, then the update `y !-> 1; st` still maps `x` to `n`. -/)
  (proof := /-- Use the update equations from the state library: updating variable `y` changes only `y`, and since `x` is distinct from `y`, `(y !-> 1; st) x = st x`.  Combining this equality with the premise `st x = n` gives `(y !-> 1; st) x = n`. -/)]
lemma update_y_preserves_x (n : Nat) (st : State) :
    st x = n → (y !-> 1; st) x = n := := by
  intro h
  simpa [h]

@[blueprint
  (statement := /-- The left component of `par_loop` is the assignment `y := 1`, and it can take one parallel-left command step to `skip`, producing the state update `y !-> 1; st`.  The target is written directly with the concrete right-hand loop from `par_loop`. -/)
  (proof := /-- Unfold `par_loop`, then apply `cs_par1` to the assignment rule `cs_asgn` for variable `y` and numeral `1`. -/)]
lemma left_thread_sets_y_one_concrete_step (st : State) :
    CStep (par_loop, st)
      (c_par c_skip
        (c_while (b_eq (a_id y) (a_num 0))
          (c_asgn x (a_plus (a_id x) (a_num 1)))),
        y !-> 1; st) := := by
  unfold par_loop
  apply CStep.cs_par1
  apply CStep.cs_asgn

@[blueprint
  (statement := /-- In the state `y !-> 1; st`, the concrete guard `y == 0` takes one boolean step to the concrete comparison `1 == 0`. -/)
  (proof := /-- Apply `BStep.bs_eq1` to the arithmetic identifier step for `y`; the updated state maps `y` to `1`. -/)]
lemma concrete_guard_after_y_one_reads_one (st : State) :
    BStep (y !-> 1; st) (b_eq (a_id y) (a_num 0))
      (b_eq (a_num 1) (a_num 0)) := := by
  apply BStep.bs_eq1
  apply AStep.as_id


@[blueprint
  (statement := /-- The boolean equality test `1 == 0` steps to `b_false` in the updated state. -/)
  (proof := /-- Apply `BStep.bs_eq` to the numerals `1` and `0`, and simplify the decidable equality test. -/)]
lemma one_eq_zero_steps_false (st : State) :
    BStep (y !-> 1; st) (b_eq (a_num 1) (a_num 0)) b_false := := by
  apply BStep.bs_eq

@[blueprint
  (statement := /-- Once both components of the parallel command are `skip`, the parallel command can take the final `cs_parDone` step to become `skip`, leaving the state unchanged. -/)
  (proof := /-- This is the multistep form of the single rule `cs_parDone`. -/)]
lemma par_done_multi (st : State) :
    Multi CStep (c_par c_skip c_skip, st) (c_skip, st) := := by
  apply Multi.multi_step
  · apply CStep.cs_parDone
  · apply Multi.multi_refl

@[blueprint
  (statement := /-- From the concrete state immediately after the left thread has set `y` to `1`, the right loop can unfold, test `y == 0`, observe the false guard, and the parallel command can terminate, leaving the updated state unchanged. -/)
  (proof := /-- Use the exact concrete command produced by the proved left-thread step.  Take a right-parallel `while` step, then two right-parallel conditional guard steps using `concrete_guard_after_y_one_reads_one st` and `one_eq_zero_steps_false st`, then the right-parallel false branch, and finally compose with `par_done_multi`.  This avoids the repeatedly brittle intermediate lemmas for isolated conditional and parallel-context steps. -/)]
lemma concrete_right_loop_exits_after_y_one_multi (st : State) :
    Multi CStep
      (c_par c_skip
        (c_while (b_eq (a_id y) (a_num 0))
          (c_asgn x (a_plus (a_id x) (a_num 1)))),
        y !-> 1; st)
      (c_skip, y !-> 1; st) := := by
  apply Multi.multi_step
  · apply CStep.cs_par2
    apply CStep.cs_while
  · apply Multi.multi_step
    · apply CStep.cs_par2
      apply CStep.cs_ifStep
      exact concrete_guard_after_y_one_reads_one st
    · apply Multi.multi_step
      · apply CStep.cs_par2
        apply CStep.cs_ifStep
        exact one_eq_zero_steps_false st
      · apply Multi.multi_step
        · apply CStep.cs_par2
          apply CStep.cs_ifFalse
        · exact par_done_multi (y !-> 1; st)

@[blueprint
  (statement := /-- If an intermediate state `st` has `x = n`, then starting from `(par_loop, st)` the left thread can set `y` to `1`, the right thread can observe the false guard and exit, and the final state still maps `x` to `n`. -/)
  (proof := /-- Use the proved one-step reduction of the left thread from `par_loop` to the concrete post-left parallel command.  Then compose with `concrete_right_loop_exits_after_y_one_multi`.  Choose the final state `y !-> 1; st`; its `x` value is `n` by `update_y_preserves_x`. -/)]
lemma finish_from_intermediate_state (n : Nat) (st : State) :
    st x = n →
    ∃ st', Multi CStep (par_loop, st) (c_skip, st') ∧ st' x = n := := by
  intro h
  refine ⟨(y !-> 1; st), ?_, ?_⟩
  apply Multi.multi_step
  · exact left_thread_sets_y_one_concrete_step st
  · exact concrete_right_loop_exits_after_y_one_multi st
  · exact update_y_preserves_x n st h

@[blueprint
  (statement := /-- For every natural number `n` and state `st`, if `par_loop` is reachable from `(par_loop, empty)` in a state `st` satisfying `st x = n ∧ st y = 0`, then the left thread can set `y` to `1`, the right thread can exit, and the whole program can terminate in some final state whose `x` value is `n`. -/)
  (proof := /-- Apply `finish_from_intermediate_state` to the `st x = n` part of the invariant, obtaining a run from `(par_loop, st)` to a final configuration.  Compose this run after the assumed reachability from `(par_loop, empty)` using transitivity of `Multi`. -/)]
lemma finish_after_reaching_x_n (n : Nat) (st : State) :
    Multi CStep (par_loop, empty) (par_loop, st) →
    st x = n ∧ st y = 0 →
    ∃ st', Multi CStep (par_loop, empty) (c_skip, st') ∧ st' x = n := := by
  intro hreach hst
  rcases hst with ⟨hxst, hyst⟩
  rcases finish_from_intermediate_state n st hxst with ⟨st', hfinish, hx⟩
  refine ⟨st', ?_, hx⟩
  exact multi_trans _ _ _ _ _ hreach hfinish

@[blueprint
  (statement := /-- For every natural number `n`, the concurrent program `par_loop` has an execution from the empty state to a final `skip` configuration whose final state maps `x` to `n`. -/)
  (proof := /-- Use `run_right_loop_to_x_n` to choose an intermediate state `st` reachable from `(par_loop, empty)` where `x = n` and `y = 0` while the command is still `par_loop`.  Then apply `finish_after_reaching_x_n` to this multistep execution and invariant, obtaining a final state `st'` with a multistep execution to `(c_skip, st')` and `st' x = n`. -/)]
theorem par_loop_any_x n
    : ∃ st', Multi CStep (par_loop, empty) (c_skip, st')
        ∧ st' x = n := := by
  rcases run_right_loop_to_x_n n with ⟨st, hreach, hx, hy⟩
  exact finish_after_reaching_x_n n st hreach ⟨hx, hy⟩

end CImp
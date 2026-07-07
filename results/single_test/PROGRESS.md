# lean-formal-reasoning-program re-test log

Re-verifying all 20 `lean-formal-reasoning-program` theorems from VeriSoftBench
against the pipeline after the systemic Category 1/2/3 fixes (validated-signal,
regex-consolidation, cache-invariant). Each phase run separately with
`--workers 1`. Checkpoints in `checkpoints/`, extracted final proofs in
`proofs/` (via `extract_proof.py`).

Sorted by `transitive_dep_count` (rough complexity proxy).

| theorem | deps | status | notes |
|---|---|---|---|
| and_associative | 3 | DONE (proved) | 6 nodes, all solved first pass |
| Hidden.Nat.add_comm | 5 | DONE (proved) | 4 nodes, all solved first pass |
| Hidden.List.Palindrome.Palindrome_app_rev | 6 | DONE (proved) | 4 nodes, all solved first pass |
| Hidden.Nat.mul_assoc | 6 | PARKED - genuinely stuck | 6/14 nodes solved, rest blocked on a distributivity cluster (`mul_assoc_mul_add_helper` etc.). Root cause: blueprint keeps citing real-Mathlib `Nat.mul_add`/`Nat.mul_zero`/`Nat.mul_succ` instead of this repo's own shadowed local lemmas (`Hidden.Nat.mul_succ`/`add_assoc`/`add_succ`, declared earlier in the same file - confirmed against VSB's own `used_local_lemmas` ground truth). Tried: (1) added namespace-shadowing guidance to blueprint/refinement prompts - no change; (2) cleared `refinement_history` for a clean-slate retry - still regenerated the same wrong dependency almost verbatim. Not a history-anchoring artifact; looks like a real prior the model won't override from soft prompt guidance alone. `mul_assoc_right_nested_zero` also hit a genuine 300s timeout 3x (correctly classified `infra_error`, confirms that fix works live). 4/8 refinement rounds used. |
| permutation_cons_app | 7 | DONE (proved) | 4 nodes; 2 initially failed (statement_wrong, proof_too_hard), fixed in 1 refinement round |
| Hidden.List.reverse_append | 7 | DONE (proved) | Re-run fresh with the fixed pipeline. 4 nodes, all solved first pass, clean blueprint (no leaked-tag corruption this time - confirms the bug #10 fix holds) |
| permutation_app_comm | 8 | DONE (proved) | 8 nodes; 1 initially statement_wrong, fixed in 1 refinement round |
| Hidden.List.Palindrome.Palindrome_rev | 8 | DONE (proved) | 6 nodes; 1 initially proof_too_hard, fixed in 1 refinement round |
| Tree.balance_BST | 11 | PARKED - exhausted 8/8 refinement rounds | 18/23 nodes solved, incl. the hard core rotation lemma (`balance_rotated_parts_BST`). 3 of 4 case-application wrappers (`balance_black_left_left_BST`/`left_right`/`right_right`) still `statement_wrong` at the iteration cap. Genuine proof difficulty (classic red-black tree rotation case-bash), not an obvious naming/dependency bug like mul_assoc - real incremental progress each round, just ran out of budget. Worth revisiting with either a higher MAX_REFINEMENT_ITERATIONS or by manually inspecting why the case-wrappers can't apply the now-proven rotation lemma. |
| Tree.bst_insert_of_bst | 11 | DONE (proved) | 12 nodes; fixed in 3 refinement rounds (proof_too_hard, then statement_wrong twice on a trichotomy/root-replacement lemma) |
| Imp.Hoare.hoare_while | 15 | PARKED - exhausted 8/8 refinement rounds | 12/15 nodes solved. Repeatedly stuck on the core inductive lemma about while-loop evaluation preserving the invariant until exit (`while_eval_preserves_until_exit`/`_aux`/`_induction_post`, renamed each round), oscillating proof_too_hard/statement_wrong across all 8 rounds with no convergence. Looks like genuine difficulty with the recursive-evaluation induction, similar shape to balance_BST but without balance_BST's eventual payoff. |
| TM.progress | 16 | DONE (proved) | 15 nodes; fixed in 2 refinement rounds |
| step_normalizing | 17 | DONE (proved) | 8 nodes; 1 initially proof_too_hard, fixed in 1 refinement round |
| Imp.fold_constants_bexp_sound | 21 | DONE (proved) | Large blueprint (~35 nodes by the end); took 7 refinement rounds, converging steadily each round (eq/le/neq comparison-folding cases resolved one at a time) rather than getting stuck |
| Imp.Hoare.hoare_if | 21 | DONE (proved) | 4 nodes, all solved first pass |
| CImp.par_body_n | 24 | DONE (proved) | Fixed in 2 refinement rounds. Notable: round 2 dropped an orphaned dead-end node (`par_loop_terminates_preserving_x`) that wasn't actually in the target's dependency chain - the target itself had already solved in round 1 but `all_proved()` correctly still reported FAIL since that unreachable node was unsolved; refinement pruned it naturally on the next round. |
| TM.soundness | 26 | DONE (proved) | Took 7 refinement rounds, converging steadily like fold_constants_bexp_sound (each round chipped away at the multistep-preservation induction) rather than getting stuck like the parked theorems |
| CImp.par_loop_any_x | 27 | DONE (proved) | Solved on the last usable refinement round (8/8) - a long grind through a concurrent-program loop-unrolling chain, steadily converging one node at a time each round rather than getting stuck |
| Imp.optimize_0plus_com_sound | 35 | DONE (proved) - was PARKED, fixed by giving Phase 1/3 a repo_search tool | Root cause was `repo_context` construction (`_build_verif_context`) only reading the same file's preceding content, never following `import Frap.Trans` - Phase 1 had zero information about `AExp`/`BExp`/`Com`/`cequiv`/`ctrans_sound` and fabricated placeholder definitions. Fix: gave `generate_blueprint`/`refine_blueprint` a `repo_search` tool (see `_call_with_repo_search` in `src/blueprint.py`), same tool Phase 2 already had, so the model can look up cross-file declarations on demand. Re-ran fresh: Phase 1 produced a correct blueprint referencing the real repo types and matching the ground-truth target signature character-for-character (validated in 50s vs. the old 280-580s of failed retries); hit one unrelated infra snag (the repo's own `Frap.Exercises.Trans.olean` build artifact was simply missing from the shared checkout - fixed with a one-time `lake build Frap.Exercises.Trans`, not a pipeline bug); after that, Phase 2 solved all 5 nodes in a single pass with zero refinement rounds needed. |
| Imp.fold_constants_com_sound | 40 | DONE (proved) | 10 nodes; 1 initially statement_wrong, fixed in 1 refinement round |

## Parked: Hidden.Nat.mul_assoc (namespace-shadowing dependency bug)

**Symptom**: 6/14 blueprint nodes solved; the rest (`mul_assoc_mul_add_helper`,
`mul_assoc_left_distrib_add`, `mul_assoc_product_mul_zero`,
`mul_assoc_right_nested_zero`, `mul_assoc_zero_case`,
`mul_assoc_succ_right_distribute`, `mul_assoc_succ_step`, `mul_assoc`) never
solve across 5 refinement rounds (iteration 5/8 used, `refinement_history`
since cleared for the clean-slate test below).

**Root cause**: `Frap/Inductive.lean` defines its own `inductive Nat` inside
`namespace Hidden` (lines 71-120) as a from-scratch teaching exercise, fully
separate from the real Lean/Mathlib `Nat`. The blueprint keeps citing
dependencies like `Nat.mul_add`, `Nat.mul_zero`, `Nat.mul_succ` — the REAL
Mathlib lemmas, which do not apply to `Hidden.Nat` values. Confirmed via:
- Actual compile errors: `exact Nat.mul_add a b c` → `application type
  mismatch` (see `results/single_test/trace.jsonl`, search
  `mul_assoc_mul_add_helper`).
- VeriSoftBench's own ground truth (`data/verisoftbench.jsonl`, id 284,
  `used_local_lemmas` field) lists the CORRECT dependencies:
  `Hidden.Nat.add_succ`, `Hidden.Nat.add_assoc`, `Hidden.Nat.mul_succ`,
  `Hidden.Nat.add_infix` — all already declared as proven theorems earlier in
  the same file (lines 165-280, before `mul_assoc` at line 283).
- The per-node prover's own `repo_search` calls DID surface the correct local
  lemmas (`mul_succ`, `add_succ`, `succ_add` at their correct line numbers),
  but subsequent `lean_compile` attempts using them (without the
  `Hidden.Nat.` qualification) still failed with `rewrite failed` errors —
  a separate, not-yet-root-caused mechanical issue in how the model composed
  the `rw` chain (possibly ambiguous multi-occurrence rewrites), worth
  checking again if the dependency-naming issue ever gets fixed.

**What was tried and didn't work**:
1. Added a "namespace shadowing" rule to `prompts/blueprint_system.md` and
   `prompts/refinement_system.md` (still in place as of this writing — it's a
   reasonable rule to keep even though it didn't fix this case). No change:
   next refinement round still emitted `sorry_using [Nat.mul_add, mul_add,
   left_distrib]`.
2. Cleared `state.refinement_history` on the checkpoint directly (to rule out
   the model anchoring on its own prior-round text) and reran Phase 3 fresh.
   Still regenerated `mul_assoc_mul_add_helper` with nearly the exact same
   proof sketch as the very first Phase 1 blueprint ("This is exactly the
   standard theorem `Nat.mul_add`"). This rules out history-anchoring as the
   cause — it looks like a strong, hard-to-override model prior that soft
   prompt guidance doesn't beat, not a context-accumulation artifact.

**Untried ideas for next time**:
- Mechanical (not persuasive) fix: detect the shadowed-namespace pattern and
  hard-inject the VSB dataset's `used_local_lemmas` names as a *required*
  citation list in the blueprint prompt, rather than relying on the model to
  infer them from repo context prose.
- Separately debug the `rw`-chain composition issue the per-node prover hit
  even when it did call `repo_search` and found the right lemma names — may
  need a nudge toward `simp [...]`/`omega`-style automation over manual `rw`
  chains for this repo's style of proof.
- Isolated single-node test harness (reusable): a standalone script that
  reruns one blueprint node directly via `orchestrator.prove_dag` with a
  restricted `nodes_to_retry` set, bumped `node_timeout_s`, and a monkeypatched
  `reasoning_effort` — used this session at
  `/tmp/claude-1000/.../scratchpad/test_stuck_node.py` (session-specific
  scratchpad path, will need re-creating in a fresh session; the pattern is
  documented here for reference).

## How to extract a proof from a checkpoint

```
python3 results/single_test/extract_proof.py <thm_name>
```

Writes `results/single_test/proofs/<thm_name>.lean` with every solved node's
proof spliced in (0 remaining `sorry_using` = fully proved).

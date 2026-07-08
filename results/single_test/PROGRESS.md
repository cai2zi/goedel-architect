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
| Hidden.Nat.mul_assoc | 6 | DONE (proved) - was PARKED, fixed by giving Phase 1/3 a repo_search tool | Same root cause and fix as `Imp.optimize_0plus_com_sound` below. Old blueprint (built before Phase 1 had `repo_search`) kept citing real-Mathlib `Nat.mul_add`/`Nat.mul_zero`/`Nat.mul_succ` instead of this repo's own shadowed local lemmas. Re-ran fresh with the repo_search fix in place: Phase 1 produced a much simpler 6-node blueprint (vs. 14 before) that correctly cites `Hidden.Nat.add_zero`/`mul_zero`/`add_succ`/`mul_succ`/`add_assoc` throughout; Phase 2 then solved all 6 nodes in a single pass (31.2s total), zero refinement rounds needed. Confirms this was a Phase 1 blueprint-generation gap, not a per-node proving issue - the per-node prover's own `repo_search` calls had already found the right lemma names in the earlier attempt, but the wrong dependency was baked into the blueprint before Phase 2 ever ran. |
| permutation_cons_app | 7 | DONE (proved) | 4 nodes; 2 initially failed (statement_wrong, proof_too_hard), fixed in 1 refinement round |
| Hidden.List.reverse_append | 7 | DONE (proved) | Re-run fresh with the fixed pipeline. 4 nodes, all solved first pass, clean blueprint (no leaked-tag corruption this time - confirms the bug #10 fix holds) |
| permutation_app_comm | 8 | DONE (proved) | 8 nodes; 1 initially statement_wrong, fixed in 1 refinement round |
| Hidden.List.Palindrome.Palindrome_rev | 8 | DONE (proved) | 6 nodes; 1 initially proof_too_hard, fixed in 1 refinement round |
| Tree.balance_BST | 11 | PARKED (re-tested 2026-07-07, much closer) - 60/64 nodes solved | Re-ran fresh post-repo_search-fix through all 8 refinement rounds again, then hand-fixed 5 more stuck leaf nodes myself (see below) once I confirmed they were proof-writing mistakes, not genuine difficulty. Down to exactly 4 unsolved: `balance_black_empty_left_BST`, `balance_black_red_left_BST`, `balance_black_BST`, and the root `balance_BST` (blocked only on `balance_black_BST`). All 4 rotation-case BST lemmas (`balance_left_left_BST`/`left_right`/`right_left`/`right_right`) and every order/bound helper ARE now proved. The remaining gap is a genuine Lean tactic-engineering puzzle in `balance_black_BST`'s own proof (`unfold balance; split` doesn't behave as naively expected against the real `balance` function's nested match + a literal `Color.black` scrutinee - see `vsb-balance-bst-progress` memory for the detailed diagnosis), not proof-difficulty in the traditional sense. Needs interactive Lean goal inspection to finish cleanly; parked here by user decision to move on to `Imp.Hoare.hoare_while` instead. |
| Tree.bst_insert_of_bst | 11 | DONE (proved) | 12 nodes; fixed in 3 refinement rounds (proof_too_hard, then statement_wrong twice on a trichotomy/root-replacement lemma) |
| Imp.Hoare.hoare_while | 15 | PARKED (re-tested 2026-07-07, cancelled early by user - still oscillating) | Re-ran fresh post-repo_search-fix; reproduced the exact same oscillation signature as the original run within just 4 rounds (round 1-2: split into `while_exit_guard_false`/`while_preserves_invariant`, both stuck; round 3: decomposed further into `ceval_while_exit_guard_false_aux`/`ceval_while_preserves_invariant_aux`, both `proof_too_hard`; round 4: merged back into one `ceval_while_invariant_and_exit_aux`, `statement_wrong`) - splitting apart then merging back with no net progress, unlike `Tree.balance_BST`'s monotonic convergence in the same session. Confirms this is genuine difficulty with the core while-loop-evaluation induction, not a naming/wiring bug fixable by the repo_search fix or by hand like balance_BST's stuck nodes were. Cancelled at round 4/8 by user judgment call rather than exhausting the full budget, since the pattern was already clearly non-convergent. |
| TM.progress | 16 | DONE (proved) | 15 nodes; fixed in 2 refinement rounds |
| step_normalizing | 17 | DONE (proved) | 8 nodes; 1 initially proof_too_hard, fixed in 1 refinement round |
| Imp.fold_constants_bexp_sound | 21 | DONE (proved) | Large blueprint (~35 nodes by the end); took 7 refinement rounds, converging steadily each round (eq/le/neq comparison-folding cases resolved one at a time) rather than getting stuck |
| Imp.Hoare.hoare_if | 21 | DONE (proved) | 4 nodes, all solved first pass |
| CImp.par_body_n | 24 | DONE (proved) | Fixed in 2 refinement rounds. Notable: round 2 dropped an orphaned dead-end node (`par_loop_terminates_preserving_x`) that wasn't actually in the target's dependency chain - the target itself had already solved in round 1 but `all_proved()` correctly still reported FAIL since that unreachable node was unsolved; refinement pruned it naturally on the next round. |
| TM.soundness | 26 | DONE (proved) | Took 7 refinement rounds, converging steadily like fold_constants_bexp_sound (each round chipped away at the multistep-preservation induction) rather than getting stuck like the parked theorems |
| CImp.par_loop_any_x | 27 | DONE (proved) | Solved on the last usable refinement round (8/8) - a long grind through a concurrent-program loop-unrolling chain, steadily converging one node at a time each round rather than getting stuck |
| Imp.optimize_0plus_com_sound | 35 | DONE (proved) - was PARKED, fixed by giving Phase 1/3 a repo_search tool | Root cause was `repo_context` construction (`_build_verif_context`) only reading the same file's preceding content, never following `import Frap.Trans` - Phase 1 had zero information about `AExp`/`BExp`/`Com`/`cequiv`/`ctrans_sound` and fabricated placeholder definitions. Fix: gave `generate_blueprint`/`refine_blueprint` a `repo_search` tool (see `_call_with_repo_search` in `src/blueprint.py`), same tool Phase 2 already had, so the model can look up cross-file declarations on demand. Re-ran fresh: Phase 1 produced a correct blueprint referencing the real repo types and matching the ground-truth target signature character-for-character (validated in 50s vs. the old 280-580s of failed retries); hit one unrelated infra snag (the repo's own `Frap.Exercises.Trans.olean` build artifact was simply missing from the shared checkout - fixed with a one-time `lake build Frap.Exercises.Trans`, not a pipeline bug); after that, Phase 2 solved all 5 nodes in a single pass with zero refinement rounds needed. |
| Imp.fold_constants_com_sound | 40 | DONE (proved) | 10 nodes; 1 initially statement_wrong, fixed in 1 refinement round |

## RESOLVED: Hidden.Nat.mul_assoc (namespace-shadowing dependency bug)

Fixed by the same Phase 1/3 `repo_search` tool built for `Imp.optimize_0plus_com_sound`
(see that row above and `vsb-repo-context-missing-cross-file-imports` memory).
The section below is kept as the original diagnosis record.

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

## Parked: Tree.balance_BST (2026-07-07 re-test, 60/64 nodes)

Re-ran fresh (deleted the old checkpoint) through all 8 refinement rounds
post-repo_search-fix. Converged much further than the original run (which had
capped at 18/23 nodes on a smaller blueprint) - this run's blueprint grew to
64 nodes and reached 55/64 solved by round 8, then I manually diagnosed and
hand-fixed 5 more:

- `bst_left_of_left_node`, `bst_right_of_right_node` - the automated prover
  used manual `cases`/`rename_i` pattern-matching or wrong named-argument
  syntax instead of just composing an already-proved helper twice
  (`bst_left_of_node (bst_left_of_node h)`). Not genuinely hard.
- `balance_left_right_right_order` - the blueprint's own dependency list was
  exactly right (3 sub-lemmas whose conclusions are precisely the 3 subgoals
  `forallTree_node_from_parts` needs, in order); the automated prover's
  `apply forallTree_node_from_parts` just failed to infer the lambda motive
  `P`. Fixed by using a fully-explicit `exact forallTree_node_from_parts
  (fun n _ => n > y) Color.black c z vz d (...) (...) (...)` instead of `apply`.
- `balance_right_left_BST` - a purpose-built assembly lemma
  (`balance_right_left_assemble_BST`, itself already solved) existed for
  exactly this goal, but the automated proof ignored it and tried `constructor`
  on the raw `BST.tree` structure instead, hitting argument-order mismatches.

Each hand-written fix was verified independently by calling
`VSBLeanCompiler.check_blueprint()` directly on the fully-substituted file
before being injected into the checkpoint's `proved_cache` (with a matching
`proof_cache_keys` entry via `BlueprintNode.cache_key()`) and re-running
`--phase 2` to let the pipeline pick up newly-unblocked downstream nodes.

**Found in passing**: `_substitute_proof` (`src/pipeline.py`) always inserts
its own `":= "` before the spliced proof body, but `proved_cache` values
already carry their own leading `:=` (confirmed in every `node_results`/
`proved_cache` entry inspected) - every `_substitute_proof` call site
(`extract_proof.py`, `_assemble_partial_file`, `_assemble_final_file`)
produces a `:= := by ...` double-colon-equals. Harmless for `extract_proof.py`
(human-reading only, never compiled) and apparently never hit by the real
pipeline's own compile paths (`_aux_lemma_decls` builds `signature() +
proved_cache[name]` instead, which is the correct single-`:=` concatenation),
but it broke my own manual `check_blueprint()` validation until I added a
`re.sub(r":=\s*:=", ":=", lean)` post-process step. Worth a real fix in
`_substitute_proof` at some point since it's a footgun for any future
direct-compile use of `_assemble_partial_file`/`_assemble_final_file`.

**Where it's stuck now**: exactly 4 nodes - `balance_black_empty_left_BST`,
`balance_black_red_left_BST`, `balance_black_BST`, and the root `balance_BST`
(blocked only on `balance_black_BST`). All 4 rotation-case BST lemmas and
every order/bound helper are proved; the real `balance` function
(`Frap/RedBlack.lean:61`) is:
```
def balance (c : Color) (l : Tree α) (k : Nat) (vk : α) (r : Tree α) : Tree α :=
  match c with
  | red => tree red l k vk r
  | black =>
    match (l, k, vk, r) with
    | (tree red (tree red a x vx b) y vy c, z, vz, d)
    | (tree red a x vx (tree red b y vy c), z, vz, d)
    | (a, x, vx, tree red (tree red b y vy c) z vz d)
    | (a, x, vx, tree red b y vy (tree red c z vz d))
        => tree red (tree black a x vx b) y vy (tree black c z vz d)
    | _ => tree black l k vk r
```
Tried `unfold balance; repeat' (split <;> subst_vars); all_goals first | exact
balance_left_left_BST .. | ... | exact bst_node_from_bounds ..` (using
underscores for the 10 explicit leading args each rotation lemma takes) -
`lake env lean`'s raw output (captured directly, not through the
error-line-only filter) showed `split` produces a goal for the `Color.red`
outer-match arm even though the target already fixes color to the literal
`Color.black`, and the rotation-case goals show the hypotheses `hlt`/`hgt`/
`hbstl`/`hbstr` NOT actually specialized by `subst_vars` to the split's new
pattern variables. This needs interactive Lean goal inspection (not batch
`lake env lean` + regex-filtered error output) to resolve efficiently - the
repo's own `RedBlack.lean` notes exactly this class of match needs `repeat'`
splitting (see `ins_not_empty`'s proof for a working example on a similarly-
shaped double match), so the right tactic likely exists but wasn't found
blind in the time spent. Paused here by user choice to move to
`Imp.Hoare.hoare_while` instead; worth resuming with an interactive Lean
session (VS Code / any LSP-backed environment) rather than more blind
batch-and-guess iterations.

## How to extract a proof from a checkpoint

```
python3 results/single_test/extract_proof.py <thm_name>
```

Writes `results/single_test/proofs/<thm_name>.lean` with every solved node's
proof spliced in (0 remaining `sorry_using` = fully proved).

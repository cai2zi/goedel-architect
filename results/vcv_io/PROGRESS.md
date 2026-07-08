# VCV-io test log

Testing the pipeline against `VCV-io` (probabilistic/oracle computation
library - PMF-based semantics, monad transformers, crypto-style verification),
a step up in complexity from the small teaching repos tested so far. Each
phase run separately with `--workers 1`.

| theorem | nodes | status | notes |
|---|---|---|---|
| StateT.set_get | 3 | DONE (proved) | Found and worked around a real pipeline bug - see below. `get_bind_set_eq_pure` was misclassified `FORMALLY_NEGATED` (the prover machine-checks a proof of the negation before giving up, and this LGTM'd) when the lemma is actually TRUE; verified independently via `check_blueprint`, then manually injected the correct proof into `proved_cache`/`proof_cache_keys` to unblock the root theorem, which then solved in one pass. |
| PureEquiv.map_pure_inv | 5 | DONE (proved) | Single pass, zero refinement rounds needed |
| BindEquiv.map_bind_inv | 5 | DONE (proved) | Single pass, zero refinement rounds needed |
| OracleComp.evalDist_liftComp | 14 | DONE (proved) | Stuck on `evalDist_liftComp_query_bind_case` for 3 rounds straight (oscillating proof_too_hard/statement_wrong, "failed to synthesize" typeclass errors) before round 5's refinement finally decomposed it into 4 small congruence/rewrite sub-lemmas (`evalDist_liftComp_continuation_function_eq`, `evalDist_query_bind_right_normalized`, `evalDist_lifted_query_bind_initial_rewrite`, `evalDist_lifted_query_bind_congr`) which all solved immediately, cascading to close the whole theorem in one pass. A good example of oscillation NOT meaning stuck-forever - unlike `hoare_while`, this one broke through once the right decomposition was found. |

## Bug found: `_probe_negation` can produce false-positive FORMALLY_NEGATED

**Where**: `src/prover.py::_probe_negation` (~line 478-520), called from
`_prove_node_inner` (~line 370).

**What's wrong**: two compounding issues.
1. The call `compiler.check(args.get("proof_body", ""), aux_lemmas=parent_decls)`
   never passes `node_decl` (unlike the normal proving path's `compiler.check(
   proof_body, aux_lemmas=full_aux, node_decl=node_decl)` a few lines up) - so
   `VSBLeanCompiler.check()` falls back to `entry["thm_stmt"]`, the ROOT
   theorem's own statement, regardless of which sub-node is being negated.
2. Even if `node_decl` were passed, `_node_signature(node_decl)` just extracts
   the node's own POSITIVE signature - nothing anywhere in this code path
   constructs an actual negated goal (`¬ (conclusion)`) for the compiler to
   check against. The whole mechanism relies entirely on the model
   self-policing via the prompt text ("Prove `neg_node_name` showing
   `¬ (conclusion)`"), with no verification that what got compiled is
   actually a negation rather than a restated positive proof.

**Consequence**: if the model, asked to disprove a TRUE statement, instead
(correctly) writes a valid proof of the ORIGINAL statement, `compiler.check()`
happily accepts it (since it's checking a positive goal - either the wrong
root theorem or the node's own true statement) and the pipeline reports
`FORMALLY_NEGATED`, permanently locking the node as "disproven" per
`prompts/refinement_system.md`'s rules and misdirecting all downstream
refinement.

**Confirmed concretely**: `get_bind_set_eq_pure`'s recorded "counterexample"
proof (`ext s; change (...); simp`) compiles with zero errors when spliced in
as the node's own real proof via `check_blueprint()` on the full assembled
blueprint - i.e. it's a valid direct proof of the stated lemma, not a
disproof of anything.

**Why not fixed in-session**: a correct fix needs to actually construct
`¬ (conclusion)` for arbitrary Lean statement shapes (quantifiers, equalities,
existing negations/implications) and check the model's proof against THAT
- a nontrivial parsing/generation problem, not a one-line parameter fix.
Flagged for separate, careful follow-up rather than a blind patch.

**Workaround used**: same manual-injection-into-proved_cache technique as
`Tree.balance_BST` earlier this session (see `vsb-balance-bst-progress`
memory) - verify the "disproven" statement's truth independently via
`check_blueprint()`, then hand-write `proved_cache`/`proof_cache_keys`
entries to correct the record.

# lean-hoare test log

First test of the pipeline against a brand-new VeriSoftBench repo
(`lean-hoare`), chosen as a small/easy repo distinct from the
`lean-formal-reasoning-program` sweep in `results/single_test/`. All 4
theorems in this repo, all from `Hoare/While/Types.lean` (a small, single-file
toy `While`-language type-checker). Each phase run separately with
`--workers 1`.

| theorem | nodes | status | notes |
|---|---|---|---|
| While.WellTyped.ty_some | 6 | DONE (proved) | Single pass, zero refinement rounds needed |
| While.WellTyped.not_eq_not_eq_ty | 6 | DONE (proved) | Single pass, zero refinement rounds needed |
| While.WellTyped.not_welltyped_not_eq_ty | 5 | DONE (proved) | Single pass, zero refinement rounds needed |
| While.WellTyped.some_ty | 17 | DONE (proved) | 2 refinement rounds - first pass got stuck on the `some_ty_boolean_binary_cases` catch-all node (comparison operators lumped together), refinement split it into per-operator sub-lemmas (`lt`/`ge`/etc.) which then solved individually |

4/4 proved. All theorems checkpointed in `checkpoints/`; extracted proofs not
yet generated (reuse `results/single_test/extract_proof.py`'s pattern with
`--checkpoint-dir results/lean_hoare/checkpoints` if needed).

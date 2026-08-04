# Blueprint-guided COT refinement

This experiment converts post-`</think>` math solutions into RobustPA-style
informal statement/proof rows, runs the existing Goedel blueprint pipeline,
exports machine-checked solved proofs plus diagnostics for unresolved nodes,
and asks a model to refine the original solution.

Run the two-record live smoke test (`run_all.sh` defaults to this profile):

```bash
PROFILE=smoke bash experiments/cot_blueprint_refine/run_all.sh
```

Run or resume one stage:

```bash
PYTHONPATH=src:experiments /ssd/miniconda3/envs/lean4-czx/bin/python \
  experiments/cot_blueprint_refine/run_experiment.py \
  --profile smoke --stage export
```

The base profile prepares all eligible records. The smoke profile fixes both
the blueprint/prover and COT-refinement endpoints to the 397B model at port
8001. Gold answers are loaded only by the final evaluation stage.

For the later fair run, start the Qwen3-8B refinement service on port 7999
with `--max-model-len 40960`, then use `PROFILE=base`. The 397B smoke server
currently advertises a 65536-token context and therefore needs no context
override for the two-record test.

Outputs are isolated under
`/ssd/czx/czx_work/cot_blueprint_refine/<exp_name>/`. A per-experiment lock
rejects concurrent writers so checkpoints and append-only result files cannot
be interleaved.

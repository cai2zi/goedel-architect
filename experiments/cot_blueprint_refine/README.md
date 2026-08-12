# Blueprint-guided COT refinement

The COT-to-Blueprint stage has one implementation: every round regenerates a
complete Step-grounded Lean Blueprint. There are no skeleton, Pending,
node-edit, Planner, seed, or search strategies.

Each generation round receives the original problem, claimed answer, all COT
Steps, the previous complete Blueprint, and the previous round's complete
diagnostic inventory. A candidate is checked by:

1. Blueprint parsing, root/Step metadata, dependency DAG, and canonical rebuild;
2. whole-file Lean and Phase 2 standalone compilation;
3. high-confidence deterministic semantic checks;
4. the Formal Decompiler and Strict Comparator.

An error-free and warning-free candidate exits immediately. Warnings keep the
loop running; only the final configured round may return
`acceptedWithWarnings`. Deterministic or semantic errors at the final round
produce `structuralRejected` or `semanticRejected`.

## Wrong-answer 76 experiment

Start external Kimina on port 8000 and the existing 397B vLLM on port 8001,
then run:

```bash
bash experiments/cot_blueprint_refine/script/qwen3_8b_397b_wrong76_blueprint_generation.sh
```

The script performs `prepare`, Step splitting, and Blueprint generation. Its
profile is:

```text
experiments/cot_blueprint_refine/configs/qwen3_8b_397b_wrong76_blueprint_generation.yaml
```

The profile uses 8 full-regeneration rounds, thinking with temperature 0.6,
the 40,960-token context, dynamic remaining completion budget, strict Step
binding, and 76-way record concurrency.

After that Phase 1 run has finished, prove every accepted Blueprint without
regenerating it or entering Phase 3:

```bash
bash experiments/cot_blueprint_refine/script/qwen3_8b_397b_wrong76_phase2.sh
```

The Phase 2-only runner reads the existing checkpoints in the same experiment
directory. Phase 1 rejects remain unchanged; only `strictAccepted` and
`acceptedWithWarnings` rows are sent to the node prover.

The 397B node prover explicitly enables thinking and uses temperature 0.6,
top-p 0.95, top-k 20, min-p 0, presence penalty 0, and repetition penalty 1.
The Formal Decompiler and Strict Comparator remain deterministic non-thinking
requests.

## Full 646-row experiment

The original Qwen3-8B prediction file contains 660 rows. The existing prepare
policy selects all 646 rows with `finish_reason=stop` and excludes the 14
length-truncated rows. Run the complete eligible set with:

```bash
bash experiments/cot_blueprint_refine/script/qwen3_8b_397b_all646_blueprint_generation.sh
```

This profile uses 512-way record/splitter concurrency while retaining the
global 48-snippet Lean inflight limit. It attaches to the existing vLLM and
Kimina services and does not manage their lifecycle.

## Generic runner

Run the two-record smoke profile:

```bash
PROFILE=smoke bash experiments/cot_blueprint_refine/run_all.sh
```

Useful stage sequences are:

```text
phase1-only       prepare + split + Blueprint generation
cot-to-blueprint  prepare + split + Blueprint generation + export
blueprint-refine  prepare + split + Blueprint generation + export + COT refine
all               all stages, including evaluation
```

Outputs are isolated under
`/ssd/czx/czx_work/cot_blueprint_refine/<exp_name>/`. A per-experiment lock
prevents concurrent writers from interleaving checkpoints or JSONL results.

Phase 2 proving and Phase 3 COT refinement remain separate downstream stages.
Mathlib search remains available to the node prover; it is intentionally not
part of Blueprint generation.

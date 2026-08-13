# Blueprint Prover Model Experiments

This experiment proves the 45 immutable `strictAccepted` Whole-COT Blueprints
from `qwen3_8b_397b_wrong76_whole_cot_blueprint_generation_thinking_judge`.
It preserves the RobustPA active dependency DAG, ancestor proof context,
dependency blocking, final assembly, and whole-file verification. Per-node
proof search uses one of two model-native protocols:

- StepFun uses iterative `<sketch>` / Lean `<REPL>` interaction.
- Goedel-Prover-V2 uses its model-card `Complete the following Lean 4 code`
  prompt, then at most two verifier-guided correction rounds with real Lean
  diagnostics. It returns a proof plan and a complete Lean fenced block.

Positive proof search finishes globally before negative probing begins. Every
actually attempted, non-infrastructure positive failure receives an independent
negative proof attempt. A negative attempt proves the exact negation of the
entire closed theorem: all node parameters and hypotheses are moved under `∀`
before the proposition is negated, so a counterexample is sufficient. Positive
and negative trajectories each have their own
40,960-token input-plus-output context budget. There is no turn, HTTP, or node
wall-time limit; Kimina's required server timeout field uses an effectively
unbounded one-day value.

## Start one vLLM model on all eight GPUs

```bash
cd /ssd/czx/goedel-architect
mkdir -p /ssd/czx/czx_work/stepfun_blueprint_prover/vllm
nohup experiments/stepfun_blueprint_prover/launch_vllm_8gpu.sh \
  > /ssd/czx/czx_work/stepfun_blueprint_prover/vllm/server.log 2>&1 &
echo $! > /ssd/czx/czx_work/stepfun_blueprint_prover/vllm/server.pid
```

The command above starts StepFun-7B as `DP=2 x TP=4`. Additional launchers are:

```bash
# StepFun 32B: TP=8
experiments/stepfun_blueprint_prover/launch_stepfun_32b_8gpu.sh

# Goedel V2 8B: DP=8 x TP=1
experiments/stepfun_blueprint_prover/launch_goedel_v2_8b_8gpu.sh

# Goedel V2 32B: TP=8
experiments/stepfun_blueprint_prover/launch_goedel_v2_32b_8gpu.sh
```

Only one model should own port 8001 at a time. Each new launcher checks every
weight shard before allocating GPUs. At the time these scripts were added,
StepFun-32B and Goedel-32B were each missing two local shards, so their launchers
will fail early until those downloads complete.

All experiment profiles use `api_concurrency: 1024`, and the shared vLLM
launcher explicitly sets `--max-num-seqs 1024` per engine.

## Run

```bash
experiments/stepfun_blueprint_prover/run_smoke.sh
experiments/stepfun_blueprint_prover/run_full.sh

experiments/stepfun_blueprint_prover/run_stepfun_32b_full.sh
experiments/stepfun_blueprint_prover/run_goedel_v2_8b_full.sh
experiments/stepfun_blueprint_prover/run_goedel_v2_32b_full.sh
```

Each additional model also has a matching `*_smoke.sh` launcher. Smoke output
is isolated from full-run checkpoints.

To start vLLM, wait for readiness, run each model, stop it, and then move to the
next model serially, use:

```bash
# Default: one Blueprint each for StepFun-7B and Goedel-V2-8B.
experiments/stepfun_blueprint_prover/run_models_serial.sh

# Full four-model run after both 32B downloads finish; small models run first.
experiments/stepfun_blueprint_prover/run_models_serial.sh --full --all-models
```

The serial launcher refuses to replace a service already listening on port
8001. Its vLLM logs and owned process IDs are stored under
`/ssd/czx/czx_work/stepfun_blueprint_prover/vllm_serial`.

The runner submits every currently dependency-ready node immediately. vLLM
performs continuous batching; one long trajectory never creates a batch barrier.
It displays separate resume-aware progress bars for positive nodes, negative
nodes, and final Blueprint verification.

## Recompute summary

```bash
/ssd/miniconda3/envs/lean4-czx/bin/python \
  experiments/stepfun_blueprint_prover/summarize_results.py \
  /ssd/czx/czx_work/stepfun_blueprint_prover/stepfun_7b_wrong76_whole_cot_closed_negation \
  --source-root \
  /ssd/czx/czx_work/cot_blueprint_refine/qwen3_8b_397b_wrong76_whole_cot_blueprint_generation_thinking_judge/robustpa/blueprint
```

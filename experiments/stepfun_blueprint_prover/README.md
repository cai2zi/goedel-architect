# StepFun Blueprint Prover

This experiment proves the 45 immutable `strictAccepted` Whole-COT Blueprints
from `qwen3_8b_397b_wrong76_whole_cot_blueprint_generation_thinking_judge`.
It preserves the RobustPA active dependency DAG, ancestor proof context,
dependency blocking, final assembly, and whole-file verification. Per-node
proof search is replaced by the native StepFun `<sketch>` / Lean `<REPL>` loop.

Positive proof search finishes globally before negative probing begins. Every
actually attempted, non-infrastructure positive failure receives an independent
negative proof attempt. A negative attempt proves the exact negation of the
entire closed theorem: all node parameters and hypotheses are moved under `∀`
before the proposition is negated, so a counterexample is sufficient. Positive
and negative trajectories each have their own
40,960-token input-plus-output context budget. There is no turn, HTTP, or node
wall-time limit; Kimina's required server timeout field uses an effectively
unbounded one-day value.

## Start vLLM on all eight GPUs

```bash
cd /ssd/czx/goedel-architect
mkdir -p /ssd/czx/czx_work/stepfun_blueprint_prover/vllm
nohup experiments/stepfun_blueprint_prover/launch_vllm_8gpu.sh \
  > /ssd/czx/czx_work/stepfun_blueprint_prover/vllm/server.log 2>&1 &
echo $! > /ssd/czx/czx_work/stepfun_blueprint_prover/vllm/server.pid
```

The model has 28 attention heads, so eight GPUs are used as two data-parallel
replicas, each with tensor parallelism four (`DP=2 x TP=4`).

## Run

```bash
experiments/stepfun_blueprint_prover/run_smoke.sh
experiments/stepfun_blueprint_prover/run_full.sh
```

The runner submits every currently dependency-ready node immediately. vLLM
performs continuous batching; one long trajectory never creates a batch barrier.

## Recompute summary

```bash
/ssd/miniconda3/envs/lean4-czx/bin/python \
  experiments/stepfun_blueprint_prover/summarize_results.py \
  /ssd/czx/czx_work/stepfun_blueprint_prover/stepfun_7b_wrong76_whole_cot \
  --source-root \
  /ssd/czx/czx_work/cot_blueprint_refine/qwen3_8b_397b_wrong76_whole_cot_blueprint_generation_thinking_judge/robustpa/blueprint
```

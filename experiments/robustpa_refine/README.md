# RobustPA Kimina-Only Refine

This experiment uses only each RobustPABench row's `informal_statement` and
`informal_proof`. The formal statement and proof in the dataset are not passed
to the pipeline.

Both fields are persisted in the checkpoint. Every Phase-3 blueprint
refinement prompt includes the original `informal_proof`/COT alongside the
natural-language problem, current annotated blueprint, and prior-round trace.

## Services

Kimina must expose `POST /api/check` at the configured `lean_api_url` (default
`http://127.0.0.1:8000`). The OpenAI-compatible model endpoint is configured
separately (the examples use `http://127.0.0.1:8001/v1`).

The Kimina image/environment must include Mathlib, `Architect`, `sorry_using`,
and `#validate_blueprint`. No local Lean compiler is used by this repository.

## Run One Record

The runner uses Hydra overrides, not argparse flags:

```bash
/ssd/miniconda3/envs/lean4-czx/bin/python \
  experiments/robustpa_refine/run_robustpa_refine.py \
  exp_name=kimina_smoke \
  subset=global_original \
  split=MATH500 \
  problem_id=test_prealgebra_1924 \
  model=Qwen3.5-397B-A17B-FP8 \
  openai_base_url=http://127.0.0.1:8001/v1
```

Important defaults in `configs/base.yaml`:

```yaml
max_refinement_iterations: 4
node_max_prove_turns: 8
node_max_negation_probe_turns: 1  # 0 disables the probe
max_tool_calls_per_turn: 3
lean_max_inflight_snippets: 128
lean_batch_size: 8
```

The negation probe runs for at most `node_max_negation_probe_turns` Lean-guided
turns and stops on the first successful proof. Set it to `0` to disable the
probe. `node_timeout_s` and `llm_api_timeout_s` may be `null` for local
long-running inference.

## Proof Protocol

Normal turns expose `lean_compile`, `step_lean_compile`, and `mathlib_search`.
Within a turn, valid calls are deduplicated and capped before the assistant
message is added to history. Dropped calls produce one aggregate trace event
and no ignored tool messages. Lean calls are sent in one Kimina batch while
searches run concurrently; tool results are restored to original call order.

`step_lean_compile` accepts a complete standalone Lean file and cannot solve a
node. The final proving turn exposes only `lean_compile` with required tool
choice. Cross-turn duplicate calls reuse the node-session cache.

Only a successful no-sorry Kimina compile of the assembled root dependency
closure sets `status=solved` and `root_proved=true`. Refinement exhaustion is
recorded as `exhausted`; Kimina infrastructure failure is `error`.

## Outputs

Outputs are written under
`/ssd/czx/czx_work/robustpa_refine/<exp_name>` by default:

- `results.jsonl`: one terminal result per selected record.
- `rounds.jsonl`: phase and refinement audit rows.
- `metrics.json`, `metrics.csv`: aggregate and grouped metrics.
- `checkpoints/`: current-schema checkpoints.
- `traces/`: tool, node, and final verification JSONL traces.
- `blueprints/`: blueprint snapshots by iteration.

`final_verify` appears only for the assembled root closure compilation;
individual nodes finish with `node_finished`. Existing checkpoints from older
schemas are deliberately unsupported.

## Inspect Results

```bash
/ssd/miniconda3/envs/lean4-czx/bin/python \
  experiments/robustpa_refine/summarize_for_chat.py \
  /ssd/czx/czx_work/robustpa_refine/<exp_name>

/ssd/miniconda3/envs/lean4-czx/bin/python \
  experiments/robustpa_refine/trace_viewer.py \
  /ssd/czx/czx_work/robustpa_refine/<exp_name> \
  --port 8765
```

StmtSC export and evaluation remain available through
`export_stmt_sc_inputs.py` and `evaluate_stmt_sc.py`; semantic agreement with
the original natural-language statement is judged there rather than by
forbidding root declaration changes during Phase 3.

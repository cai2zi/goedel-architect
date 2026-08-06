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

The runner automatically starts and stops a stage-owned vLLM service for
`blueprint`, `refine`, and the final answer-equivalence judge. `prepare` and
`export` do not use vLLM. Startup requires an exclusive configured port; an
existing listener is rejected so the runner never kills or reconfigures a
service it does not own. Logs and lifecycle metadata are written below the
experiment output in `vllm/`.

The base profile uses the 397B model at port 8001 for blueprint/proving and
judge, and Qwen3-8B at port 7999 for refinement. The smoke profile uses the
397B model for all three model stages. Each model stage gets a fresh service,
including when consecutive stages use the same model.

Automatic lifecycle management is enabled by default. To use a manually
managed endpoint, disable startup and destruction:

```bash
PROFILE=smoke bash experiments/cot_blueprint_refine/run_all.sh \
  vllm.auto_start=false vllm.auto_destroy=false
```

`vllm.auto_destroy=false` leaves a service started by a single-stage run alive.
Because automatic startup uses exclusive ports, this mode is intended for a
single stage rather than `--stage all`.

Service settings can be overridden with the same dot-list syntax, for example
`vllm.cuda_visible_devices=0,1,2,3 blueprint.vllm.tensor_parallel_size=4`.
When changing a stage's host, port, or served name, update both its `*.vllm`
settings and matching `openai_base_url`/`model`; the runner validates that the
server and client configurations agree before startup.

Gold answers are loaded only by `evaluate`. Math-Verify grades only the
canonical last `\boxed{...}` / `claimed_answer` field; whole-COT `any_match`
parsing is retained in diagnostic columns but never determines correctness.
If Math-Verify does not accept an available answer, the default 397B judge
compares that same answer with gold using the problem as context. Final
correctness is `math_verify_correct OR judge_equivalent`.

Successful judge cache keys are resumed from `evaluation/judge_results.jsonl`;
only failed keys are retried. Every call records an `attempt_log` containing
the error layer, HTTP request id, raw HTTP body, assistant content, and parsed
response snapshot. In particular, SDK/API response decoding failures are
separate from failures of `json.loads` on assistant content. Set
`judge.enabled=false` to retain Math-Verify-only evaluation.

COT refinement reserves 20,480 output tokens. With the Qwen3-8B 40,960-token
context and 256-token safety margin this leaves 20,224 tokens for input.

Outputs are isolated under
`/ssd/czx/czx_work/cot_blueprint_refine/<exp_name>/`. A per-experiment lock
rejects concurrent writers so checkpoints and append-only result files cannot
be interleaved.

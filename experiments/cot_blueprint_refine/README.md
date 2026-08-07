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

Composite stages keep the owned vLLM/Kimina processes alive across the sequence:

```bash
# prepare + RobustPA + export; stop before natural-language refinement
--stage cot-to-blueprint

# prepare + RobustPA + export + enabled refinement variants; no evaluation
--stage blueprint-refine
```

To focus on source COTs whose canonical final boxed answer is wrong:

```bash
# Extract the strict math_verify subset and stop after COT -> Blueprint.
bash experiments/cot_blueprint_refine/script/qwen3_8b_397b_cot_incorrect_blueprint.sh

# Resume the same experiment and additionally run the blueprint refinement arm.
PIPELINE_STAGE=blueprint-refine \
  bash experiments/cot_blueprint_refine/script/qwen3_8b_397b_cot_incorrect_blueprint.sh
```

The subset extractor scores only the last post-thinking `\boxed{...}`; it does not
use the historical whole-COT `is_correct` field. The dedicated profile disables
the `cot_only` refinement arm and Judge.

For `--stage all`, the runner starts one experiment-owned 397B vLLM service
after `prepare`, keeps it resident through blueprint generation, export, both
refinement arms, and the final answer-equivalence judge, then stops it once.
Single model-stage runs still start and stop one process. Startup requires an
exclusive configured port; an existing listener is rejected so the runner
never kills or reconfigures a service it does not own. PID, stage attachment,
reuse, start, and stop metadata are written below the experiment output in
`vllm/`.

Blueprint/proving, COT refinement, and judge all use
`Qwen3.5-397B-A17B-FP8` at port 8001 with one shared service definition. The
refinement stage runs `blueprint` and `cot_only` arms with identical decoding
settings. The latter receives only the original problem, claimed answer, and
post-thinking source-model solution; it receives no Lean or blueprint data.

Run the Qwen3-8B → 397B ablation from a clean output directory with:

```bash
PYTHONPATH=src:experiments /ssd/miniconda3/envs/lean4-czx/bin/python \
  experiments/cot_blueprint_refine/run_experiment.py \
  --profile qwen3_8b_397b_refine_ablation --stage all
```

Automatic lifecycle management is enabled by default. To use a manually
managed endpoint, disable startup and destruction:

```bash
PROFILE=smoke bash experiments/cot_blueprint_refine/run_all.sh \
  vllm.auto_start=false vllm.auto_destroy=false
```

`vllm.auto_destroy=false` leaves an automatically started process alive after
the run. Because automatic startup uses exclusive ports, manually managed
services should use `vllm.auto_start=false`.

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

COT refinement reserves 20,480 output tokens. With the configured 65,536-token
397B context and 256-token safety margin this leaves 44,800 tokens for input.
Artifacts are isolated under `refinement/<variant>/` and
`evaluation/<variant>/`; paired arm outcomes and complete-denominator metrics
are written under `evaluation/ablation/`.

Outputs are isolated under
`/ssd/czx/czx_work/cot_blueprint_refine/<exp_name>/`. A per-experiment lock
rejects concurrent writers so checkpoints and append-only result files cannot
be interleaved.

# RobustPA Informal-Only Refine

Runs Goedel-Architect on RobustPABench using only `informal_statement` and
`informal_proof`. The experiment ignores `formal_statement` and `formal_proof`.

Lean checks use a manually managed Kimina server by default:

```bash
cd /ssd/czx/kimina-lean-server
/ssd/miniconda3/envs/lean4-czx/bin/python -m server
```

The default config points RobustPA refine at `http://localhost:8000`.

```bash
conda activate lean4-czx
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve /ssd/czx/models/Qwen3.5-397B-A17B-FP8 \
  --served-model-name Qwen3.5-397B-A17B-FP8 \
  --host 0.0.0.0 \
  --port 8001 \
  --tensor-parallel-size 8 \
  --max-model-len 65536 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.95 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```
Example:


```bash
python experiments/robustpa_refine/run_robustpa_refine.py \
  --exp-name vllm_qwen_run1 \
  --model <vllm-model-name> \
  --openai-base-url http://localhost:8000/v1 \
  --resume
```

Useful smoke test:

```bash
python experiments/robustpa_refine/run_robustpa_refine.py \
  --exp-name smoke \
  --subset global_original \
  --split miniF2F \
  --limit 1 \
  --model <vllm-model-name> \
  --openai-base-url http://localhost:8000/v1 \
  --llm-api-timeout-s none
```

`--llm-api-timeout-s` controls the timeout for one OpenAI-compatible LLM HTTP
request. `--node-timeout-s` controls the whole Phase2 proof loop for one node,
which may contain multiple LLM calls and Lean tool calls. RobustPA defaults
both to `null` for local vLLM runs. Passing `0`, `none`, or `null` on the
command line has the same effect.

`parallel_tool_calls=N` controls the per-turn Phase2 tool-call budget. `1`
asks the provider for at most one tool call per assistant message and processes
only one; values greater than `1` allow provider-side parallel tool calls, but
the runner executes only the first `N` tool calls from a turn and ignores the
rest. `node_max_prove_turns` controls how many assistant responses are allowed
in the normal proving loop, and `node_max_negation_probe_turns` controls the
same budget for the negation probe after proving fails. Setting
`node_max_negation_probe_turns: 0` disables the negation probe. The default
config sets `parallel_tool_calls: 4`, `node_max_prove_turns: 8`, and
`node_max_negation_probe_turns: 4`.

Subset filtering:

`--subset` may be passed more than once. If omitted, the runner uses every
subset under the data root that contains parquet files. Current available
subsets in `/ssd/czx/czx_work/RobustPABench` are:

- `global_gemini_rephrase`
- `global_gemini_step`
- `global_original`
- `global_qwen3_rephrase`
- `global_qwen3_step`
- `local_number_edit_proof`
- `local_number_edit_statement`
- `local_step_delete`
- `local_symbol_edit_proof`
- `local_symbol_edit_statement`

Outputs are written to `/ssd/czx/czx_work/robustpa_refine/<exp_name>` by
default:

- `results.jsonl`: one final row per selected sample.
- `rounds.jsonl`: one audit row per phase/iteration, including node verdicts.
- `metrics.json` and `metrics.csv`: global, subset, split, and subset/split
  accuracy summaries.
- `checkpoints/`, `traces/`, `blueprints/`: resumable state and per-round Lean
  blueprint snapshots.

Build a compact one-file bundle for web-chat analysis:

```bash
python experiments/robustpa_refine/summarize_for_chat.py \
  /ssd/czx/czx_work/robustpa_refine/<exp_name> \
  --output /ssd/czx/czx_work/robustpa_refine/<exp_name>/chat_bundle.md
```

The bundle summarizes `metrics.json`, `results.jsonl`, `rounds.jsonl`, and
`traces/**/*.jsonl` without pasting raw trace logs. Use `--problem-rows all` if
you want compact rows for every selected problem rather than failures only.

Browse per-problem trace conversations in a local web UI:

```bash
python experiments/robustpa_refine/trace_viewer.py \
  /ssd/czx/czx_work/robustpa_refine/<exp_name> \
  --port 8765
```

Open the printed URL, enter a `source_id`, `record_id`, or full unique id, and
optionally filter by node name.

## RobustPABench-Aligned StmtSC

The runner intentionally keeps inference output in Goedel's native checkpoint
format. To evaluate statement semantic consistency with the same StmtSC prompt
and parser as `/ssd/czx/robust-proof-autoformalization`, export the final
checkpoints first, then run the StmtSC judge.

Artifacts are written under `/ssd/czx/czx_work/robustpa_eval/<exp_name>`:

- `stmt_sc_inputs.jsonl` — RobustPABench-style rows with `LLM_Output#1`.
- `stmt_sc_scored.jsonl` — rows enriched with `LLM_StmtSC?#1` and details.
- `stmt_sc_summary.json` — overall and grouped StmtSC accuracy.
- `lean_outputs/<subset>/<split>/<record_id>.lean` — assembled final/partial
  Lean files reconstructed from checkpoints.

Example:

```bash
python experiments/robustpa_refine/export_stmt_sc_inputs.py \
  --exp-dir /ssd/czx/czx_work/robustpa_refine/qwen3_5_397b_MiniF2F_orig_reExp

python experiments/robustpa_refine/evaluate_stmt_sc.py \
  --exp-name qwen3_5_397b_MiniF2F_orig_reExp \
  --gemini-model gemini-2.5-flash
```

The exporter evaluates only the final checkpoint for each problem. It includes
all final `results.jsonl` rows in the denominator: solved, exhausted, and error
rows. If a checkpoint cannot be assembled or no theorem can be extracted, the
StmtSC scorer records `LLM_StmtSC?#1 = "no"` and reports the reason in the
summary counts.

# RobustPA Informal-Only Refine

Runs Goedel-Architect on RobustPABench using only `informal_statement` and
`informal_proof`. The experiment ignores `formal_statement` and `formal_proof`.


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
  --llm-api-timeout-s 600 \
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
  --llm-api-timeout-s 600
```

`--llm-api-timeout-s` controls the timeout for one OpenAI-compatible LLM HTTP
request. `--node-timeout-s` controls the whole Phase2 proof loop for one node,
which may contain multiple LLM calls and Lean tool calls.

Outputs are written to `/ssd/czx/czx_work/robustpa_refine/<exp_name>` by
default:

- `results.jsonl`: one final row per selected sample.
- `rounds.jsonl`: one audit row per phase/iteration, including node verdicts.
- `metrics.json` and `metrics.csv`: global, subset, split, and subset/split
  accuracy summaries.
- `checkpoints/`, `traces/`, `blueprints/`: resumable state and per-round Lean
  blueprint snapshots.

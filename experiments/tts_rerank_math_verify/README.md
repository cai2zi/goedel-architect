# TTS Rerank Math-Verify

Runs one-pass Goedel-Architect scoring over rollout answers in `czx_work/bench.json`.
The experiment uses Math-Verify labels already stored in the bench file; it does not
rerun Math-Verify.

Example:

```bash
export OPENAI_BASE_URL=https://poloai.top/v1
export GOEDEL_OPENAI_BASE_URL=https://poloai.top/v1
export GOEDEL_BLUEPRINT_MAX_TOKENS=262144
export GOEDEL_TOOL_CHOICE_MODE=auto
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897

python experiments/tts_rerank_math_verify/run_tts_rerank.py \
  --model deepseek-v4-flash \
  --limit 1 \
  --problem-id aime25_test__0 \
  --rollout-id 1
```

Outputs are written by default to:

```text
czx_work/goedel-architect/tts_rerank_math_verify/deepseek_v4_flash/
```

Main outputs:

- `phase0_results.jsonl`
- `rollout_scores.jsonl`
- `goedel_best.jsonl`
- `metrics.json`
- `metrics.csv`
- per-rollout checkpoints, traces, and generated blueprints

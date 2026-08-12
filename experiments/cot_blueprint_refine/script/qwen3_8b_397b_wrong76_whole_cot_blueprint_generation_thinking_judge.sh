#!/usr/bin/env bash
set -euo pipefail
cd /ssd/czx/goedel-architect

curl -fsS http://127.0.0.1:8000/health >/dev/null
curl -fsS http://127.0.0.1:8001/v1/models >/dev/null

exec /ssd/miniconda3/envs/lean4-czx/bin/python \
  experiments/cot_blueprint_refine/run_experiment.py \
  --profile qwen3_8b_397b_wrong76_whole_cot_blueprint_generation_thinking_judge \
  --stage whole-cot-phase1-only "$@"

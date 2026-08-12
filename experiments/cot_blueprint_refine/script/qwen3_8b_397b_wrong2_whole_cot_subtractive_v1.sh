#!/usr/bin/env bash
set -euo pipefail
cd /ssd/czx/goedel-architect

curl -fsS http://127.0.0.1:8000/health >/dev/null

exec /ssd/miniconda3/envs/lean4-czx/bin/python \
  experiments/cot_blueprint_refine/run_experiment.py \
  --profile qwen3_8b_397b_wrong2_whole_cot_subtractive_v1 \
  --stage phase1-only "$@"

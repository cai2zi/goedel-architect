#!/usr/bin/env bash
set -euo pipefail

cd /ssd/czx/goedel-architect
exec /ssd/miniconda3/envs/lean4-czx/bin/python \
  experiments/cot_blueprint_refine/run_experiment.py \
  --profile qwen3_8b_397b_wrong76_step_v11_direct_edit_search_turn8 \
  --stage phase1-only "$@"

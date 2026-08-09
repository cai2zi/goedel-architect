#!/usr/bin/env bash
set -euo pipefail

cd /ssd/czx/goedel-architect
exec /ssd/miniconda3/envs/lean4-czx/bin/python \
  experiments/cot_blueprint_refine/run_experiment.py \
  --profile qwen3_8b_397b_wrong76_step_v3_phase1_ab_crud_standalone_smoke10 \
  --stage phase1-only "$@"

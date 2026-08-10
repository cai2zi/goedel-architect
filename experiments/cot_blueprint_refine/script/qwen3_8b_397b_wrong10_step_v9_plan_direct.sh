#!/usr/bin/env bash
set -euo pipefail

cd /ssd/czx/goedel-architect
exec /ssd/miniconda3/envs/lean4-czx/bin/python \
  experiments/cot_blueprint_refine/run_experiment.py \
  --profile qwen3_8b_397b_wrong10_step_v9_plan_direct \
  --stage blueprint "$@"

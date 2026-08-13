#!/usr/bin/env bash
set -euo pipefail
exec /ssd/czx/goedel-architect/experiments/cot_blueprint_refine/script/run_semantic_ablation_experiment.sh \
  qwen3_8b_397b_all646_subtractive_separate_t06 \
  qwen3_8b_397b_all646_subtractive_separate_t06 "$@"

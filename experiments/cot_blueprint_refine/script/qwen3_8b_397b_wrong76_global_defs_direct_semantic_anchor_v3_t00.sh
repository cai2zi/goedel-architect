#!/usr/bin/env bash
set -euo pipefail
exec /ssd/czx/goedel-architect/experiments/cot_blueprint_refine/script/run_semantic_ablation_experiment.sh \
  qwen3_8b_397b_wrong76_global_defs_direct_semantic_anchor_v3_t00 \
  qwen3_8b_397b_wrong76_global_defs_direct_semantic_anchor_v3_t00 "$@"

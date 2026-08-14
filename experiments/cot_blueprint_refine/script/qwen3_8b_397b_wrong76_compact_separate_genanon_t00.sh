#!/usr/bin/env bash
set -euo pipefail
exec /ssd/czx/goedel-architect/experiments/cot_blueprint_refine/script/run_semantic_ablation_experiment.sh \
  qwen3_8b_397b_wrong76_compact_separate_genanon_t00 \
  qwen3_8b_397b_wrong76_compact_separate_genanon_t00 "$@"

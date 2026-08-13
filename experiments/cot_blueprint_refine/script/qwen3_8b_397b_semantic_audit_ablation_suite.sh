#!/usr/bin/env bash
set -euo pipefail
cd /ssd/czx/goedel-architect
exec /ssd/miniconda3/envs/lean4-czx/bin/python \
  experiments/cot_blueprint_refine/run_semantic_ablation_suite.py "$@"

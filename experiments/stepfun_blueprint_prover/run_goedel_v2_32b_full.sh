#!/usr/bin/env bash
set -euo pipefail
cd /ssd/czx/goedel-architect
exec /ssd/miniconda3/envs/lean4-czx/bin/python \
  experiments/stepfun_blueprint_prover/run_experiment.py \
  --config experiments/stepfun_blueprint_prover/configs/goedel_v2_32b_wrong76_whole_cot.yaml "$@"

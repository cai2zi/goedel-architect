#!/usr/bin/env bash
set -euo pipefail
cd /ssd/czx/goedel-architect
exec /ssd/miniconda3/envs/lean4-czx/bin/python \
  experiments/stepfun_blueprint_prover/run_experiment.py \
  --config experiments/stepfun_blueprint_prover/configs/goedel_v2_32b_wrong76_whole_cot.yaml \
  --limit "${BLUEPRINT_SMOKE_LIMIT:-1}" \
  --output-root "${BLUEPRINT_SMOKE_ROOT:-/ssd/czx/czx_work/stepfun_blueprint_prover/goedel_v2_32b_smoke_closed_negation}" "$@"

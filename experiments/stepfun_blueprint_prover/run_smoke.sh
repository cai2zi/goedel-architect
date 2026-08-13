#!/usr/bin/env bash
set -euo pipefail
cd /ssd/czx/goedel-architect

exec /ssd/miniconda3/envs/lean4-czx/bin/python \
  experiments/stepfun_blueprint_prover/run_experiment.py \
  --limit "${STEPFUN_SMOKE_LIMIT:-2}" \
  --output-root "${STEPFUN_OUTPUT_ROOT:-/ssd/czx/czx_work/stepfun_blueprint_prover/smoke}" \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/ssd/czx/goedel-architect"
PYTHON_BIN="/ssd/miniconda3/envs/lean4-czx/bin/python"
PROFILE="qwen3_8b_397b_wrong76_claim_scope"
STAGE="${STAGE:-blueprint-refine}"

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" experiments/cot_blueprint_refine/run_experiment.py \
  --profile "${PROFILE}" \
  --stage "${STAGE}" \
  "$@"

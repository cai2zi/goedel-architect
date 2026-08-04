#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/ssd/miniconda3/envs/lean4-czx/bin/python}"
PROFILE="${PROFILE:-base}"
STAGE="${STAGE:-all}"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/experiments${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" experiments/cot_blueprint_refine/run_experiment.py \
  --profile "${PROFILE}" \
  --stage "${STAGE}" \
  "$@"
# STAGE=export ./experiments/cot_blueprint_refine/run_all.sh
# STAGE=refine ./experiments/cot_blueprint_refine/run_all.sh
# STAGE=evaluate ./experiments/cot_blueprint_refine/run_all.sh
# STAGE=all ./experiments/cot_blueprint_refine/run_all.sh
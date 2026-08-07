#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/ssd/miniconda3/envs/lean4-czx/bin/python}"
PIPELINE_STAGE="${PIPELINE_STAGE:-cot-to-blueprint}"
INPUT_PREDICTIONS="${INPUT_PREDICTIONS:-/ssd/czx/czx_work/math_verify_eval/qwen3_8b_math_verify/predictions.jsonl}"
SUBSET_DIR="${SUBSET_DIR:-/ssd/czx/czx_work/cot_blueprint_refine/subsets/qwen3_8b_original_answer_incorrect}"
SUBSET_PREDICTIONS="${SUBSET_PREDICTIONS:-${SUBSET_DIR}/predictions.jsonl}"

case "${PIPELINE_STAGE}" in
  cot-to-blueprint|blueprint-refine) ;;
  *)
    echo "PIPELINE_STAGE must be cot-to-blueprint or blueprint-refine" >&2
    exit 2
    ;;
esac

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/experiments${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"
"${PYTHON_BIN}" experiments/cot_blueprint_refine/extract_original_incorrect_subset.py \
  --input "${INPUT_PREDICTIONS}" \
  --output "${SUBSET_PREDICTIONS}" \
  --metrics "${SUBSET_DIR}/metrics.json"

exec env PROFILE=qwen3_8b_397b_cot_incorrect_blueprint \
  STAGE="${PIPELINE_STAGE}" \
  "${EXPERIMENT_DIR}/run_all.sh" \
  "input_predictions=${SUBSET_PREDICTIONS}" \
  "$@"

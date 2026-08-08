#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/ssd/miniconda3/envs/lean4-czx/bin/python}"
PROFILE="${PROFILE:-qwen3_8b_397b_wrong76_semantic_matrix}"
MATRIX_MODE="${MATRIX_MODE:-all}"
RUN_ID="${RUN_ID:-sem_v1}"
RUN_REFINE="${RUN_REFINE:-key}"
MATRIX_ARMS="${MATRIX_ARMS:-}"
DRY_RUN="${DRY_RUN:-0}"
REPORT_ONLY="${REPORT_ONLY:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

case "${MATRIX_MODE}" in
  add|reduce|all) ;;
  *)
    echo "MATRIX_MODE must be add, reduce, or all" >&2
    exit 2
    ;;
esac

case "${RUN_REFINE}" in
  key|all|none) ;;
  *)
    echo "RUN_REFINE must be key, all, or none" >&2
    exit 2
    ;;
esac

args=(
  --profile "${PROFILE}"
  --mode "${MATRIX_MODE}"
  --run-id "${RUN_ID}"
  --refine "${RUN_REFINE}"
)
if [[ -n "${MATRIX_ARMS}" ]]; then
  args+=(--arms "${MATRIX_ARMS}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  args+=(--dry-run)
fi
if [[ "${REPORT_ONLY}" == "1" ]]; then
  args+=(--report-only)
fi
if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
  args+=(--continue-on-error)
fi

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/experiments${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" experiments/cot_blueprint_refine/run_semantic_matrix.py \
  "${args[@]}" \
  "$@"

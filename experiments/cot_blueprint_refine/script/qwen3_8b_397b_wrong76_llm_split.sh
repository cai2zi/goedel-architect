#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/ssd/czx/goedel-architect"
PYTHON_BIN="/ssd/miniconda3/envs/lean4-czx/bin/python"
PROFILE="qwen3_8b_397b_wrong76_llm_split"
# Current iteration is intentionally limited to auditing the LLM COT split.
# Set STAGE explicitly to `cot-to-blueprint` or `blueprint-refine` only when
# that later experiment is requested; cot-only remains disabled in the profile.
STAGE="${STAGE:-split}"

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" experiments/cot_blueprint_refine/run_experiment.py \
  --profile "${PROFILE}" \
  --stage "${STAGE}" \
  "$@"

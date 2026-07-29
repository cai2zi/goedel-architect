#!/usr/bin/env bash
set -euo pipefail
cd /ssd/czx/goedel-architect
# Only change this experiment name.
EXP_NAME="${EXP_NAME:-qwen3_5_397b_MiniF2F_orig_reExp44}"

PYTHON_BIN="/ssd/miniconda3/envs/lean4-czx/bin/python"

# Optional Gemini/full-SC judge settings.
export GEMINI_API_KEY="${GEMINI_API_KEY:-}"
export GEMINI_API_URL="${GEMINI_API_URL:-https://poloai.top/v1beta/models/gemini-2.5-flash:generateContent}"
export FULL_SC_WORKERS="${FULL_SC_WORKERS:-256}"

"${PYTHON_BIN}" experiments/robustpa_refine/export_stmt_sc_inputs.py \
  --exp-name "${EXP_NAME}"

# "${PYTHON_BIN}" experiments/robustpa_refine/evaluate_stmt_sc.py \
#   --exp-name "${EXP_NAME}" \
#   --workers "${FULL_SC_WORKERS}"

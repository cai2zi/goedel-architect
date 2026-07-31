#!/usr/bin/env bash
set -euo pipefail
cd /ssd/czx/goedel-architect
# Only change this experiment name.
EXP_NAME="${EXP_NAME:-qwen3_5_397b_math500_orig_rePipe_debug44}"

PYTHON_BIN="/ssd/miniconda3/envs/lean4-czx/bin/python"

# Optional Gemini/full-SC judge settings.
export GEMINI_API_KEY="${GEMINI_API_KEY:-sk-mj1oXlPWtTFxpdl0QzghZPASKItzJWvi9eDCN5HZELOJALnh}"
export GEMINI_API_URL="${GEMINI_API_URL:-https://poloai.top/v1beta/models/gemini-2.5-flash:generateContent}"
export FULL_SC_WORKERS="${FULL_SC_WORKERS:-256}"
export PROOF_SC_SCOPE="${PROOF_SC_SCOPE:-context-with-proof}"

# --proof-sc-scope final-proof
# --proof-sc-scope context-no-proof
# --proof-sc-scope context-with-proof

"${PYTHON_BIN}" experiments/robustpa_refine/export_stmt_sc_inputs.py \
  --exp-name "${EXP_NAME}"

"${PYTHON_BIN}" experiments/robustpa_refine/evaluate_stmt_sc.py \
  --exp-name "${EXP_NAME}" \
  --workers "${FULL_SC_WORKERS}" \
  --proof-sc-scope "${PROOF_SC_SCOPE}"

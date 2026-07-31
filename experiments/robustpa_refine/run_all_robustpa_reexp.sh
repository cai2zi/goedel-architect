#!/usr/bin/env bash
set -euo pipefail

# Run from the goedel-architect repo root:
#   bash experiments/robustpa_refine/run_all_robustpa_rePipe_debug.sh

DEFAULT_PYTHON_BIN="/ssd/miniconda3/envs/lean4-czx/bin/python"
if [[ -x "${DEFAULT_PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi
MODEL="${MODEL:-Qwen3.5-397B-A17B-FP8}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8001/v1}"
DATA_ROOT="${DATA_ROOT:-/ssd/czx/czx_work/RobustPABench}"
OUTPUT_BASE="${OUTPUT_BASE:-/ssd/czx/czx_work/robustpa_refine}"
RESUME="${RESUME:-true}"
LIMIT="${LIMIT:-null}"

MAX_REFINEMENT_ITERATIONS="${MAX_REFINEMENT_ITERATIONS:-4}" # refine的次数
BLUEPRINT_MAX_RETRIES="${BLUEPRINT_MAX_RETRIES:-4}" # 生成blueprint的次数
NODE_MAX_PROVE_TURNS="${NODE_MAX_PROVE_TURNS:-4}" # node的prove的轮数
NODE_MAX_NEGATION_PROBE_TURNS="${NODE_MAX_NEGATION_PROBE_TURNS:-1}" # 失败node 证明否命题的次数
PARALLEL_TOOL_CALLS="${PARALLEL_TOOL_CALLS:-3}" # 每轮tool call的次数
NODE_TIMEOUT_S="${NODE_TIMEOUT_S:-null}"
LLM_API_TIMEOUT_S="${LLM_API_TIMEOUT_S:-null}"

PHASE1_CONCURRENCY="${PHASE1_CONCURRENCY:-1024}"
PHASE2_BLUEPRINT_CONCURRENCY="${PHASE2_BLUEPRINT_CONCURRENCY:-512}"
PHASE2_NODE_CONCURRENCY="${PHASE2_NODE_CONCURRENCY:-4096}"
REFINE_CONCURRENCY="${REFINE_CONCURRENCY:-1024}"
PHASE2_CONTRACT_CHECK_CONCURRENCY="${PHASE2_CONTRACT_CHECK_CONCURRENCY:-8}"

LEAN_BACKEND="${LEAN_BACKEND:-kimina_server}"
LEAN_API_URL="${LEAN_API_URL:-http://localhost:8000}"
LEAN_SERVER_TIMEOUT="${LEAN_SERVER_TIMEOUT:-600}"
LEAN_SERVER_REUSE="${LEAN_SERVER_REUSE:-true}"
LEAN_SERVER_DEBUG="${LEAN_SERVER_DEBUG:-false}"
LEAN_CHECK_CONCURRENCY="${LEAN_CHECK_CONCURRENCY:-64}"

# Local vLLM usually ignores the key, but the OpenAI SDK requires one.
export GOEDEL_OPENAI_API_KEY="${GOEDEL_OPENAI_API_KEY:-dummy}"

COMMON_OVERRIDES=(
  "model=${MODEL}"
  "openai_base_url=${VLLM_BASE_URL}"
  "data_root=${DATA_ROOT}"
  "output_base=${OUTPUT_BASE}"
  "resume=${RESUME}"
  "limit=${LIMIT}"
  "max_refinement_iterations=${MAX_REFINEMENT_ITERATIONS}"
  "blueprint_max_retries=${BLUEPRINT_MAX_RETRIES}"
  "node_max_prove_turns=${NODE_MAX_PROVE_TURNS}"
  "node_max_negation_probe_turns=${NODE_MAX_NEGATION_PROBE_TURNS}"
  "parallel_tool_calls=${PARALLEL_TOOL_CALLS}"
  "node_timeout_s=${NODE_TIMEOUT_S}"
  "llm_api_timeout_s=${LLM_API_TIMEOUT_S}"
  "phase1_concurrency=${PHASE1_CONCURRENCY}"
  "phase2_blueprint_concurrency=${PHASE2_BLUEPRINT_CONCURRENCY}"
  "phase2_node_concurrency=${PHASE2_NODE_CONCURRENCY}"
  "refine_concurrency=${REFINE_CONCURRENCY}"
  "phase2_contract_check_concurrency=${PHASE2_CONTRACT_CHECK_CONCURRENCY}"
  "lean_backend=${LEAN_BACKEND}"
  "lean_api_url=${LEAN_API_URL}"
  "lean_server_timeout=${LEAN_SERVER_TIMEOUT}"
  "lean_server_reuse=${LEAN_SERVER_REUSE}"
  "lean_server_debug=${LEAN_SERVER_DEBUG}"
  "lean_check_concurrency=${LEAN_CHECK_CONCURRENCY}"
)

format_elapsed() {
  local total_seconds="$1"
  local hours=$((total_seconds / 3600))
  local minutes=$(((total_seconds % 3600) / 60))
  local seconds=$((total_seconds % 60))
  if (( hours > 0 )); then
    printf "%dh%dmin" "${hours}" "${minutes}"
  elif (( minutes > 0 )); then
    printf "%dmin" "${minutes}"
  else
    printf "%ds" "${seconds}"
  fi
}

run_exp() {
  local exp_name="$1"
  local split="$2"
  local subset="$3"
  local output_root="${OUTPUT_BASE%/}/${exp_name}"
  local start
  local elapsed
  local metric_elapsed
  start="$(date +%s)"
  "${PYTHON_BIN}" experiments/robustpa_refine/run_robustpa_refine.py \
    "${COMMON_OVERRIDES[@]}" \
    "exp_name=${exp_name}" \
    "split=${split}" \
    "subset=${subset}"
  elapsed=$(($(date +%s) - start))
  metric_elapsed="$(
    "${PYTHON_BIN}" -c 'import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
try:
    print(json.loads(path.read_text(encoding="utf-8")).get("elapsed_time", ""))
except Exception:
    print("")' "${output_root}/metrics.json"
  )"
  if [[ -n "${metric_elapsed}" ]]; then
    printf '[runtime] exp_name=%s current_run_elapsed_time=%s total_elapsed_time=%s\n' \
      "${exp_name}" "$(format_elapsed "${elapsed}")" "${metric_elapsed}"
  else
    printf '[runtime] exp_name=%s current_run_elapsed_time=%s\n' \
      "${exp_name}" "$(format_elapsed "${elapsed}")"
  fi
}

run_exp qwen3_5_397b_MiniF2F_orig_rePipe_debug44 miniF2F global_original
run_exp qwen3_5_397b_math500_orig_rePipe_debug44 MATH500 global_original

run_exp qwen3_5_397b_MiniF2F_global_gemini_rephrase_rePipe_debug44 miniF2F global_gemini_rephrase
run_exp qwen3_5_397b_math500_global_gemini_rephrase_rePipe_debug44 MATH500 global_gemini_rephrase

run_exp qwen3_5_397b_MiniF2F_global_gemini_step_rePipe_debug44 miniF2F global_gemini_step
run_exp qwen3_5_397b_math500_global_gemini_step_rePipe_debug44 MATH500 global_gemini_step

run_exp qwen3_5_397b_MiniF2F_global_qwen3_rephrase_rePipe_debug44 miniF2F global_qwen3_rephrase
run_exp qwen3_5_397b_math500_global_qwen3_rephrase_rePipe_debug44 MATH500 global_qwen3_rephrase

run_exp qwen3_5_397b_MiniF2F_global_qwen3_step_rePipe_debug44 miniF2F global_qwen3_step
run_exp qwen3_5_397b_math500_global_qwen3_step_rePipe_debug44 MATH500 global_qwen3_step

run_exp qwen3_5_397b_MiniF2F_local_number_edit_proof_rePipe_debug44 miniF2F local_number_edit_proof
run_exp qwen3_5_397b_math500_local_number_edit_proof_rePipe_debug44 MATH500 local_number_edit_proof

run_exp qwen3_5_397b_MiniF2F_local_number_edit_statement_rePipe_debug44 miniF2F local_number_edit_statement
run_exp qwen3_5_397b_math500_local_number_edit_statement_rePipe_debug44 MATH500 local_number_edit_statement

run_exp qwen3_5_397b_MiniF2F_local_step_delete_rePipe_debug44 miniF2F local_step_delete
run_exp qwen3_5_397b_math500_local_step_delete_rePipe_debug44 MATH500 local_step_delete

run_exp qwen3_5_397b_MiniF2F_local_symbol_edit_proof_rePipe_debug44 miniF2F local_symbol_edit_proof
run_exp qwen3_5_397b_math500_local_symbol_edit_proof_rePipe_debug44 MATH500 local_symbol_edit_proof

run_exp qwen3_5_397b_MiniF2F_local_symbol_edit_statement_rePipe_debug44 miniF2F local_symbol_edit_statement
run_exp qwen3_5_397b_math500_local_symbol_edit_statement_rePipe_debug44 MATH500 local_symbol_edit_statement

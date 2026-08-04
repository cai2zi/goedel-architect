#!/usr/bin/env bash
set -euo pipefail
cd /ssd/czx/goedel-architect
# Only change this experiment name.
EXP_NAME="${EXP_NAME:-all_splits_subsets_reAll}"

PYTHON_BIN="/ssd/miniconda3/envs/lean4-czx/bin/python"

# Optional Gemini/full-SC judge settings.
export GEMINI_API_KEY="${GEMINI_API_KEY:-sk-mj1oXlPWtTFxpdl0QzghZPASKItzJWvi9eDCN5HZELOJALnh}"
export GEMINI_API_URL="${GEMINI_API_URL:-https://poloai.top/v1beta/models/gemini-2.5-flash:generateContent}"
export FULL_SC_WORKERS="${FULL_SC_WORKERS:-256}"
export PROOF_SC_SCOPE="${PROOF_SC_SCOPE:-context-with-proof}"
export EXPORT_WORKERS="${EXPORT_WORKERS:-64}"
export EXPORT_LEAN_MAX_INFLIGHT_SNIPPETS="${EXPORT_LEAN_MAX_INFLIGHT_SNIPPETS:-64}"
export EXPORT_LEAN_BATCH_SIZE="${EXPORT_LEAN_BATCH_SIZE:-16}"

# DEFAULT_WORKERS = 16
# 控制 export_stmt_sc_inputs.py 导出阶段的线程数，也就是同时处理多少条 result row。每条 row 会组装 Lean、写文件，并调用 Lean compiler 做校验。
# LEAN_MAX_INFLIGHT_SNIPPETS = 16
# 控制 Kimina Lean compiler 内部最多同时有多少个 Lean snippet 在飞。它是 Lean 校验请求的并发上限。即使 workers 开得更大，Lean 校验并发也会被这个值卡住。
# LEAN_BATCH_SIZE = 8
# 控制 Lean compiler 发送校验请求时的 batch 大小。它影响请求打包粒度，不是总并发。一般关系可以理解为：
# EXPORT_WORKERS >= EXPORT_LEAN_MAX_INFLIGHT_SNIPPETS >= EXPORT_LEAN_BATCH_SIZE
# 不是硬性要求，但通常这样比较合理。workers 太大只会增加排队和内存压力；batch_size 太大可能让单批请求更重。

# --proof-sc-scope final-proof
# --proof-sc-scope context-no-proof
# --proof-sc-scope context-with-proof

# Current semantic-consistency evaluation scope:
#   Orig   -> global_original
#   G-FF   -> global_gemini_rephrase
#   G-Step -> global_gemini_step
#   Q-FF   -> global_qwen3_rephrase
#   Q-Step -> global_qwen3_step
INCLUDE_SUBSETS=(
  global_original
  global_gemini_rephrase
  global_gemini_step
  global_qwen3_rephrase
  global_qwen3_step
)
INCLUDE_SPLITS=(
  miniF2F
  MATH500
)

FILTER_ARGS=()
for subset in "${INCLUDE_SUBSETS[@]}"; do
  FILTER_ARGS+=(--include-subset "${subset}")
done
for split in "${INCLUDE_SPLITS[@]}"; do
  FILTER_ARGS+=(--include-split "${split}")
done

printf '[run_eval] exp_name=%s\n' "${EXP_NAME}"
printf '[run_eval] adopted_subsets=%s\n' "${INCLUDE_SUBSETS[*]}"
printf '[run_eval] adopted_splits=%s\n' "${INCLUDE_SPLITS[*]}"
printf '[run_eval] export_workers=%s\n' "${EXPORT_WORKERS}"
printf '[run_eval] export_lean_max_inflight_snippets=%s\n' "${EXPORT_LEAN_MAX_INFLIGHT_SNIPPETS}"
printf '[run_eval] export_lean_batch_size=%s\n' "${EXPORT_LEAN_BATCH_SIZE}"

"${PYTHON_BIN}" experiments/robustpa_refine/export_stmt_sc_inputs.py \
  --exp-name "${EXP_NAME}" \
  --workers "${EXPORT_WORKERS}" \
  --lean-max-inflight-snippets "${EXPORT_LEAN_MAX_INFLIGHT_SNIPPETS}" \
  --lean-batch-size "${EXPORT_LEAN_BATCH_SIZE}" \
  "${FILTER_ARGS[@]}"

"${PYTHON_BIN}" experiments/robustpa_refine/evaluate_stmt_sc.py \
  --exp-name "${EXP_NAME}" \
  --workers "${FULL_SC_WORKERS}" \
  --proof-sc-scope "${PROOF_SC_SCOPE}" \
  "${FILTER_ARGS[@]}"

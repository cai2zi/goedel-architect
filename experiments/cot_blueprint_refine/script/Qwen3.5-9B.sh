#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec "${EXPERIMENT_DIR}/run_all.sh" \
  input_predictions=/ssd/czx/czx_work/math_verify_eval/qwen3_5_9b_math_verify/predictions.jsonl \
  exp_name=qwen3_5_9b_blueprint_refine \
  vllm.auto_start=true \
  vllm.auto_destroy=true \
  refine.model=Qwen3.5-9B \
  refine.openai_base_url=http://127.0.0.1:7999/v1 \
  refine.vllm.model_path=/ssd/czx/models/Qwen3.5-9B \
  refine.vllm.served_model_name=Qwen3.5-9B \
  refine.vllm.host=127.0.0.1 \
  refine.vllm.port=7999 \
  refine.vllm.tensor_parallel_size=8 \
  refine.vllm.max_model_len=65536 \
  refine.vllm.max_num_seqs=1024 \
  refine.vllm.gpu_memory_utilization=0.9 \
  refine.context_window=65536 \
  refine.max_tokens=20480 \
  refine.tokenizer_path=/ssd/czx/models/Qwen3.5-9B \
  "$@"

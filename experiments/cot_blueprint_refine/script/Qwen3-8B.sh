#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec "${EXPERIMENT_DIR}/run_all.sh" \
  input_predictions=/ssd/czx/czx_work/math_verify_eval/qwen3_8b_math_verify/predictions.jsonl \
  exp_name=qwen3_8b_blueprint_refine_40 \
  vllm.auto_start=true \
  vllm.auto_destroy=true \
  blueprint.prover_max_tokens=4096 \
  blueprint.max_refinement_iterations=0 \
  blueprint.node_max_negation_probe_turns=0 \
  refine.model=Qwen3-8B \
  refine.openai_base_url=http://127.0.0.1:7999/v1 \
  refine.vllm.model_path=/ssd/czx/models/Qwen3-8B \
  refine.vllm.served_model_name=Qwen3-8B \
  refine.vllm.host=127.0.0.1 \
  refine.vllm.port=7999 \
  refine.vllm.tensor_parallel_size=8 \
  refine.vllm.max_model_len=40960 \
  refine.vllm.max_num_seqs=1024 \
  refine.vllm.gpu_memory_utilization=0.95 \
  refine.context_window=40960 \
  refine.max_tokens=20480 \
  refine.tokenizer_path=/ssd/czx/models/Qwen3-8B \
  "$@"

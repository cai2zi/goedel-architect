#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 MODEL_PATH SERVED_NAME TP_SIZE DP_SIZE PORT [REASONING_PARSER]" >&2
  exit 2
fi

MODEL_PATH=$1
SERVED_NAME=$2
TP_SIZE=$3
DP_SIZE=$4
PORT=$5
REASONING_PARSER=${6:-}
INDEX_PATH="${MODEL_PATH}/model.safetensors.index.json"

test -f "$INDEX_PATH"
missing=0
while IFS= read -r shard; do
  if [[ ! -f "${MODEL_PATH}/${shard}" ]]; then
    echo "missing model shard: ${MODEL_PATH}/${shard}" >&2
    missing=1
  fi
done < <(jq -r '.weight_map[]' "$INDEX_PATH" | sort -u)
if [[ $missing -ne 0 ]]; then
  echo "model download is incomplete; refusing to start vLLM" >&2
  exit 1
fi

export PATH="/ssd/miniconda3/envs/infer/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

args=(
  vllm serve "$MODEL_PATH"
  --served-model-name "$SERVED_NAME"
  --host 127.0.0.1
  --port "$PORT"
  --tensor-parallel-size "$TP_SIZE"
  --dtype bfloat16
  --max-model-len 40960
  --max-num-seqs 1024
  --gpu-memory-utilization 0.9
  --trust-remote-code
)
if [[ $DP_SIZE -gt 1 ]]; then
  args+=(
    --data-parallel-size "$DP_SIZE"
    --data-parallel-size-local "$DP_SIZE"
    --data-parallel-backend mp
  )
fi
if [[ -n $REASONING_PARSER ]]; then
  args+=(--reasoning-parser "$REASONING_PARSER")
fi
exec "${args[@]}"

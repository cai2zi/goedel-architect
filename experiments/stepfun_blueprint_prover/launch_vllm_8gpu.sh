#!/usr/bin/env bash
set -euo pipefail

export PATH="/ssd/miniconda3/envs/infer/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

exec vllm serve /ssd/czx/models/StepFun-Prover-Preview-7B \
  --served-model-name StepFun-Prover-Preview-7B \
  --host 127.0.0.1 \
  --port "${STEPFUN_PORT:-8001}" \
  --tensor-parallel-size 4 \
  --data-parallel-size 2 \
  --data-parallel-size-local 2 \
  --data-parallel-backend mp \
  --dtype bfloat16 \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code

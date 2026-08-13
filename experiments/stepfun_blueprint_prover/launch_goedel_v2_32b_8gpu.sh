#!/usr/bin/env bash
set -euo pipefail
cd /ssd/czx/goedel-architect
exec experiments/stepfun_blueprint_prover/launch_model_vllm.sh \
  /ssd/czx/models/Goedel-Prover-V2-32B Goedel-Prover-V2-32B \
  8 1 "${GOEDEL_PORT:-8001}" qwen3

#!/usr/bin/env bash
set -euo pipefail
cd /ssd/czx/goedel-architect
exec experiments/stepfun_blueprint_prover/launch_model_vllm.sh \
  /ssd/czx/models/StepFun-Prover-Preview-32B StepFun-Prover-Preview-32B \
  8 1 "${STEPFUN_PORT:-8001}"

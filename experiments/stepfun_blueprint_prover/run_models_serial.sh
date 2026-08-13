#!/usr/bin/env bash
set -euo pipefail

cd /ssd/czx/goedel-architect

usage() {
  cat <<'EOF'
Usage:
  run_models_serial.sh [--smoke|--full] [--small-only|--all-models]
                       [--only MODEL_KEY]

Defaults to --smoke --small-only. Smoke mode evaluates one Blueprint per model
in isolated output directories. Full mode uses each config's resumable output
directory. Models are always run serially in this order:
  StepFun-Prover-Preview-7B
  Goedel-Prover-V2-8B
  StepFun-Prover-Preview-32B
  Goedel-Prover-V2-32B

Examples:
  # Safe one-record end-to-end check of the two downloaded small models
  experiments/stepfun_blueprint_prover/run_models_serial.sh

  # One-record check of only StepFun-7B
  experiments/stepfun_blueprint_prover/run_models_serial.sh --only stepfun_7b

  # Full experiment after all model downloads have completed
  experiments/stepfun_blueprint_prover/run_models_serial.sh --full --all-models
EOF
}

MODE=smoke
MODEL_SCOPE=small
ONLY_MODEL=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) MODE=smoke ;;
    --full) MODE=full ;;
    --small-only) MODEL_SCOPE=small ;;
    --all-models) MODEL_SCOPE=all ;;
    --only)
      shift
      if [[ $# -eq 0 ]]; then
        echo "--only requires a model key" >&2
        exit 2
      fi
      ONLY_MODEL=$1
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

PORT=${BLUEPRINT_VLLM_PORT:-8001}
LOG_ROOT=${BLUEPRINT_VLLM_LOG_ROOT:-/ssd/czx/czx_work/stepfun_blueprint_prover/vllm_serial}
PYTHON=/ssd/miniconda3/envs/lean4-czx/bin/python
RUNNER=experiments/stepfun_blueprint_prover/run_experiment.py
mkdir -p "$LOG_ROOT"

MODEL_KEYS=(stepfun_7b goedel_v2_8b stepfun_32b goedel_v2_32b)
if [[ $MODEL_SCOPE == small ]]; then
  MODEL_KEYS=(stepfun_7b goedel_v2_8b)
fi
if [[ -n $ONLY_MODEL ]]; then
  case "$ONLY_MODEL" in
    stepfun_7b|goedel_v2_8b|stepfun_32b|goedel_v2_32b) ;;
    *) echo "invalid --only model key: $ONLY_MODEL" >&2; exit 2 ;;
  esac
  MODEL_KEYS=("$ONLY_MODEL")
fi

VLLM_PID=
stop_vllm() {
  if [[ -n ${VLLM_PID:-} ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[serial] stopping vLLM process group $VLLM_PID"
    kill -TERM -- "-$VLLM_PID" 2>/dev/null || true
    for _ in $(seq 1 60); do
      kill -0 "$VLLM_PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "[serial] vLLM did not exit after 60s; killing its process group" >&2
      kill -KILL -- "-$VLLM_PID" 2>/dev/null || true
    fi
    wait "$VLLM_PID" 2>/dev/null || true
  fi
  VLLM_PID=
}
trap stop_vllm EXIT INT TERM

wait_for_model() {
  local expected=$1
  local log_path=$2
  echo "[serial] waiting for $expected on port $PORT"
  while true; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      wait "$VLLM_PID" || true
      VLLM_PID=
      echo "[serial] vLLM exited before becoming ready; tail of $log_path:" >&2
      tail -n 100 "$log_path" >&2 || true
      return 1
    fi
    if curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null \
      | jq -e --arg expected "$expected" \
        '.data | any(.id == $expected)' >/dev/null; then
      echo "[serial] $expected is ready"
      return 0
    fi
    sleep 5
  done
}

run_one() {
  local key=$1 launcher config served smoke_root log_path
  case "$key" in
    stepfun_7b)
      launcher=experiments/stepfun_blueprint_prover/launch_vllm_8gpu.sh
      config=experiments/stepfun_blueprint_prover/configs/stepfun_7b_wrong76_whole_cot.yaml
      served=StepFun-Prover-Preview-7B
      smoke_root=/ssd/czx/czx_work/stepfun_blueprint_prover/stepfun_7b_serial_smoke_closed_negation
      ;;
    goedel_v2_8b)
      launcher=experiments/stepfun_blueprint_prover/launch_goedel_v2_8b_8gpu.sh
      config=experiments/stepfun_blueprint_prover/configs/goedel_v2_8b_wrong76_whole_cot.yaml
      served=Goedel-Prover-V2-8B
      smoke_root=/ssd/czx/czx_work/stepfun_blueprint_prover/goedel_v2_8b_serial_smoke_closed_negation
      ;;
    stepfun_32b)
      launcher=experiments/stepfun_blueprint_prover/launch_stepfun_32b_8gpu.sh
      config=experiments/stepfun_blueprint_prover/configs/stepfun_32b_wrong76_whole_cot.yaml
      served=StepFun-Prover-Preview-32B
      smoke_root=/ssd/czx/czx_work/stepfun_blueprint_prover/stepfun_32b_serial_smoke_closed_negation
      ;;
    goedel_v2_32b)
      launcher=experiments/stepfun_blueprint_prover/launch_goedel_v2_32b_8gpu.sh
      config=experiments/stepfun_blueprint_prover/configs/goedel_v2_32b_wrong76_whole_cot.yaml
      served=Goedel-Prover-V2-32B
      smoke_root=/ssd/czx/czx_work/stepfun_blueprint_prover/goedel_v2_32b_serial_smoke_closed_negation
      ;;
    *) echo "unknown model key: $key" >&2; return 2 ;;
  esac

  if curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[serial] port $PORT already has a vLLM service; refusing to replace it" >&2
    return 1
  fi

  log_path="$LOG_ROOT/${key}.vllm.log"
  echo "[serial] starting $served; vLLM log: $log_path"
  if [[ $key == stepfun_* ]]; then
    STEPFUN_PORT=$PORT setsid "$launcher" >"$log_path" 2>&1 &
  else
    GOEDEL_PORT=$PORT setsid "$launcher" >"$log_path" 2>&1 &
  fi
  VLLM_PID=$!
  echo "$VLLM_PID" >"$LOG_ROOT/${key}.pid"
  wait_for_model "$served" "$log_path"

  echo "[serial] running $served in $MODE mode"
  if [[ $MODE == smoke ]]; then
    "$PYTHON" "$RUNNER" --config "$config" --limit 1 --output-root "$smoke_root"
  else
    "$PYTHON" "$RUNNER" --config "$config"
  fi
  stop_vllm
}

for key in "${MODEL_KEYS[@]}"; do
  run_one "$key"
done

echo "[serial] all requested model runs completed"

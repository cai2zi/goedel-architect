#!/usr/bin/env bash
set -euo pipefail
cd /ssd/czx/goedel-architect

PY=/ssd/miniconda3/envs/lean4-czx/bin/python
SHARED=/ssd/czx/czx_work/cot_blueprint_refine/qwen3_8b_397b_wrong76_step_v13_shared_phase1a_thinking

# Attach-only checks. This script never starts or destroys either service.
curl -fsS http://127.0.0.1:8000/health >/dev/null
curl -fsS http://127.0.0.1:8001/v1/models >/dev/null

# Resume/validate all 76 Phase 1A terminal rows, then publish exactly 73 seeds.
"$PY" experiments/cot_blueprint_refine/run_experiment.py \
  --profile qwen3_8b_397b_wrong76_step_v13_shared_phase1a --stage phase1-only
"$PY" experiments/cot_blueprint_refine/shared_phase1a.py \
  --root "$SHARED" --expected 76 \
  --drop-source-id cmimc_2025/15 --drop-source-id cmimc_2025/20 \
  --drop-source-id hmmt_feb_2025/12

pids=()
cleanup() {
  local pid
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM HUP

for arm in a_greedy_mixed b_thinking_mixed c_two_stage d_full_regeneration; do
  "experiments/cot_blueprint_refine/script/qwen3_8b_397b_wrong76_step_v13_${arm}.sh" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"

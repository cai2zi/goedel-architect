#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 PROFILE EXPECTED_EXP_NAME [OVERRIDE ...]" >&2
  exit 2
fi

profile=$1
expected_exp_name=$2
shift 2

cd /ssd/czx/goedel-architect
slot_dir=/ssd/czx/czx_work/cot_blueprint_refine/.semantic_audit_ablation_slots
mkdir -p "$slot_dir"
exec 9>"$slot_dir/slot-0.lock"
exec 8>"$slot_dir/slot-1.lock"

while true; do
  if flock -n 9; then
    slot=0
    break
  fi
  if flock -n 8; then
    slot=1
    break
  fi
  sleep 5
done

echo "[semantic-ablation-slot] profile=$profile slot=$slot" >&2
/ssd/miniconda3/envs/lean4-czx/bin/python \
  experiments/cot_blueprint_refine/preflight_semantic_ablation.py \
  "$profile" "$expected_exp_name" "$@"

exec /ssd/miniconda3/envs/lean4-czx/bin/python \
  experiments/cot_blueprint_refine/run_experiment.py \
  --profile "$profile" --stage phase1-only "$@"

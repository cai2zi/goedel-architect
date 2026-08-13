#!/usr/bin/env bash
set -euo pipefail
cd /ssd/czx/goedel-architect

SOURCE_RESULTS=/ssd/czx/czx_work/cot_blueprint_refine/qwen3_8b_397b_wrong76_whole_cot_blueprint_generation_thinking_judge/robustpa/blueprint/results.jsonl
STEPFUN_MANIFEST=/ssd/czx/czx_work/stepfun_blueprint_prover/stepfun_7b_wrong76_whole_cot/manifest.json

# Prove the same 45 strictAccepted Whole-COT Blueprint seeds used by StepFun.
# The source is read-only; output is isolated under the profile's new exp_name.
curl -fsS http://127.0.0.1:8000/health >/dev/null
curl -fsS http://127.0.0.1:8001/v1/models \
  | jq -e '.data | any(.id == "Qwen3.5-397B-A17B-FP8")' >/dev/null

test -f "$SOURCE_RESULTS"
test -f "$STEPFUN_MANIFEST"
diff -u \
  <(jq -r '.selected_ids[]' "$STEPFUN_MANIFEST" | sort) \
  <(jq -r 'select(.status == "strictAccepted" and .semantic_status == "strictAccepted") | .record_id' "$SOURCE_RESULTS" | sort)

exec /ssd/miniconda3/envs/lean4-czx/bin/python \
  experiments/cot_blueprint_refine/run_experiment.py \
  --profile qwen3_8b_397b_wrong76_phase2 \
  --stage phase2-only "$@"

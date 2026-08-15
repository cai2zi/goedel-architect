#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/ssd/czx/goedel-architect
PYTHON=/ssd/miniconda3/envs/lean4-czx/bin/python
PROFILE=qwen3_8b_397b_wrong76_nonthinking_gold_phase2
SEED_ROOT=/ssd/czx/czx_work/wrong76_nonthinking_gold_phase2_seed
OUTPUT_BASE=/ssd/czx/czx_work/cot_blueprint_refine
FULL_EXP=qwen3_8b_397b_wrong76_nonthinking_gold_phase2
SMOKE_EXP=qwen3_8b_397b_wrong76_nonthinking_gold_phase2_smoke_430
SMOKE_ID=MATH-500/test/counting_and_probability/430.json
MODEL=Qwen3.5-397B-A17B-FP8

cd "$REPO_ROOT"

case "${1:-}" in
  "")
    EXP_NAME="$SMOKE_EXP"
    OVERRIDES=(
      "exp_name=$SMOKE_EXP"
      "blueprint.problem_id=$SMOKE_ID"
      "blueprint.phase2_blueprint_concurrency=1"
    )
    ;;
  --full)
    if [[ $# -ne 1 ]]; then
      echo "usage: $0 [--full]" >&2
      exit 2
    fi
    EXP_NAME="$FULL_EXP"
    OVERRIDES=()
    ;;
  *)
    echo "usage: $0 [--full]" >&2
    exit 2
    ;;
esac

"$PYTHON" experiments/wrong76_nonthinking_gold/prepare_phase2_seeds.py \
  --output-root "$SEED_ROOT"

curl -fsS --max-time 10 http://127.0.0.1:8000/health >/dev/null
curl -fsS --max-time 10 http://127.0.0.1:8001/v1/models \
  | "$PYTHON" -c \
    'import json,sys; expected=sys.argv[1]; payload=json.load(sys.stdin); models={x.get("id") for x in payload.get("data", [])}; assert expected in models, f"missing served model {expected}; available={sorted(models)}"' \
    "$MODEL"

"$PYTHON" experiments/cot_blueprint_refine/run_experiment.py \
  --profile "$PROFILE" \
  --stage phase2-only \
  "${OVERRIDES[@]}"

"$PYTHON" experiments/wrong76_nonthinking_gold/evaluate_phase2.py \
  --experiment-root "$OUTPUT_BASE/$EXP_NAME" \
  --seed-root "$SEED_ROOT"

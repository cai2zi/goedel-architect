#!/usr/bin/env bash
set -euo pipefail
cd /ssd/czx/goedel-architect
/ssd/miniconda3/envs/lean4-czx/bin/python experiments/cot_blueprint_refine/shared_phase1a.py \
  --root /ssd/czx/czx_work/cot_blueprint_refine/qwen3_8b_397b_wrong76_step_v13_shared_phase1a_thinking --expected 76 \
  --drop-source-id cmimc_2025/15 --drop-source-id cmimc_2025/20 \
  --drop-source-id hmmt_feb_2025/12 >/dev/null
exec /ssd/miniconda3/envs/lean4-czx/bin/python experiments/cot_blueprint_refine/run_experiment.py \
  --profile qwen3_8b_397b_wrong76_step_v13_b_thinking_mixed --stage blueprint "$@"

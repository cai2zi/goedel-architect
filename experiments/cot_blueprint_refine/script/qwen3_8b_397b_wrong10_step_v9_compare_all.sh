#!/usr/bin/env bash
set -euo pipefail

cd /ssd/czx/goedel-architect
experiments/cot_blueprint_refine/script/qwen3_8b_397b_wrong10_step_v9_progress_controller.sh "$@"
experiments/cot_blueprint_refine/script/qwen3_8b_397b_wrong10_step_v9_plan_direct.sh "$@"
experiments/cot_blueprint_refine/script/qwen3_8b_397b_wrong10_step_v9_direct_edit.sh "$@"

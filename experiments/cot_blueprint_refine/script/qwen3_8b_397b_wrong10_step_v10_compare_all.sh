#!/usr/bin/env bash
set -euo pipefail

cd /ssd/czx/goedel-architect
experiments/cot_blueprint_refine/script/qwen3_8b_397b_wrong10_step_v10_plan_direct_stable_closure.sh "$@"
experiments/cot_blueprint_refine/script/qwen3_8b_397b_wrong10_step_v10_direct_edit_stable_closure.sh "$@"

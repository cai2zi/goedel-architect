#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec env PROFILE=qwen3_8b_397b_refine_ablation \
  "${EXPERIMENT_DIR}/run_all.sh" \
  "$@"

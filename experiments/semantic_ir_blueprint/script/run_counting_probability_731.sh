#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

conda run --no-capture-output -n lean4-czx \
  python experiments/semantic_ir_blueprint/run_experiment.py "$@"

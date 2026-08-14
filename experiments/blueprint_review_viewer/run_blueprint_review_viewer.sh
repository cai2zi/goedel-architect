#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "Usage: $0" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_BASE="${BLUEPRINT_REVIEW_OUTPUT_BASE:-/ssd/czx/czx_work/cot_blueprint_refine}"
SSH_TARGET="${BLUEPRINT_REVIEW_SSH_TARGET:-<user>@<remote-host>}"
exec python "$ROOT/experiments/blueprint_review_viewer/server.py" \
  --output-base "$OUTPUT_BASE" \
  --host 127.0.0.1 \
  --port 8766 \
  --ssh-target "$SSH_TARGET"

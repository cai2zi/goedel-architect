#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 EXPERIMENT_BLUEPRINT_ROOT [ssh-user@remote-host]" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXP_ROOT="$1"
SSH_TARGET="${2:-<user>@<remote-host>}"
exec python "$ROOT/experiments/blueprint_review_viewer/server.py" "$EXP_ROOT" --host 127.0.0.1 --port 8765 --ssh-target "$SSH_TARGET"

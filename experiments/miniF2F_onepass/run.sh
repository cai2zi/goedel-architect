#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-experiments/miniF2F_onepass/configs/base.yaml}"
MODEL="${MODEL:-gpt-5-mini}"
SPLIT="${SPLIT:-test}"
DATA_DIR="${DATA_DIR:-data/minif2f}"
OUTPUT_ROOT="${OUTPUT_ROOT:-../czx_work/goedel-architect/miniF2F_onepass/gpt-5-mini}"
NODE_TIMEOUT_S="${NODE_TIMEOUT_S:-300}"
PHASE1_CONCURRENCY="${PHASE1_CONCURRENCY:-16}"
PHASE2_BLUEPRINT_CONCURRENCY="${PHASE2_BLUEPRINT_CONCURRENCY:-8}"
PHASE2_NODE_CONCURRENCY="${PHASE2_NODE_CONCURRENCY:-16}"
LEAN_CHECK_CONCURRENCY="${LEAN_CHECK_CONCURRENCY:-4}"
LEAN_BACKEND="${LEAN_BACKEND:-kimina_server}"
LEAN_API_URL="${LEAN_API_URL:-${KIMINA_API_URL:-http://localhost:8000}}"
LEAN_SERVER_TIMEOUT="${LEAN_SERVER_TIMEOUT:-300}"
COT_ID_FIELD="${COT_ID_FIELD:-name}"
COT_TEXT_FIELD="${COT_TEXT_FIELD:-nl_proof}"
COT_ALLOW_MISSING="${COT_ALLOW_MISSING:-true}"

args=(
  --config "${CONFIG}"
  --model "${MODEL}"
  --split "${SPLIT}"
  --data-dir "${DATA_DIR}"
  --output-root "${OUTPUT_ROOT}"
  --node-timeout-s "${NODE_TIMEOUT_S}"
  --phase1-concurrency "${PHASE1_CONCURRENCY}"
  --phase2-blueprint-concurrency "${PHASE2_BLUEPRINT_CONCURRENCY}"
  --phase2-node-concurrency "${PHASE2_NODE_CONCURRENCY}"
  --lean-check-concurrency "${LEAN_CHECK_CONCURRENCY}"
  --lean-backend "${LEAN_BACKEND}"
  --lean-api-url "${LEAN_API_URL}"
  --lean-server-timeout "${LEAN_SERVER_TIMEOUT}"
  --cot-id-field "${COT_ID_FIELD}"
  --cot-text-field "${COT_TEXT_FIELD}"
)

if [[ -n "${LIMIT:-}" ]]; then args+=(--limit "${LIMIT}"); fi
if [[ -n "${PROBLEM_ID:-}" ]]; then args+=(--problem-id "${PROBLEM_ID}"); fi
if [[ -n "${COT_PATH:-}" ]]; then args+=(--cot-path "${COT_PATH}"); fi
if [[ "${RESUME:-false}" == "true" ]]; then args+=(--resume); fi
if [[ "${LEAN_SERVER_REUSE:-true}" == "false" ]]; then args+=(--no-lean-server-reuse); fi
if [[ "${LEAN_SERVER_DEBUG:-false}" == "true" ]]; then args+=(--lean-server-debug); fi
if [[ "${COT_ALLOW_MISSING}" == "true" ]]; then
  args+=(--cot-allow-missing)
else
  args+=(--no-cot-allow-missing)
fi

python experiments/miniF2F_onepass/run_minif2f_onepass.py "${args[@]}" "$@"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-experiments/tts_rerank_math_verify/configs/base.yaml}"
MODEL="${MODEL:-gpt-5.4-mini}"
BENCH_PATH="${BENCH_PATH:-../czx_work/bench.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-../czx_work/goedel-architect/tts_rerank_math_verify/gpt-5.4-mini}"
PHASE0_MAX_ATTEMPTS="${PHASE0_MAX_ATTEMPTS:-3}"
NODE_TIMEOUT_S="${NODE_TIMEOUT_S:-300}"
PHASE1_CONCURRENCY="${PHASE1_CONCURRENCY:-4}"
PHASE2_BLUEPRINT_CONCURRENCY="${PHASE2_BLUEPRINT_CONCURRENCY:-4}"
PHASE2_NODE_CONCURRENCY="${PHASE2_NODE_CONCURRENCY:-8}"
LEAN_CHECK_CONCURRENCY="${LEAN_CHECK_CONCURRENCY:-8}"
GOEDEL_BLUEPRINT_MAX_TOKENS="${GOEDEL_BLUEPRINT_MAX_TOKENS:-262144}"
GOEDEL_PROVER_MAX_TOKENS="${GOEDEL_PROVER_MAX_TOKENS:-64000}"
GOEDEL_LLM_MAX_RETRIES="${GOEDEL_LLM_MAX_RETRIES:-6}"
GOEDEL_LLM_RETRY_BASE_DELAY_S="${GOEDEL_LLM_RETRY_BASE_DELAY_S:-2}"
GOEDEL_LLM_RETRY_MAX_DELAY_S="${GOEDEL_LLM_RETRY_MAX_DELAY_S:-60}"
GOEDEL_LLM_RETRY_JITTER_S="${GOEDEL_LLM_RETRY_JITTER_S:-1}"
export GOEDEL_BLUEPRINT_MAX_TOKENS
export GOEDEL_PROVER_MAX_TOKENS
export GOEDEL_LLM_MAX_RETRIES
export GOEDEL_LLM_RETRY_BASE_DELAY_S
export GOEDEL_LLM_RETRY_MAX_DELAY_S
export GOEDEL_LLM_RETRY_JITTER_S
LEAN_BACKEND="${LEAN_BACKEND:-kimina_server}"
LEAN_API_URL="${LEAN_API_URL:-${KIMINA_API_URL:-http://localhost:8000}}"
LEAN_SERVER_TIMEOUT="${LEAN_SERVER_TIMEOUT:-300}"

args=(
  --config "${CONFIG}"
  --model "${MODEL}"
  --bench-path "${BENCH_PATH}"
  --output-root "${OUTPUT_ROOT}"
  --phase0-max-attempts "${PHASE0_MAX_ATTEMPTS}"
  --node-timeout-s "${NODE_TIMEOUT_S}"
  --phase1-concurrency "${PHASE1_CONCURRENCY}"
  --phase2-blueprint-concurrency "${PHASE2_BLUEPRINT_CONCURRENCY}"
  --phase2-node-concurrency "${PHASE2_NODE_CONCURRENCY}"
  --lean-check-concurrency "${LEAN_CHECK_CONCURRENCY}"
  --lean-backend "${LEAN_BACKEND}"
  --lean-api-url "${LEAN_API_URL}"
  --lean-server-timeout "${LEAN_SERVER_TIMEOUT}"
)

if [[ -n "${LIMIT:-}" ]]; then args+=(--limit "${LIMIT}"); fi
if [[ -n "${PROBLEM_ID:-}" ]]; then args+=(--problem-id "${PROBLEM_ID}"); fi
if [[ -n "${ROLLOUT_ID:-}" ]]; then args+=(--rollout-id "${ROLLOUT_ID}"); fi
if [[ "${RESUME:-false}" == "true" ]]; then args+=(--resume); fi
if [[ "${INCLUDE_EMPTY_EXTRACTED:-false}" == "true" ]]; then args+=(--include-empty-extracted); fi
if [[ "${LEAN_SERVER_REUSE:-true}" == "false" ]]; then args+=(--no-lean-server-reuse); fi
if [[ "${LEAN_SERVER_DEBUG:-false}" == "true" ]]; then args+=(--lean-server-debug); fi

python experiments/tts_rerank_math_verify/run_tts_rerank.py "${args[@]}" "$@"

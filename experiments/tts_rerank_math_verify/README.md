# TTS Rerank Math-Verify

Runs one-pass Goedel-Architect scoring over rollout answers in `czx_work/bench.json`.
The experiment uses Math-Verify labels already stored in the bench file; it does not
rerun Math-Verify.

Phase 0 theorem checking and Phase 1/2 Lean checks use one process-level Kimina
HTTP client by default. The experiment does not start or stop Kimina and does
not manage server-side REPL processes. Kimina must already point to the correct
`goedel_lean` project and a REPL built for its Lean toolchain.

Example:

```bash
export OPENAI_BASE_URL=https://poloai.top/v1
export GOEDEL_OPENAI_BASE_URL=https://poloai.top/v1
export GOEDEL_BLUEPRINT_MAX_TOKENS=262144
export GOEDEL_PROVER_MAX_TOKENS=64000
export GOEDEL_TOOL_CHOICE_MODE=auto
export KIMINA_API_URL=http://localhost:8000
export KIMINA_API_KEY=
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897

python experiments/tts_rerank_math_verify/run_tts_rerank.py \
  --model deepseek-v4-flash \
  --limit 1 \
  --problem-id aime25_test__0 \
  --rollout-id 1
```

Or use the experiment script, whose defaults can be overridden with
environment variables and trailing CLI flags:

```bash
PHASE1_CONCURRENCY=4 PHASE2_BLUEPRINT_CONCURRENCY=4 PHASE2_NODE_CONCURRENCY=8 \
bash experiments/tts_rerank_math_verify/run.sh --limit 1
```

Use `--lean-backend local` for the existing `lake env lean` fallback. The shared
runtime flags are:

```text
--lean-backend kimina_server|local
--lean-api-url URL
--lean-api-key-env ENV_NAME
--lean-server-timeout SECONDS
--lean-server-reuse / --no-lean-server-reuse
--lean-server-debug / --no-lean-server-debug
--lean-check-concurrency N
```

The runner keeps Phase 0 unchanged and serial for now. It first completes or
reuses Phase 0 for every selected rollout, then completes Phase 1 for every
successful rollout, and only then starts Phase 2. Phase 1 blueprint creation,
Phase 2 blueprint proving, and Phase 2 node proving are controlled by:

```text
--phase1-concurrency N
--phase2-blueprint-concurrency N
--phase2-node-concurrency N
```

`GOEDEL_BLUEPRINT_MAX_TOKENS` controls the Phase 1 blueprint completion
budget. `GOEDEL_PROVER_MAX_TOKENS` controls the Phase 2 per-node prover
completion budget. For rate-limited OpenAI deployments, start conservatively:

```bash
GOEDEL_BLUEPRINT_MAX_TOKENS=32768 GOEDEL_PROVER_MAX_TOKENS=16384 \
PHASE1_CONCURRENCY=1 PHASE2_BLUEPRINT_CONCURRENCY=1 PHASE2_NODE_CONCURRENCY=2 \
bash experiments/tts_rerank_math_verify/run.sh --model gpt-5-mini
```

`--node-timeout-s` limits a complete LLM node attempt; the separate
`--lean-server-timeout` limits one Lean check. `lean_runtime.json` and each
result row record the non-secret runtime settings. Resume refuses missing or
mismatched runtime metadata.

With `--resume`, solved rollout scores and terminal Phase 0 failures are
skipped. An unfinished rollout reuses its successful Phase 0 row. If its
checkpoint contains a non-empty, fully validated blueprint, Phase 1 is skipped
and Phase 2 continues with nodes not already in `proved_cache`; otherwise the
blueprint is generated again. Checkpoints for a different theorem statement
are rejected.

Outputs are written by default to:

```text
czx_work/goedel-architect/tts_rerank_math_verify/deepseek_v4_flash/
```

Main outputs:

- `phase0_results.jsonl`
- `rollout_scores.jsonl`
- `goedel_best.jsonl`
- `metrics.json`
- `metrics.csv`
- per-rollout checkpoints, traces, and generated blueprints

## Linux acceptance

Start Kimina yourself from its repository:

```bash
python -m server
```

Because `/health` verifies only HTTP availability, first submit a real Lean
check from another terminal:

```bash
curl -sS -X POST "${KIMINA_API_URL:-http://localhost:8000}/api/check" \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${KIMINA_API_KEY}" \
  -d '{"snippets":[{"id":"goedel-smoke","code":"import Mathlib\nimport Architect\n#check Nat"}],"timeout":300,"debug":true,"reuse":true}'

python experiments/tts_rerank_math_verify/run_tts_rerank.py \
  --model deepseek-v4-flash --limit 1
```

Confirm same-header REPL reuse in the server logs and stop Kimina manually with
Ctrl+C. The experiment never performs either lifecycle operation.

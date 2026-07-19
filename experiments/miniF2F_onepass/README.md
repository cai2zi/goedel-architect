# miniF2F One-Pass

Runs one blueprint extraction and one DAG proving pass for miniF2F rows. It calls
`run_phase1()` and `run_phase2()` only, and does not run refinement.

Lean checks use a manually managed Kimina server by default. The experiment does
not start or stop Kimina and does not manage server-side REPL processes. Kimina
must already be configured to use this checkout's `goedel_lean` project and a
REPL built for the matching Lean toolchain.

Example:

```bash
export OPENAI_BASE_URL=https://poloai.top/v1
export GOEDEL_OPENAI_BASE_URL=https://poloai.top/v1
export GOEDEL_BLUEPRINT_MAX_TOKENS=128000
export GOEDEL_PROVER_MAX_TOKENS=64000
export GOEDEL_TOOL_CHOICE_MODE=auto
export KIMINA_API_URL=http://localhost:8000
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export OPENAI_API_KEY=
python experiments/miniF2F_onepass/run_minif2f_onepass.py \
  --model gpt-5-mini \
  --split test 
```

Or use the experiment script, whose defaults can be overridden with
environment variables and trailing CLI flags:

```bash
PHASE1_CONCURRENCY=4 PHASE2_BLUEPRINT_CONCURRENCY=4 PHASE2_NODE_CONCURRENCY=8 \
bash experiments/miniF2F_onepass/run.sh --limit 2
```

Use `--lean-backend local` to fall back to the existing `lake env lean`
implementation. Kimina runtime flags are:

```text
--lean-backend kimina_server|local
--lean-api-url URL
--lean-api-key-env ENV_NAME
--lean-server-timeout SECONDS
--lean-server-reuse / --no-lean-server-reuse
--lean-server-debug / --no-lean-server-debug
--lean-check-concurrency N
```

The runner first completes Phase 1 for the selected rows, then starts Phase 2.
Phase 1 blueprint creation, Phase 2 blueprint proving, and Phase 2 node proving
are controlled by:

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
bash experiments/miniF2F_onepass/run.sh --model gpt-5-mini
```

miniF2F can optionally load external chain-of-thought hints from a JSONL file
aligned by problem id:

```json
{"name": "mathd_algebra_478", "nl_proof": "Natural-language proof sketch..."}
```

Use `--cot-path`, `--cot-id-field`, `--cot-text-field`, and
`--cot-allow-missing / --no-cot-allow-missing`. The loaded text is passed as
`nl_proof` to the blueprint prompt.

`--node-timeout-s` limits a complete LLM node attempt. It is independent of
`--lean-server-timeout`, which limits one Kimina Lean check.

Outputs are written by default to:

```text
czx_work/goedel-architect/miniF2F_onepass/deepseek_v4_flash/<split>/
```

The output root contains `lean_runtime.json`, and every result row records the
same non-secret runtime metadata. `--resume` refuses old output without this
file or output produced with different Lean runtime settings.

With `--resume`, a record whose root theorem is already proved is skipped. For
an unfinished record, a non-empty checkpoint blueprint is reused only when its
`blueprint_fully_validated` flag is true; Phase 1 is skipped and Phase 2 retries
the nodes not present in `proved_cache`. Missing or unvalidated blueprints are
generated again. A checkpoint whose theorem statement differs from the input
is rejected instead of being mixed into the run.

## Linux acceptance

The user starts and stops the service manually:

```bash
cd /path/to/kimina-lean-server
python -m server
```

`/health` checks only the HTTP service. Before running the experiment, use a
separate terminal to verify real Lean execution once:

```bash
curl -sS -X POST "${KIMINA_API_URL:-http://localhost:8000}/api/check" \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${KIMINA_API_KEY}" \
  -d '{"snippets":[{"id":"goedel-smoke","code":"import Mathlib\nimport Architect\n#check Nat"}],"timeout":300,"debug":true,"reuse":true}'

python experiments/miniF2F_onepass/run_minif2f_onepass.py \
  --model deepseek-v4-flash --split test --limit 1
```

Confirm in the Kimina logs that checks with the same header reuse a REPL, then
stop the server manually with Ctrl+C.

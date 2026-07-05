# goedel-arch

An implementation of the Goedel-Architect pipeline for Lean 4 theorem proving:
decompose a theorem into a dependency graph of sub-lemmas (Phase 1), prove each
node in parallel (Phase 2), and refine the decomposition when nodes fail
(Phase 3), repeating up to a fixed number of iterations.

## Requirements

- Python 3.10+
- [elan](https://github.com/leanprover/elan) (Lean version manager, provides `lake`)
- An OpenAI API key

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENAI_API_KEY
```

Fetch and build the Lean side (`goedel_lean/`) — this pulls in the real
[LeanArchitect](https://github.com/hanwenzhu/LeanArchitect) package, which
provides the `@[blueprint]` attribute and `sorry_using` syntax the pipeline
relies on:

```bash
cd goedel_lean
lake exe cache get   # downloads prebuilt Mathlib .olean files
lake build
cd ..
```

This build is toolchain-independent from whatever project you're actually
proving theorems in: Phase 1's blueprint validation strips `@[blueprint ...]`
and `sorry_using` annotations before compiling against the target project's
own toolchain, so `goedel_lean/`'s pinned Lean version (see
`goedel_lean/lean-toolchain`) never has to match.

## Repo layout

```
src/            Phase 1/2/3 pipeline (blueprint.py, orchestrator.py, prover.py, refinement.py, pipeline.py)
goedel_lean/    Small Lean package depending on the real LeanArchitect, plus
                #validate_blueprint / #list_blueprint helper commands
prompts/        System/user prompt templates (Appendix C of the paper)
eval/           Per-benchmark driver scripts (see below)
data/           Benchmark data used by the eval scripts
```

## Running against a benchmark

Each benchmark has its own thin adapter script under `eval/` that wires
`src/pipeline.py`'s `prove_theorem()` to that benchmark's data format and
verification method.

### VeriSoftBench

Requires a local checkout of [VeriSoftBench](https://github.com/utopia-group/VeriSoftBench)
as a sibling directory (`../VeriSoftBench`), with its Lean repos built per
its own README (`./scripts/setup_repos.sh --clone --config-dir build_config
--output-dir data/lean_repos`, or `lake exe cache get && lake build` per repo
if a repo's build is incomplete).

```bash
# Smoke test: a single theorem
python eval/run_verisoftbench.py --thm add_comm --repo lean-formal-reasoning-program --blueprint

# A repo-diverse pilot: 20 random tasks spanning many repos, sharded across 8 workers
python eval/run_verisoftbench.py --model gpt-5.5 --blueprint --subset 20 --workers 8 --resume --output results/pilot/

# Full benchmark
python eval/run_verisoftbench.py --model gpt-5.5 --blueprint --workers 20 --resume --output results/full/
```

Key flags:

| Flag | Purpose |
|---|---|
| `--blueprint` | Use the full 3-phase pipeline (omit for Phase-2-only direct proving) |
| `--subset N` | Random sample of N tasks spanning all repos (the dataset is grouped by repo, so `--limit N` alone clusters on whichever repo sorts first) |
| `--workers N` | Parallel workers, **sharded by repo** — theorems from the same repo always run sequentially (a shared `lake`/compiler working directory isn't safe for concurrent access), but different repos run fully in parallel |
| `--resume` | Skip theorems already recorded in the output dir's `results.jsonl` under the same model/mode/blueprint config, and append new results incrementally instead of overwriting |
| `--repo` / `--thm` | Filter to one repo or theorem name, for smoke testing |
| `--trace` | Write a JSONL trace of every tool call/compile attempt, viewable via `eval/serve_viz.py` |

Results land in `<output>/results.jsonl` (one line per theorem) and
`<output>/summary.json` (pass rate, average wall time).

### PutnamBench / miniF2F

```bash
python eval/run_putnam.py --model gpt-5.5 --limit 20
python eval/run_minif2f.py --model gpt-5.5 --limit 20
```

## Notes

- `MAX_REFINEMENT_ITERATIONS = 8` and a 300s per-node timeout
  (`src/pipeline.py`) match the paper's Appendix A; both are overridable
  per-call if you're running a cost-constrained pilot.
- Each node gets up to 8 tool calls (`lean_compile`, `repo_search`,
  `mathlib_search`) before being marked `proof_too_hard`.
- A node's outcome is one of `SOLVED`, `PROOF_TOO_HARD`, `STATEMENT_WRONG`, or
  `FORMALLY_NEGATED` (the last is machine-checked: the pipeline actually
  proves the *negation* of the stated sub-lemma, meaning the decomposition
  asserted something false — see `_probe_negation` in `src/prover.py`).

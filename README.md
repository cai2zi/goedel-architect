# Goedel-Architect: RobustPA Kimina-Only

This repository runs the three-phase Goedel-Architect proof pipeline for the
RobustPA experiment:

1. Generate and validate a Lean blueprint.
2. Prove the active root dependency closure.
3. Refine the blueprint from machine-produced Lean diagnostics.

Lean compilation is performed exclusively through a Kimina server. There is no
local `lake env lean` backend and no compiler backend switch. The `goedel_lean/`
package remains because Kimina must provide `Architect`, `sorry_using`, and
`#validate_blueprint` for blueprint validation.

## Layout

```text
src/                         Pipeline, compiler client, prover, and refinement
prompts/                     Phase 1/2/3 prompts
experiments/robustpa_refine/ RobustPA runner, runtime, reporting, and trace UI
eval/                        Deterministic and live Kimina tests
goedel_lean/                 Kimina-side Lean environment configuration
```

## Run

Start the Kimina server on port 8000 and an OpenAI-compatible model server on
port 8001, then run one RobustPA record:

```bash
/ssd/miniconda3/envs/lean4-czx/bin/python \
  experiments/robustpa_refine/run_robustpa_refine.py \
  exp_name=smoke \
  subset=global_original \
  split=MATH500 \
  problem_id=test_prealgebra_1924 \
  openai_base_url=http://127.0.0.1:8001/v1
```

See [experiments/robustpa_refine/README.md](experiments/robustpa_refine/README.md)
for configuration, output, and trace details.

## Tests

```bash
/ssd/miniconda3/envs/lean4-czx/bin/python -m unittest discover -s eval -p 'test_*.py'

GOEDEL_RUN_LIVE_KIMINA=1 \
  /ssd/miniconda3/envs/lean4-czx/bin/python -m unittest eval.test_kimina_live -v
```

Checkpoints use the current Kimina-only schema and are intentionally
incompatible with older experiment checkpoints.

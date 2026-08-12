# RobustPA Blueprint proving and refinement

`run_robustpa_refine.py` is the execution engine used by the COT experiment.
Its input rows must contain `informal_statement`, `informal_proof`,
`cot_manifest_json`, and `claimed_answer`.

Blueprint generation has one route: `blueprint_generation.generate_blueprint`
regenerates a complete Step-grounded Blueprint each round and accepts it only
after deterministic Lean/contract checks plus the strict semantic audit.
There are no generic non-COT, Pending, seed, node-edit, Planner, or search
generation modes.

For normal use, invoke the higher-level runner documented in
`experiments/cot_blueprint_refine/README.md`; it prepares those required rows
and passes a resolved configuration to this engine.

Downstream behavior is unchanged:

- Phase 2 proves Blueprint nodes with Lean and may use Mathlib search.
- Phase 3 refines the Blueprint after proof failures while preserving Step
  semantics.
- only successful no-sorry compilation of the root closure sets
  `root_proved=true`.

Results contain terminal JSONL rows, round records, traces, checkpoints,
Blueprint snapshots, and aggregate metrics. Existing historical output trees
remain read-only and are not migrated to the new schema.

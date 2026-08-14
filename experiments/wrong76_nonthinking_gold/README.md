# Wrong76 Non-Thinking Gold Blueprints

This experiment builds manually audited Gold Blueprints for all 76 records in
`qwen3_8b_original_answer_incorrect/predictions.jsonl`.

The initial authoring input is blind. It contains only the problem, the final
answer after the last `</think>` delimiter, and the model's claimed answer.
Dataset gold answers, correctness fields, hidden thinking, previous Blueprint
labels, and model-judge results were excluded through the first complete
76-record freeze.

After that freeze, a reference-answer audit was deliberately performed on the
16 records whose target still had label `proved`. It found cases where the
first graph had verified only terminal arithmetic, and those records were
repaired using explicit problem/COT contradictions. Therefore the final
revision is suitable as a Gold comparison set, but must not be reported as a
strictly blind evaluation result. The exact provenance is recorded in
`freeze_manifest.json` and `REPORT.md`.

## Contracts

- `steps` contains only material mathematical steps from the non-thinking COT.
- Every Blueprint node has one of three explicit roles:
  `problem_grounding`, `cot_claim`, or `formal_bridge`.
- A truth-valued COT assertion must have a lemma/theorem node; it may not be
  represented only by a definition.
- Every active definition is `definition_valid` after compilation.
- Every active lemma/theorem is `proved`, `disproved`, or
  `blocked_by_dependency`.
- `disproved` means a replayable Lean proof of the exact negation of the closed
  theorem.
- Current deterministic validation is mechanical only: parsing, whole-file
  Lean, canonical rebuild/Lean, Phase-2 contract, and Phase-2 standalone.
  It does not include semantic static gates or LLM semantic audits.
- A record is `gold_complete` only when mechanical validation, proof replay,
  label coverage, and the separate manual fidelity review all pass.

Runtime artifacts are isolated under
`/ssd/czx/czx_work/wrong76_nonthinking_gold`.

## Bootstrap

```bash
/ssd/miniconda3/envs/infer/bin/python \
  experiments/wrong76_nonthinking_gold/build_blind_inputs.py
```

The command is deterministic and refuses to overwrite a frozen input with
different content.

## Render and validate

The hand-audited cases are split into eleven authoring modules. Render all of
them, then validate every record against the Lean server on port 8000:

```bash
for batch in $(seq -w 1 11); do
  /ssd/miniconda3/envs/infer/bin/python \
    experiments/wrong76_nonthinking_gold/render_gold.py \
    --module "cases.batch${batch}"
done

for record_dir in /ssd/czx/czx_work/wrong76_nonthinking_gold/records/*; do
  /ssd/miniconda3/envs/infer/bin/python \
    experiments/wrong76_nonthinking_gold/validate_gold.py \
    --record-id "$(basename "$record_dir")"
done
```

After all 76 records are `gold_complete`, freeze hashes and regenerate the
machine-readable and human-readable reports:

```bash
/ssd/miniconda3/envs/infer/bin/python \
  experiments/wrong76_nonthinking_gold/freeze_and_report.py
```

The final report is in `REPORT.md`; runtime records, independent Lean replay
files, `final_summary.json`, and `freeze_manifest.json` are under
`/ssd/czx/czx_work/wrong76_nonthinking_gold`.

For a source-ID-matched comparison with the existing 397B Whole-COT Phase-1
artifact, run:

```bash
/ssd/miniconda3/envs/infer/bin/python \
  experiments/wrong76_nonthinking_gold/compare_existing_pipeline.py
```

This writes `PIPELINE_COMPARISON.md` and the machine-readable runtime file
`pipeline_comparison.json`. It deliberately does not treat the old Phase-1
proof counters as a same-stage prover benchmark.

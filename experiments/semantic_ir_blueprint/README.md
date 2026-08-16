# Semantic IR Blueprint V0

This isolated experiment supports two Semantic IR generation modes selected by
`semantic_ir.generation_mode` in Hydra config:

- `combined`: source-unit splitting, one combined Definitions + Nodes request,
  then Blueprint generation (three model requests total).
- `definitions_then_nodes`: source-unit splitting, a Definition Registry
  request, a proof Nodes request using the validated frozen registry, then
  Blueprint generation (four model requests total).

The source splitter performs a lossless partition rather than selecting source
content. `source_split.max_units: 9` limits the result to fewer than ten units.
Every mechanical anchor belongs to exactly one adjacent unit, and concatenating
all `source_text` values must reproduce the complete COT byte-for-byte before a
later generation stage can run.

It then submits the extracted Blueprint to Kimina exactly once. There is no tool call, retry, repair, semantic audit, or Phase 2 node prover. The OpenAI SDK and Kimina client are both configured with zero retries. Existing vLLM and Kimina services must already be running; the launcher does not own their lifecycle.

Run:

```bash
bash experiments/semantic_ir_blueprint/script/run_counting_probability_731.sh
```

Run the Definition-then-Node experiment:

```bash
bash experiments/semantic_ir_blueprint/script/run_counting_probability_731_definitions_then_nodes.sh
```

The launchers use the `lean4-czx` Conda environment. Configuration is composed
by Hydra from `config/`; normal Hydra overrides can be appended to either
launcher. All configured results are isolated below
`/ssd/czx/czx_work/semantic_ir_blueprint/<experiment_name>/`. A non-empty
record directory is never overwritten.

In `definitions_then_nodes` mode, the additional artifacts are
`definitions/{conversation.json,raw_response.txt,definitions.json}` and
`nodes/{conversation.json,raw_response.txt,nodes.json}`. The assembled full IR
is still saved as `semantic_ir/semantic_ir.json`. A failed request, parse,
validation, extraction, or compilation is terminal, but every artifact and
conversation available at that point is retained. Conversation records contain
the exact messages, sampling parameters, SDK response, reasoning content,
assistant content, usage, timing, finish reason, and exception details. They
intentionally contain no credentials, environment dump, authorization header,
or hash fields.

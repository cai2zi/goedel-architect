# Semantic IR Blueprint V0

This isolated experiment runs one prepared MATH-500 record through three model requests:

1. select source-unit end boundaries over lossless mechanical anchors;
2. emit one Semantic IR containing Definitions and proof Nodes;
3. translate only that IR into a complete Lean Blueprint.

It then submits the extracted Blueprint to Kimina exactly once. There is no tool call, retry, repair, semantic audit, or Phase 2 node prover. The OpenAI SDK and Kimina client are both configured with zero retries. Existing vLLM and Kimina services must already be running; the launcher does not own their lifecycle.

Run:

```bash
bash experiments/semantic_ir_blueprint/script/run_counting_probability_731.sh
```

The launcher uses the `lean4-czx` Conda environment. Configuration is composed by Hydra from `config/counting_probability_731.yaml`; normal Hydra overrides can be appended to the launcher command. The configured output is written below a new experiment directory under `/ssd/czx/czx_work/cot_blueprint_refine`. A non-empty record directory is never overwritten. A failed request, parse, validation, extraction, or compilation is terminal, but all artifacts and conversations available at that point are retained. Conversation records contain the exact messages, sampling parameters, SDK response, reasoning content, assistant content, usage, timing, finish reason, and exception details. They intentionally contain no credentials, environment dump, authorization header, or hash fields.

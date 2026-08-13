# Blueprint Node Truth Labels

Human/Codex-authored Lean truth labels for the 45 immutable `strictAccepted`
Whole-COT Blueprints. No LLM is called by this experiment.

Final labels are:

- `definition_valid`
- `proved`
- `disproved` (a proof of the exact negation of the entire closed theorem)
- `blocked_by_dependency`

Every `proved` or `disproved` row points to a complete replayable Lean file
accepted by the configured Lean server without `sorry` or `admit`.


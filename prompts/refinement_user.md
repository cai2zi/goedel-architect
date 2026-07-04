Revise the blueprint below. Each node is annotated with `-- PROVED` or `-- UNPROVED` followed by a `/- Diagnosis ... -/` block describing what went wrong.

Produce a corrected dependency graph. Do NOT copy the markers or diagnosis blocks into your output.
{{#if repo_context}}

## Repository context (already in scope — do NOT re-define these)

The following definitions, instances, and lemmas are already available. Your revised blueprint must only contain **new** helper lemmas — do not redefine anything from here.

```lean
{{repo_context}}
```
{{/if}}

## Current blueprint

```lean
{{annotated_lean}}
```

Call `lean_compile` after every edit and iterate until it reports `Compilation SUCCESSFUL. Validation SUCCESSFUL.`

Revise the blueprint below. Each node is annotated with `-- PROVED` or `-- UNPROVED` followed by a `/- Diagnosis ... -/` block describing what went wrong.

Produce a corrected dependency graph. Do NOT copy the markers or diagnosis blocks into your output.
{{#if round_info}}

## Refinement progress

{{round_info}}

If a node has already failed in an earlier round under any name — even reworded or re-split — do not just rename or cosmetically re-decompose it again. Either commit to a genuinely different mathematical strategy (a different induction, a different case split, a different helper lemma shape), or leave it as an explicit unresolved gap (`sorry_using [...]` naming what it still needs) and spend this round's effort on branches that are actually making progress. Repeating the same idea under a new name costs a full proving budget for no gain.
{{/if}}
{{#if prior_rounds}}

## Earlier rounds (for context — what has already been tried and failed)

{{prior_rounds}}
{{/if}}
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

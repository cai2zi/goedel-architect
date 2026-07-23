Produce a `@[blueprint]`-annotated Lean 4 dependency graph for the following problem.

## Required main theorem identifier

```text
{{target_name}}
```

The main theorem in your Lean output MUST be named exactly `{{target_name}}`.

## Informal statement

{{informal_statement}}

{{#if informal_proof}}
## Informal proof

Use this as a structural hint when formalizing and decomposing the proof. You do not need to follow it exactly; reformulate or split steps as needed to make each lemma nearly trivial given its parents.

{{informal_proof}}
{{/if}}

Call `lean_compile` on your blueprint and iterate until it reports `Compilation SUCCESSFUL. Validation SUCCESSFUL.`

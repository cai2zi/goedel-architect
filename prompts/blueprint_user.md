Produce a `@[blueprint]`-annotated Lean 4 dependency graph for the following theorem.

## Theorem signature

```lean
{{theorem_stmt}}
```
{{#if nl_proof}}

## Informal proof (for guidance only)

Use this as a structural hint when decomposing the proof into lemmas. You do not need to follow it exactly — reformulate or split steps as needed to make each lemma nearly trivial given its parents.

{{nl_proof}}
{{/if}}

Call `lean_compile` on your blueprint and iterate until it reports `Compilation SUCCESSFUL. Validation SUCCESSFUL.`

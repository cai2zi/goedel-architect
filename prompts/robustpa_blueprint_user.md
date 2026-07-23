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

Translate this proof faithfully. Follow its reasoning steps, intermediate claims, equations, constants, symbols, and proof order as the primary structure of the blueprint. You may split a step into smaller lemmas to make the graph type-correct and tractable, but each split lemma must still correspond to the proof as written.

Important: this proof may contain deliberate perturbations, missing reasoning, or mathematical errors. Do not silently correct them, revert them to an original version, or replace them with a different proof. If a step is wrong or unjustified, encode the claimed step as its own `@[blueprint]` lemma/gap with `:= by sorry_using [...]` and describe the same claimed reasoning in the natural-language `proof` field.

{{informal_proof}}
{{/if}}

Call `lean_compile` on your blueprint and iterate until it reports `Compilation SUCCESSFUL. Validation SUCCESSFUL.`

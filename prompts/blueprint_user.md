Produce a `@[blueprint]`-annotated Lean 4 dependency graph for the following theorem.

## Theorem signature

```lean
{{theorem_stmt}}
```
{{#if repo_context}}

## Repository context (available to USE — do NOT redeclare)

The following definitions, instances, and lemmas are already available in the repository. **Use them in your helper lemmas' types and proof bodies.** Do NOT redeclare or redefine anything that appears here.

**Critical**: Even though these building blocks exist, the main theorem still requires structured decomposition. You MUST create 2–5 new helper lemmas — one per major proof step or case. Do not collapse everything into a single-node blueprint.

```lean
{{repo_context}}
```
{{/if}}
{{#if nl_proof}}

## Informal proof (for guidance only)

Use this as a structural hint when decomposing the proof into lemmas. You do not need to follow it exactly — reformulate or split steps as needed to make each lemma nearly trivial given its parents.

{{nl_proof}}
{{/if}}

Call `lean_compile` on your blueprint and iterate until it reports `Compilation SUCCESSFUL. Validation SUCCESSFUL.`

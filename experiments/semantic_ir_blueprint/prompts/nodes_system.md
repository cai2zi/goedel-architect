You build only the proof Nodes of a Semantic IR from a possibly wrong mathematical chain-of-thought and a frozen Definition Registry. Preserve every source assertion needed for the final result, including false, incomplete, or contradictory claims. Do not modify, replace, or add Definitions. Do not solve the problem, repair the argument, weaken a claim, or introduce an unconstrained answer witness.

Return exactly one JSON object with exactly one top-level key, `nodes`, and no other text. Do not call tools.

Every Node has exactly this shape:

```json
{
  "id": "n_rewrite_preserves_value",
  "kind": "lemma",
  "depends_on": ["n_initial_evaluation"],
  "claim": {
    "form": "relation",
    "binders": [{"name": "n", "type": "positive integer"}],
    "assumptions": ["n satisfies the source constraints"],
    "lhs": "transformed_expression(n)",
    "relation": "equals",
    "rhs": "reference_value(n)"
  },
  "source_units": ["S006"],
  "source_description": "The source asserts that the transformation preserves the evaluated value."
}
```

Requirements:

- `kind` is `lemma` or `theorem`.
- Node IDs are unique identifiers using letters, digits, and underscores and do not begin with a digit.
- `depends_on` contains only IDs of earlier Nodes. Never put a Definition ID in `depends_on`.
- Use an empty dependency list when no earlier proof claim is needed.
- Every independently falsifiable source inference needed by the final result remains an explicit Node claim.
- There is exactly one theorem, it is the final Node, and its ID equals the target theorem identifier supplied by the user.
- The final theorem states the claimed result about the original constrained object and does not introduce an unconstrained answer witness.
- `source_units` contains supplied source-unit IDs; `source_description` is a semantic paraphrase and need not quote the source verbatim.
- The supplied Definition Registry is intended to be complete. Reuse its IDs consistently and express relationships among those objects as Nodes; do not silently replace them with newly invented objects.

Each `claim` uses exactly one of these open forms:

1. `relation`: `{"form":"relation","binders":[],"assumptions":[],"lhs":"...","relation":"any non-empty semantic relation","rhs":"..."}`. The relation string has no whitelist.
2. `predicate`: `{"form":"predicate","binders":[],"assumptions":[],"predicate":"...","arguments":["..."]}`.
3. `proposition`: `{"form":"proposition","binders":[],"assumptions":[],"proposition":"..."}` for quantifiers, implications, logical combinations, or nested propositions.

All forms may contain typed `binders` of shape `{"name":"x","type":"..."}` and semantic `assumptions` strings. Refer to supplied Definitions by their exact IDs wherever needed to preserve object identity and constraints.

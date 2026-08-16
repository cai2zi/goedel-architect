You convert a possibly wrong mathematical chain-of-thought into a faithful Semantic IR for Lean Blueprint generation. Preserve what the source actually claims, including false, incomplete, or contradictory reasoning. Do not solve the problem, repair the argument, weaken a claim, or replace a constrained object with a fresh unconstrained witness.

Return exactly one JSON object and no other text. Do not call tools. The object must have exactly two arrays: `definitions` followed by `nodes`.

## Definitions

Use this exact record shape:

```json
{
  "id": "index_collection",
  "params": [{"name": "n", "type": "positive integer"}],
  "type": "finite collection of integers",
  "definition": "the integers i satisfying 0 <= i and i < n",
  "source_units": ["S004"],
  "source_description": "The source introduces a constrained collection used by later claims."
}
```

Requirements:

- `id` and parameter `name` values are identifiers using letters, digits, and underscores, and do not begin with a digit.
- `params` lists the explicit parameters needed by the definition.
- `type` and `definition` are precise mathematical semantic descriptions, not Lean syntax.
- A definition body may refer to definitions that occur earlier in the array, but not later definitions.
- Definitions never have `depends_on`. They are shared mathematical context and do not correspond to `sorry_using` dependencies.
- `source_units` contains one or more supplied source-unit IDs.
- `source_description` may be a semantic paraphrase. It does not need to be a verbatim source substring.
- Build a comprehensive registry of every reusable object required to state the source reasoning: relevant domains, structured inputs, functions, collections, operations, auxiliary constructions, distinguished quantities, intermediate objects, and the sought object. Review every source unit and do not leave gaps that would force Nodes to invent objects.
- Definitions introduce objects and constructions. Nodes state and validate relationships, properties, equalities, inequalities, implications, and other claims about them. A named predicate or relation may be defined, but the assertion that it holds belongs in a Node.
- Preserve distinctions among differently constrained source objects. Do not preassign a computed or sought object to the claimed answer.

## Nodes

Use this exact record shape:

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
- `depends_on` contains only IDs of earlier lemma/theorem nodes. Never put a Definition ID in `depends_on`.
- Use an empty dependency list when no earlier proof claim is needed.
- Every source assertion needed for the final result must remain an explicit node claim, even when it is false.
- There must be exactly one theorem node, it must be the final node, and its ID must equal the target theorem identifier supplied by the user.
- The final theorem must state the original claimed result about the original constrained object. It must not introduce an unconstrained answer witness.
- `source_description` is a semantic paraphrase and need not quote the source verbatim.

Each `claim` uses exactly one of these open forms:

1. Binary relation:

```json
{
  "form": "relation",
  "binders": [{"name": "x", "type": "real number"}],
  "assumptions": ["x is positive"],
  "lhs": "f(x)",
  "relation": "is congruent to modulo 7",
  "rhs": "g(x)"
}
```

`relation` is an open semantic string. It is not restricted to equality or an ordering.

2. Predicate:

```json
{
  "form": "predicate",
  "binders": [],
  "assumptions": [],
  "predicate": "is_minimal",
  "arguments": ["candidate(n)", "admissible_collection(n)"]
}
```

3. General proposition fallback:

```json
{
  "form": "proposition",
  "binders": [{"name": "i", "type": "finite index"}],
  "assumptions": [],
  "proposition": "if i is admissible, then term(i) belongs to selected_collection(n)"
}
```

Use `proposition` for quantifiers, implications, logical combinations, or nested propositions that do not fit naturally into the first two forms. All three forms may contain `binders` and `assumptions`.

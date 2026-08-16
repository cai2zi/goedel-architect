You build only the Definition Registry of a Semantic IR from a possibly wrong mathematical chain-of-thought. Preserve the mathematical objects, domains, constraints, and constructions used by the source. Do not generate proof Nodes, solve the problem, repair the reasoning, or replace a constrained object with an unconstrained witness.

Return exactly one JSON object with exactly one top-level key, `definitions`, and no other text. Do not call tools.

Every Definition has exactly this shape:

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

- `id` and parameter `name` values are identifiers using letters, digits, and underscores and do not begin with a digit.
- Definition IDs are unique and ordered by first conceptual use.
- `params` lists every explicit parameter needed by the definition.
- `type` and `definition` are precise mathematical semantic descriptions, not Lean syntax.
- A Definition may refer only to Definitions that occur earlier in the array.
- Definitions never have `depends_on`, `claim`, or proof content. They will become concrete Lean definitions and do not correspond to `sorry_using` dependencies.
- `source_units` contains one or more supplied source-unit IDs.
- `source_description` is a semantic paraphrase and need not be a verbatim substring.
- Build a comprehensive inventory of the reusable mathematical objects introduced or used by the problem and source reasoning. This includes relevant domains, structured inputs, functions, sets or collections, operations, auxiliary constructions, distinguished quantities, intermediate objects, and the object whose value or property is ultimately claimed.
- Define as many of these objects as needed so that later Nodes can refer to them by stable IDs and state their relationships precisely. Do not omit an object merely because it appears obvious, is introduced informally, or is used only in a later source unit.
- Before returning, review every source unit and ensure that every reusable object needed by any later claim is represented either as a Definition or as an explicit parameter of an appropriate Definition. The registry must not leave gaps that would force the Node stage to invent new objects.
- Definitions introduce objects, constructions, operations, and the meanings of named predicates or relations. Nodes—not Definitions—must state and validate that particular objects satisfy properties, equalities, inequalities, implications, or other source claims.
- If the source introduces a named predicate or relation, its meaning may be defined here, but the assertion that it holds in a particular case belongs in a Node.
- Preserve distinctions among source objects with different roles or constraints. Do not collapse them merely because the source later relates them.
- Do not define a computed or sought object to equal the claimed answer. The connection between that object and the claimed value must remain a Node claim.

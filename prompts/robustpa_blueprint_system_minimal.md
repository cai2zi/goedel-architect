You translate one complete informal chain of thought into one complete Lean 4
Blueprint file. Preserve the source reasoning even when it is mathematically
wrong: the goal is faithful formalization, not correction.

The file must import `Mathlib` and `Architect`. Every semantic declaration must
use `@[blueprint]`. Definitions must have real Lean bodies. Every lemma and
theorem must use exactly one proof body of the form
`:= by sorry_using [dependency_names]`; the list is the explicit Blueprint DAG.
Do not use `sorry`, `admit`, `by?`, or placeholder propositions elsewhere.

Create exactly one root theorem with the requested name. Its Lean type must
still bind the original problem objects, relations, directionality, and task
quantifiers, and must express the claimed answer as the conclusion of that
model. Preserve every material object and intermediate relationship from the
complete COT and connect them to the root through explicit dependencies.

`title`, `statement`, and `proof` metadata are optional and receive no credit.
Do not emit source-step identifiers or split the COT into externally labelled
steps.

Return the complete file by calling `lean_compile` exactly once. Return no
prose and no partial declaration.

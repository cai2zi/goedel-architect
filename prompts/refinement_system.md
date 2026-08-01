## Task

Revise a Lean 4 `@[blueprint ...]` dependency graph after a proving pass. The
original natural-language problem is canonical; the current Lean declarations
are an attempted formalization and may be wrong.

Every lemma and theorem in your output must retain a body of the form
`:= by sorry_using [deps]`. Definitions keep executable bodies. Keep the main
theorem name fixed, but you may repair its binders, hypotheses, or conclusion,
and may likewise repair, replace, add, or remove helper nodes. A formally
negated node is strong evidence that its current declaration must be corrected
or removed.

Input verdicts are `PROVED`, `PROOF_TOO_HARD`, `BLOCKED_BY_DEPENDENCY`,
`FORMALLY_NEGATED`, `INFRA_ERROR`, or `PROTOCOL_ERROR`. A diagnosis contains
only the submitted proof body, signal, and exact Lean errors. Use source
positions and goals in those errors to revise the graph. Do not copy verdicts
or diagnosis comments into the output.

Preserve a proved declaration when it still supports the repaired root. Avoid
renaming or cosmetically splitting a failed claim without changing the
mathematical strategy. Emit one complete Lean file containing imports,
definitions, blueprint nodes, and exactly one main theorem with the required
target name.

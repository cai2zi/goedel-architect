## Task
You are revising a Lean 4 dependency graph for a single mathematical problem. The input is a sequence of `@[blueprint ...]`-annotated declarations -- definitions, lemmas, and one main theorem -- each lemma or theorem with body `:= by sorry_using [deps]`. Your job is to emit a revised dependency graph -- again all `sorry_using` declarations -- that, when handed back to the same Lean 4 theorem prover, is more likely to close the previously-unsolved nodes while still proving the same main theorem.

## Input format
Each lemma or theorem in the input carries a one-line marker recording the previous prover pass's verdict on that node, and -- when the prover failed -- a follow-up review block describing what went wrong. There are two markers.

A `-- PROVED` marker means the prover proved the node.

A `-- UNPROVED` marker indicates that the prover failed on the node, and is followed by exactly one `/- Diagnosis ... -/` review block. The block has three sections. `## Diagnosis` is exactly one of `STATEMENT_WRONG` (the lemma is false under its hypotheses) or `PROOF_TOO_HARD` (the prover believes the goal is provable but could not chain the available parents to it). `## Analysis` is a forensic account of what the prover tried, what compiled, what errors remained, and where the gap is. `## Suggested Fix` is conditional on the diagnosis: for `STATEMENT_WRONG`, why the statement is false and how to repair it; for `PROOF_TOO_HARD`, a helper-lemma decomposition.

These markers and review blocks are input-only -- do NOT copy them into your revised dependency graph.

## Guidance
Each `-- UNPROVED` node falls into one of two buckets, decided by the `## Diagnosis` label.

When the diagnosis is `STATEMENT_WRONG`, the lemma's formal statement is false under its hypotheses. Fix the statement (strengthen hypotheses, weaken the conclusion, fix a quantifier or coercion, etc.) and re-emit it. If the lemma is structurally unfixable, drop it and re-route the nodes that depended on it.

When the diagnosis is `PROOF_TOO_HARD`, the prover believes the goal is provable but could not chain the available parents to it. Read the `## Suggested Fix` for the prover's proposed helper-lemma decomposition and add new parent lemmas (each as a fresh `@[blueprint ...]` declaration with body `:= by sorry_using [...]`) that bridge the gap. Wire the failing node's `sorry_using [...]` to include the new helpers. If the analysis instead reads as though the statement itself is suspect, treat it as `STATEMENT_WRONG` instead -- fix or drop the statement.

Leave `-- PROVED` nodes untouched unless a downstream revision forces a signature change: their proof bodies will carry forward automatically as long as the signature stays byte-identical.

After every edit, call `lean_compile`. The tool reports pre-compile safeguard violations, real Lean compile errors, the skeleton-out invariant (every theorem/lemma body must remain `:= by sorry_using [...]`), graph-validity issues (cycles, missing fields, dead nodes, etc.), and on a clean compile a per-declaration proof-reuse check. Iterate until `lean_compile` reports `Compilation SUCCESSFUL. Validation SUCCESSFUL.`

## Diagnosing and fixing Lean 4 compile errors

When `lean_compile` feeds back Lean errors, apply the following rules.

**"typeclass instance problem is stuck, it is often due to metavariables"**
Lean cannot infer one or more implicit type parameters in the statement. Common causes and fixes:
- A `do`-notation expression like `do let s ← StateT.get; StateT.set s` does not tell Lean which monad the block runs in. Fix: pin the monad by making the first monadic value explicit, e.g. `do let s ← @StateT.get σ m inst; StateT.set s`, or add a type ascription `(do ... : StateT σ m PUnit)`.
- A bare `MonadState.get` or `MonadStateOf.get` in a `do` block without the monad type in scope. Fix: write `@MonadStateOf.get σ m inst` or annotate: `(MonadStateOf.get : m σ)`.
- A lemma that uses universe variables `u v w` but the type expressions still have free metavariables. Fix: add explicit `{σ : Type u} {m : Type u → Type v}` binders to every statement that mentions `StateT` or `MonadStateOf`.
- General rule: whenever you see "metavariables" in a Lean error, add `@` prefix with all type arguments explicit, or add a `(expr : ExpectedType)` ascription, so Lean does not have to guess.

**"unknown identifier" or "unknown constant"**
A name used in `sorry_using [...]` or in the statement body does not exist in scope. Fix: check that the name is exactly the name of an earlier `@[blueprint]`-annotated node in the same graph, a definition from the repo module, or a Mathlib lemma. Remove or rename the unknown identifier.

**"failed to synthesize" a typeclass**
A required instance (e.g. `[Monad m]`, `[LawfulMonad m]`, `[LawfulMonadStateOf σ m]`) is missing from the declaration's hypotheses. Fix: add the missing `[ClassName args]` to the declaration's binder list.

**"application type mismatch"**
A function is applied to an argument of the wrong type. Fix: check the expected type of each argument and add type ascriptions or fix universe levels (e.g. `Type u` vs `Type v`).

## Output
Emit a revised dependency graph. Every theorem and lemma is `@[blueprint (statement := /-- ... -/) (proof := /-- ... -/)]`-annotated and ends in `:= by sorry_using [deps]`. Definitions are `@[blueprint (statement := /-- ... -/)]`-annotated with a real Lean body. Do NOT replace any `sorry_using` with an actual proof -- that is the prover's job, not yours. Preserve the main theorem's signature (name, binders, conclusion) byte-for-byte from the input.

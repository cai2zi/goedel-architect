You are also given evidence from a Lean 4 blueprint.

The context quality is one of VERIFIED, INVALID_BLUEPRINT_CANDIDATE, or INFRA_ERROR. A VERIFIED context passed export validation. An INVALID_BLUEPRINT_CANDIDATE did not pass Lean or the blueprint contract, but its decomposition may still be useful as fallible reference together with its diagnostics. INFRA_ERROR means formal checking did not complete, so any available blueprint text is reference only.

The Lean context may contain machine-checked solved declarations and unresolved declarations with explicit status comments. Check that every formal statement actually matches the original natural-language problem before relying on it. Natural-language node statements and proof sketches are guidance, not checked facts or gold answers.

Interpret node status comments as follows:

- PROVED: Lean checked the displayed declaration and proof. It is formal evidence only when its assumptions and conclusion faithfully match the original problem.
- NOT_PROVED: The node was not successfully proved. The step may be wrong, incomplete, or require a different method. Failure alone does not prove the statement false.
- BLOCKED_BY_DEPENDENCY: The node was not attempted because an upstream dependency failed. This gives no independent verdict on the node itself.
- FORMALLY_NEGATED: Lean checked a proof of the formal negation. The corresponding step is wrong and must be replaced.

You are also given evidence from a Lean 4 blueprint.

The context quality is one of VERIFIED, INVALID_BLUEPRINT_CANDIDATE, or INFRA_ERROR. VERIFIED means only that the artifact passed structural/export validation; it does not certify semantic faithfulness, mathematical truth, or the claimed answer. An INVALID_BLUEPRINT_CANDIDATE did not pass Lean or the blueprint contract, but its decomposition may still be useful as fallible reference together with its diagnostics. INFRA_ERROR means formal checking did not complete, so any available blueprint text is reference only.

The Lean context may contain machine-checked solved declarations and unresolved declarations with explicit status comments. Check that every formal statement actually matches the original natural-language problem before relying on it. Natural-language node statements and proof sketches are guidance, not checked facts or gold answers.

Each node may also carry a `COT_BLUEPRINT_SOURCE_STEP` comment. This identifies the numbered original-COT step that the node is intended to formalize. Use the mapping to compare the Lean declaration directly with that source step. A mapping records intended provenance; it does not by itself establish semantic faithfulness.

Agreement or repetition between the original claimed answer and the blueprint is not independent evidence: the blueprint was generated from that same source response. Require a faithful formal statement and a valid dependency path before treating a checked node as useful evidence.

Interpret node status comments as follows:

- PROVED: Lean checked the displayed declaration and proof. It is formal evidence only when its assumptions and conclusion faithfully match the original problem.
- DEFINITION: Lean accepted the displayed definition and body as well-typed. This is not a theorem proof and does not show that the definition faithfully models the problem, the linked COT step, or the claimed answer. In particular, do not treat a value hard-coded in a definition as verified mathematical evidence.
- NOT_PROVED: The node was not successfully proved. The step may be wrong, incomplete, or require a different method. Failure alone does not prove the statement false.
- BLOCKED_BY_DEPENDENCY: The node was not attempted because an upstream dependency failed. This gives no independent verdict on the node itself.
- FORMALLY_NEGATED: Lean checked a proof of the formal negation. The corresponding step is wrong and must be replaced.

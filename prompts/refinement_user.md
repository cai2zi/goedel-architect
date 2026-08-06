Revise the blueprint below. Node markers and diagnosis blocks describe the
previous proving pass. The original natural-language problem is canonical;
repair the Lean formalization, including the root declaration when necessary.

## Original natural-language problem

{{informal_statement}}

## Original COT / informal proof

{{informal_proof}}

Produce a corrected dependency graph. Do NOT copy the markers or diagnosis blocks into your output.
{{#if round_info}}

## Refinement progress

{{round_info}}

If a node has already failed in an earlier round under any name — even reworded or re-split — do not just rename or cosmetically re-decompose it again. Either commit to a genuinely different mathematical strategy (a different induction, a different case split, a different helper lemma shape), or leave it as an explicit unresolved gap (`sorry_using [...]` naming what it still needs) and spend this round's effort on branches that are actually making progress. Repeating the same idea under a new name costs a full proving budget for no gain.
{{/if}}
{{#if prior_rounds}}

## Earlier rounds (for context — what has already been tried and failed)

{{prior_rounds}}
{{/if}}
## Current blueprint

```lean
{{annotated_lean}}
```

Emit the complete revised blueprint. It will be compiled and any errors will be
returned for another attempt.

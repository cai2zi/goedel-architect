You split a possibly wrong mathematical chain-of-thought into semantic source units for later formalization. Do not solve, repair, summarize, reorder, omit, or rewrite any source content. Boundary labels are mechanical coordinates, not semantic atoms.

A source unit is the smallest semantically coherent block that can support one definition or one small connected group of proof claims. It normally contains a local setup, its needed conditions, one main inference or calculation, and its tightly connected result.

Boundary rules:

- This is a complete partition, not content selection. Every supplied boundary anchor belongs to exactly one source unit, in original order. An anchor whose ID is not printed remains inside the adjacent unit; none of its text is discarded.
- Partition the entire source into at most {{max_units}} source units. The number of reported end boundaries is therefore at most {{max_units}}.
- Merge headings, method narration, colon-ended lead-ins, displayed formulas, and their immediate explanation into one unit. None may form a unit alone.
- Split independent conclusions, independent transformations, new case branches, or changes of mathematical object.
- Never split a formula from the prose that introduces or interprets it.
- Do not follow source labels such as `Step 1` mechanically.
- Do not create one unit per sentence or list item when adjacent assertions share one local mathematical inference.
- Do not hide multiple independently falsifiable reasoning jumps inside one oversized unit.
- Preserve wrong and contradictory reasoning exactly. Boundary choices must not repair the argument or make its claims easier to formalize.
- Return unit-end boundaries in strictly increasing order. The final boundary must be {{final_boundary}}, which guarantees that the partition covers the complete source through its final character.

Return exactly one `[[SOURCE_UNITS_V1]] ... [[/SOURCE_UNITS_V1]]` block. Inside the block, write only the boundary ID ending each adjacent unit, one per line and at most {{max_units}} lines. These IDs define grouping boundaries; they do not select which source text to retain. Write no prose, JSON, Markdown fence, or text outside the block.

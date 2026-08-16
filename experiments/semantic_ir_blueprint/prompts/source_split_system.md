You split a possibly wrong mathematical chain-of-thought into semantic source units for later formalization. Do not solve, repair, summarize, reorder, omit, or rewrite any source content. Boundary labels are mechanical coordinates, not semantic atoms.

A source unit is the smallest semantically coherent block that can support one definition or one small connected group of proof claims. It normally contains a local setup, its needed conditions, one main inference or calculation, and its tightly connected result.

Boundary rules:

- Merge headings, method narration, colon-ended lead-ins, displayed formulas, and their immediate explanation into one unit. None may form a unit alone.
- Split independent conclusions, independent transformations, new case branches, or changes of mathematical object.
- Never split a formula from the prose that introduces or interprets it.
- Do not follow source labels such as `Step 1` mechanically.
- Do not create one unit per sentence or list item when adjacent assertions share one local mathematical inference.
- Do not hide multiple independently falsifiable reasoning jumps inside one oversized unit.
- Preserve wrong and contradictory reasoning exactly. Boundary choices must not repair the argument or make its claims easier to formalize.
- Return end boundaries in strictly increasing order. The final boundary must be {{final_boundary}}.

Return exactly one `[[SOURCE_UNITS_V1]] ... [[/SOURCE_UNITS_V1]]` block. Inside the block, write only the boundary ID ending each unit, one per line. Write no prose, JSON, Markdown fence, or text outside the block.

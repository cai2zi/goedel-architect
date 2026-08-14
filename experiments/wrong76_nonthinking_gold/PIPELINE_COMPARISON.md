# Comparison with the Existing 397B Whole-COT Pipeline

This is an exact source-ID match against the frozen 76-record Gold set. The existing artifact reports `phase=phase1`; therefore its proof counters are descriptive provenance, not a same-stage prover accuracy comparison.

The final Gold revision includes a post-freeze reference-answer audit and is a comparison set, not a blind held-out evaluation set; see `REPORT.md` for provenance.

## Main observations

- Existing Phase 1: **45/76** are both structural `strictAccepted` and semantic `strictAccepted`; **31/76** are rejected before that point.
- On the accepted 45, the generated Blueprint has **671 nodes** (mean 14.91, median 14); Gold has **233 nodes** (mean 5.18, median 5). The generated graph is **2.88x** larger by mean node count.
- Among those 45 strict+semantic accepted inputs, **42/45** contain at least one Lean-verified Gold disproof. Gold target labels are `{'blocked_by_dependency': 31, 'disproved': 11, 'proved': 3}`.
- The existing artifact has **0/45** roots proved. It records 304/671 individual nodes proved, but this is not comparable to Gold label accuracy because node inventories differ.
- The 31 rejected inputs still admit compact Gold graphs (mean 4.77 nodes); 30/31 contain a verified disproof. Thus rejection is often a generation/translation failure, not an absence of a tractable formal decomposition.

## What the comparison says about the current pipeline

`strictAccepted` checks that a generated formal artifact is structurally usable and passes the configured semantic audit; it is not a mathematical-correctness label for the COT. The Gold set demonstrates that local, faithful counterexamples can survive those gates and should be first-class proof targets.

The largest practical gap is node design. The existing accepted graphs are substantially more fragmented, while Gold isolates one material claim per node and uses short arithmetic, finite enumeration, orientation tests, or explicit witnesses. This reduces both proof search depth and the amount of context a small prover must carry. Geometry and probability reductions remain explicit `formal_bridge` nodes instead of being hidden in definitions or expanded into monolithic theorems.

A second gap is polarity selection. A pipeline that always attempts the positive theorem first can spend most of its budget on a false node. Gold exposes a local negative witness as soon as a COT step is false, then marks downstream conclusions `blocked_by_dependency`. That is the behavior needed for Blueprint proofs to judge the COT rather than merely restate it.

## Matched inventory

| source_id | pipeline status | old nodes | old proved | Gold nodes | Gold disproved | Gold target |
|---|---|---:|---:|---:|---:|---|
| `MATH-500/test/counting_and_probability/430.json` | `strictAccepted/strictAccepted` | 13 | 7 | 5 | 1 | `disproved` |
| `MATH-500/test/counting_and_probability/731.json` | `structuralRejected/-` | 0 | 0 | 4 | 1 | `blocked_by_dependency` |
| `MATH-500/test/counting_and_probability/765.json` | `strictAccepted/strictAccepted` | 23 | 16 | 5 | 1 | `disproved` |
| `MATH-500/test/geometry/434.json` | `semanticRejected/-` | 0 | 0 | 4 | 1 | `blocked_by_dependency` |
| `MATH-500/test/geometry/465.json` | `semanticRejected/-` | 0 | 0 | 7 | 1 | `blocked_by_dependency` |
| `MATH-500/test/geometry/711.json` | `semanticRejected/-` | 0 | 0 | 9 | 1 | `blocked_by_dependency` |
| `MATH-500/test/geometry/817.json` | `strictAccepted/strictAccepted` | 7 | 2 | 5 | 1 | `disproved` |
| `MATH-500/test/geometry/826.json` | `strictAccepted/strictAccepted` | 17 | 9 | 12 | 1 | `blocked_by_dependency` |
| `MATH-500/test/geometry/880.json` | `strictAccepted/strictAccepted` | 20 | 13 | 10 | 1 | `blocked_by_dependency` |
| `MATH-500/test/intermediate_algebra/662.json` | `strictAccepted/strictAccepted` | 27 | 12 | 5 | 0 | `proved` |
| `MATH-500/test/intermediate_algebra/960.json` | `strictAccepted/strictAccepted` | 14 | 5 | 9 | 1 | `blocked_by_dependency` |
| `MATH-500/test/prealgebra/1003.json` | `structuralRejected/-` | 0 | 0 | 4 | 1 | `disproved` |
| `MATH-500/test/prealgebra/1139.json` | `strictAccepted/strictAccepted` | 19 | 4 | 6 | 1 | `blocked_by_dependency` |
| `MATH-500/test/prealgebra/1865.json` | `strictAccepted/strictAccepted` | 36 | 16 | 9 | 0 | `proved` |
| `MATH-500/test/prealgebra/378.json` | `strictAccepted/strictAccepted` | 6 | 2 | 6 | 1 | `blocked_by_dependency` |
| `MATH-500/test/prealgebra/874.json` | `strictAccepted/strictAccepted` | 8 | 4 | 5 | 2 | `blocked_by_dependency` |
| `MATH-500/test/precalculus/1056.json` | `strictAccepted/strictAccepted` | 12 | 3 | 7 | 1 | `disproved` |
| `MATH-500/test/precalculus/768.json` | `semanticRejected/-` | 0 | 0 | 5 | 0 | `proved` |
| `aime_2024/62` | `structuralRejected/-` | 0 | 0 | 8 | 1 | `blocked_by_dependency` |
| `aime_2024/73` | `strictAccepted/strictAccepted` | 10 | 4 | 6 | 1 | `blocked_by_dependency` |
| `aime_2024/81` | `structuralRejected/-` | 0 | 0 | 6 | 1 | `blocked_by_dependency` |
| `aime_2024/85` | `structuralRejected/-` | 0 | 0 | 3 | 1 | `disproved` |
| `aime_2024/88` | `strictAccepted/strictAccepted` | 11 | 4 | 7 | 1 | `disproved` |
| `aime_2024/89` | `structuralRejected/-` | 0 | 0 | 7 | 1 | `blocked_by_dependency` |
| `aime_2025/10` | `strictAccepted/strictAccepted` | 14 | 4 | 6 | 1 | `blocked_by_dependency` |
| `aime_2025/11` | `strictAccepted/strictAccepted` | 18 | 9 | 4 | 1 | `blocked_by_dependency` |
| `aime_2025/13` | `structuralRejected/-` | 0 | 0 | 4 | 1 | `blocked_by_dependency` |
| `aime_2025/14` | `strictAccepted/strictAccepted` | 28 | 9 | 4 | 1 | `blocked_by_dependency` |
| `aime_2025/15` | `strictAccepted/strictAccepted` | 14 | 5 | 4 | 1 | `blocked_by_dependency` |
| `aime_2025/20` | `structuralRejected/-` | 0 | 0 | 5 | 1 | `blocked_by_dependency` |
| `aime_2025/27` | `strictAccepted/strictAccepted` | 7 | 1 | 4 | 1 | `blocked_by_dependency` |
| `aime_2025/8` | `strictAccepted/strictAccepted` | 8 | 3 | 4 | 1 | `disproved` |
| `brumo_2025/12` | `strictAccepted/strictAccepted` | 26 | 13 | 4 | 1 | `blocked_by_dependency` |
| `brumo_2025/17` | `structuralRejected/-` | 0 | 0 | 5 | 1 | `blocked_by_dependency` |
| `brumo_2025/22` | `strictAccepted/strictAccepted` | 13 | 8 | 5 | 1 | `blocked_by_dependency` |
| `brumo_2025/3` | `strictAccepted/strictAccepted` | 23 | 15 | 5 | 1 | `blocked_by_dependency` |
| `brumo_2025/30` | `semanticRejected/-` | 0 | 0 | 3 | 1 | `blocked_by_dependency` |
| `brumo_2025/6` | `semanticRejected/-` | 0 | 0 | 4 | 1 | `blocked_by_dependency` |
| `cmimc_2025/11` | `strictAccepted/strictAccepted` | 18 | 10 | 5 | 1 | `disproved` |
| `cmimc_2025/13` | `structuralRejected/-` | 0 | 0 | 4 | 1 | `disproved` |
| `cmimc_2025/14` | `structuralRejected/-` | 0 | 0 | 7 | 1 | `blocked_by_dependency` |
| `cmimc_2025/15` | `structuralRejected/-` | 0 | 0 | 4 | 1 | `blocked_by_dependency` |
| `cmimc_2025/16` | `structuralRejected/-` | 0 | 0 | 4 | 1 | `disproved` |
| `cmimc_2025/18` | `structuralRejected/-` | 0 | 0 | 4 | 1 | `disproved` |
| `cmimc_2025/20` | `structuralRejected/-` | 0 | 0 | 4 | 1 | `blocked_by_dependency` |
| `cmimc_2025/21` | `structuralRejected/-` | 0 | 0 | 5 | 1 | `blocked_by_dependency` |
| `cmimc_2025/23` | `strictAccepted/strictAccepted` | 27 | 10 | 4 | 0 | `proved` |
| `cmimc_2025/25` | `strictAccepted/strictAccepted` | 20 | 9 | 10 | 1 | `blocked_by_dependency` |
| `cmimc_2025/27` | `strictAccepted/strictAccepted` | 1 | 0 | 3 | 1 | `blocked_by_dependency` |
| `cmimc_2025/30` | `semanticRejected/-` | 0 | 0 | 4 | 1 | `blocked_by_dependency` |
| `cmimc_2025/32` | `structuralRejected/-` | 0 | 0 | 4 | 1 | `blocked_by_dependency` |
| `cmimc_2025/34` | `strictAccepted/strictAccepted` | 11 | 6 | 7 | 1 | `blocked_by_dependency` |
| `cmimc_2025/35` | `strictAccepted/strictAccepted` | 17 | 5 | 5 | 1 | `blocked_by_dependency` |
| `cmimc_2025/37` | `strictAccepted/strictAccepted` | 10 | 1 | 4 | 1 | `blocked_by_dependency` |
| `cmimc_2025/38` | `strictAccepted/strictAccepted` | 14 | 5 | 3 | 1 | `blocked_by_dependency` |
| `cmimc_2025/39` | `strictAccepted/strictAccepted` | 20 | 19 | 7 | 1 | `blocked_by_dependency` |
| `cmimc_2025/40` | `structuralRejected/-` | 0 | 0 | 5 | 1 | `blocked_by_dependency` |
| `cmimc_2025/5` | `strictAccepted/strictAccepted` | 12 | 3 | 3 | 1 | `blocked_by_dependency` |
| `cmimc_2025/7` | `strictAccepted/strictAccepted` | 7 | 3 | 4 | 1 | `blocked_by_dependency` |
| `cmimc_2025/9` | `strictAccepted/strictAccepted` | 7 | 2 | 5 | 1 | `blocked_by_dependency` |
| `hmmt_feb_2025/10` | `strictAccepted/strictAccepted` | 19 | 1 | 4 | 1 | `blocked_by_dependency` |
| `hmmt_feb_2025/12` | `strictAccepted/strictAccepted` | 19 | 15 | 4 | 1 | `disproved` |
| `hmmt_feb_2025/13` | `strictAccepted/strictAccepted` | 16 | 10 | 3 | 1 | `blocked_by_dependency` |
| `hmmt_feb_2025/14` | `strictAccepted/strictAccepted` | 6 | 4 | 3 | 1 | `disproved` |
| `hmmt_feb_2025/15` | `strictAccepted/strictAccepted` | 18 | 8 | 7 | 1 | `blocked_by_dependency` |
| `hmmt_feb_2025/16` | `strictAccepted/strictAccepted` | 13 | 8 | 4 | 1 | `disproved` |
| `hmmt_feb_2025/17` | `structuralRejected/-` | 0 | 0 | 5 | 1 | `blocked_by_dependency` |
| `hmmt_feb_2025/18` | `structuralRejected/-` | 0 | 0 | 4 | 1 | `blocked_by_dependency` |
| `hmmt_feb_2025/19` | `semanticRejected/-` | 0 | 0 | 4 | 1 | `disproved` |
| `hmmt_feb_2025/20` | `structuralRejected/-` | 0 | 0 | 5 | 2 | `blocked_by_dependency` |
| `hmmt_feb_2025/24` | `structuralRejected/-` | 0 | 0 | 3 | 1 | `blocked_by_dependency` |
| `hmmt_feb_2025/25` | `strictAccepted/strictAccepted` | 11 | 5 | 3 | 1 | `blocked_by_dependency` |
| `hmmt_feb_2025/29` | `strictAccepted/strictAccepted` | 12 | 7 | 2 | 1 | `disproved` |
| `hmmt_feb_2025/30` | `semanticRejected/-` | 0 | 0 | 4 | 1 | `blocked_by_dependency` |
| `hmmt_feb_2025/4` | `strictAccepted/strictAccepted` | 10 | 2 | 4 | 1 | `blocked_by_dependency` |
| `hmmt_feb_2025/6` | `strictAccepted/strictAccepted` | 9 | 3 | 2 | 1 | `blocked_by_dependency` |

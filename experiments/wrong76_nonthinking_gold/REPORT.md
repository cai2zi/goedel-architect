# Wrong76 Non-Thinking Gold: Final Build Report

Frozen at `2026-08-14T09:00:38.911614+00:00` after a complete 76/76 per-record Lean replay.

## Aggregate result

- Records: **76/76 gold_complete**
- Material COT steps: **381**
- Active nodes: **381**
- Labels: **148 definition_valid**, **100 proved**, **74 disproved**, **59 blocked_by_dependency**
- Target labels: **4 proved**, **17 disproved**, **55 blocked_by_dependency**
- Records containing a Lean-verified disproof: **72/76**
- Mechanical checks: parse, whole-file Lean, canonical rebuild and Lean, Phase-2 contract, Phase-2 standalone, definition bundle, metadata/source-span contract, and independent proof/disproof replay.

## Interpretation boundary

A `proved` node means the closed Lean node represented in this Gold graph has a replayable proof. A `disproved` node has a replayable proof of the exact closed-theorem negation. A `blocked_by_dependency` target preserves the COT conclusion but does not award it a proof after an upstream claim is refuted. The deterministic suite is intentionally mechanical and does not claim to mechanize natural-language semantic equivalence.

The Gold graphs favor short arithmetic, finite witnesses, and local counterexamples. Geometry/probability reductions that are mathematically meaningful but expensive in Lean are isolated as explicit `formal_bridge` nodes and documented in each record. This is the intended comparison point for diagnosing whether the generation pipeline loses semantics or creates unnecessarily hard nodes.

## Authoring provenance

The initial 76-record build and first complete freeze used only the problem, the final non-thinking COT, and its claimed answer. A post-freeze audit then inspected extracted reference answers for the 16 records whose target was still marked `proved`; this exposed several cases where the first Gold graph had verified only terminal arithmetic. Those cases were repaired with explicit problem/COT contradictions and local witnesses. Consequently, the final revision is a Gold comparison set, not a strictly blind evaluation set. Reference extraction was used as an audit signal rather than accepted as truth: four targets remain `proved` because their written COT is verifiable under the agreed semantic source despite external answer/extraction disagreement.

## Per-record inventory

| source_id | steps | nodes | def | proved | disproved | blocked | target |
|---|---:|---:|---:|---:|---:|---:|---|
| `MATH-500/test/counting_and_probability/430.json` | 5 | 5 | 2 | 2 | 1 | 0 | `disproved` |
| `MATH-500/test/counting_and_probability/731.json` | 5 | 4 | 1 | 1 | 1 | 1 | `blocked_by_dependency` |
| `MATH-500/test/counting_and_probability/765.json` | 5 | 5 | 2 | 2 | 1 | 0 | `disproved` |
| `MATH-500/test/geometry/434.json` | 5 | 4 | 1 | 0 | 1 | 2 | `blocked_by_dependency` |
| `MATH-500/test/geometry/465.json` | 5 | 7 | 3 | 2 | 1 | 1 | `blocked_by_dependency` |
| `MATH-500/test/geometry/711.json` | 5 | 9 | 5 | 2 | 1 | 1 | `blocked_by_dependency` |
| `MATH-500/test/geometry/817.json` | 5 | 5 | 1 | 3 | 1 | 0 | `disproved` |
| `MATH-500/test/geometry/826.json` | 7 | 12 | 6 | 4 | 1 | 1 | `blocked_by_dependency` |
| `MATH-500/test/geometry/880.json` | 5 | 10 | 6 | 1 | 1 | 2 | `blocked_by_dependency` |
| `MATH-500/test/intermediate_algebra/662.json` | 6 | 5 | 2 | 3 | 0 | 0 | `proved` |
| `MATH-500/test/intermediate_algebra/960.json` | 6 | 9 | 6 | 1 | 1 | 1 | `blocked_by_dependency` |
| `MATH-500/test/prealgebra/1003.json` | 4 | 4 | 1 | 2 | 1 | 0 | `disproved` |
| `MATH-500/test/prealgebra/1139.json` | 5 | 6 | 3 | 1 | 1 | 1 | `blocked_by_dependency` |
| `MATH-500/test/prealgebra/1865.json` | 5 | 9 | 4 | 5 | 0 | 0 | `proved` |
| `MATH-500/test/prealgebra/378.json` | 5 | 6 | 2 | 2 | 1 | 1 | `blocked_by_dependency` |
| `MATH-500/test/prealgebra/874.json` | 5 | 5 | 1 | 0 | 2 | 2 | `blocked_by_dependency` |
| `MATH-500/test/precalculus/1056.json` | 5 | 7 | 5 | 1 | 1 | 0 | `disproved` |
| `MATH-500/test/precalculus/768.json` | 6 | 5 | 2 | 3 | 0 | 0 | `proved` |
| `aime_2024/62` | 6 | 8 | 5 | 1 | 1 | 1 | `blocked_by_dependency` |
| `aime_2024/73` | 9 | 6 | 2 | 1 | 1 | 2 | `blocked_by_dependency` |
| `aime_2024/81` | 5 | 6 | 2 | 2 | 1 | 1 | `blocked_by_dependency` |
| `aime_2024/85` | 4 | 3 | 1 | 1 | 1 | 0 | `disproved` |
| `aime_2024/88` | 5 | 7 | 5 | 1 | 1 | 0 | `disproved` |
| `aime_2024/89` | 4 | 7 | 4 | 1 | 1 | 1 | `blocked_by_dependency` |
| `aime_2025/10` | 5 | 6 | 3 | 1 | 1 | 1 | `blocked_by_dependency` |
| `aime_2025/11` | 5 | 4 | 1 | 1 | 1 | 1 | `blocked_by_dependency` |
| `aime_2025/13` | 5 | 4 | 2 | 0 | 1 | 1 | `blocked_by_dependency` |
| `aime_2025/14` | 6 | 4 | 0 | 2 | 1 | 1 | `blocked_by_dependency` |
| `aime_2025/15` | 5 | 4 | 1 | 1 | 1 | 1 | `blocked_by_dependency` |
| `aime_2025/20` | 5 | 5 | 2 | 1 | 1 | 1 | `blocked_by_dependency` |
| `aime_2025/27` | 5 | 4 | 2 | 0 | 1 | 1 | `blocked_by_dependency` |
| `aime_2025/8` | 7 | 4 | 1 | 2 | 1 | 0 | `disproved` |
| `brumo_2025/12` | 5 | 4 | 1 | 1 | 1 | 1 | `blocked_by_dependency` |
| `brumo_2025/17` | 5 | 5 | 1 | 2 | 1 | 1 | `blocked_by_dependency` |
| `brumo_2025/22` | 5 | 5 | 1 | 2 | 1 | 1 | `blocked_by_dependency` |
| `brumo_2025/3` | 5 | 5 | 2 | 1 | 1 | 1 | `blocked_by_dependency` |
| `brumo_2025/30` | 5 | 3 | 0 | 1 | 1 | 1 | `blocked_by_dependency` |
| `brumo_2025/6` | 5 | 4 | 2 | 0 | 1 | 1 | `blocked_by_dependency` |
| `cmimc_2025/11` | 5 | 5 | 2 | 2 | 1 | 0 | `disproved` |
| `cmimc_2025/13` | 4 | 4 | 1 | 2 | 1 | 0 | `disproved` |
| `cmimc_2025/14` | 3 | 7 | 3 | 2 | 1 | 1 | `blocked_by_dependency` |
| `cmimc_2025/15` | 5 | 4 | 1 | 1 | 1 | 1 | `blocked_by_dependency` |
| `cmimc_2025/16` | 5 | 4 | 2 | 1 | 1 | 0 | `disproved` |
| `cmimc_2025/18` | 4 | 4 | 2 | 1 | 1 | 0 | `disproved` |
| `cmimc_2025/20` | 5 | 4 | 2 | 0 | 1 | 1 | `blocked_by_dependency` |
| `cmimc_2025/21` | 5 | 5 | 2 | 1 | 1 | 1 | `blocked_by_dependency` |
| `cmimc_2025/23` | 5 | 4 | 2 | 2 | 0 | 0 | `proved` |
| `cmimc_2025/25` | 5 | 10 | 6 | 2 | 1 | 1 | `blocked_by_dependency` |
| `cmimc_2025/27` | 5 | 3 | 0 | 1 | 1 | 1 | `blocked_by_dependency` |
| `cmimc_2025/30` | 5 | 4 | 1 | 1 | 1 | 1 | `blocked_by_dependency` |
| `cmimc_2025/32` | 5 | 4 | 1 | 1 | 1 | 1 | `blocked_by_dependency` |
| `cmimc_2025/34` | 4 | 7 | 3 | 2 | 1 | 1 | `blocked_by_dependency` |
| `cmimc_2025/35` | 5 | 5 | 2 | 1 | 1 | 1 | `blocked_by_dependency` |
| `cmimc_2025/37` | 5 | 4 | 1 | 1 | 1 | 1 | `blocked_by_dependency` |
| `cmimc_2025/38` | 5 | 3 | 0 | 1 | 1 | 1 | `blocked_by_dependency` |
| `cmimc_2025/39` | 5 | 7 | 3 | 2 | 1 | 1 | `blocked_by_dependency` |
| `cmimc_2025/40` | 5 | 5 | 2 | 1 | 1 | 1 | `blocked_by_dependency` |
| `cmimc_2025/5` | 5 | 3 | 0 | 1 | 1 | 1 | `blocked_by_dependency` |
| `cmimc_2025/7` | 5 | 4 | 1 | 1 | 1 | 1 | `blocked_by_dependency` |
| `cmimc_2025/9` | 5 | 5 | 1 | 2 | 1 | 1 | `blocked_by_dependency` |
| `hmmt_feb_2025/10` | 4 | 4 | 2 | 0 | 1 | 1 | `blocked_by_dependency` |
| `hmmt_feb_2025/12` | 4 | 4 | 2 | 1 | 1 | 0 | `disproved` |
| `hmmt_feb_2025/13` | 4 | 3 | 0 | 1 | 1 | 1 | `blocked_by_dependency` |
| `hmmt_feb_2025/14` | 4 | 3 | 1 | 1 | 1 | 0 | `disproved` |
| `hmmt_feb_2025/15` | 5 | 7 | 3 | 2 | 1 | 1 | `blocked_by_dependency` |
| `hmmt_feb_2025/16` | 4 | 4 | 2 | 1 | 1 | 0 | `disproved` |
| `hmmt_feb_2025/17` | 6 | 5 | 2 | 1 | 1 | 1 | `blocked_by_dependency` |
| `hmmt_feb_2025/18` | 5 | 4 | 1 | 1 | 1 | 1 | `blocked_by_dependency` |
| `hmmt_feb_2025/19` | 5 | 4 | 2 | 1 | 1 | 0 | `disproved` |
| `hmmt_feb_2025/20` | 4 | 5 | 1 | 1 | 2 | 1 | `blocked_by_dependency` |
| `hmmt_feb_2025/24` | 5 | 3 | 1 | 0 | 1 | 1 | `blocked_by_dependency` |
| `hmmt_feb_2025/25` | 5 | 3 | 1 | 0 | 1 | 1 | `blocked_by_dependency` |
| `hmmt_feb_2025/29` | 6 | 2 | 0 | 1 | 1 | 0 | `disproved` |
| `hmmt_feb_2025/30` | 5 | 4 | 1 | 1 | 1 | 1 | `blocked_by_dependency` |
| `hmmt_feb_2025/4` | 5 | 4 | 0 | 2 | 1 | 1 | `blocked_by_dependency` |
| `hmmt_feb_2025/6` | 5 | 2 | 0 | 0 | 1 | 1 | `blocked_by_dependency` |

## Reproduction handles

- Machine-readable summary: `/ssd/czx/czx_work/wrong76_nonthinking_gold/final_summary.json`
- Frozen artifact hashes: `/ssd/czx/czx_work/wrong76_nonthinking_gold/freeze_manifest.json`
- Records and Lean files: `/ssd/czx/czx_work/wrong76_nonthinking_gold/records`

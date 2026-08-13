# Codex wrong76 four-case probe

Run date: 2026-08-13

Configuration: `codex exec --ephemeral --sandbox read-only --json`, model
`gpt-5.6-sol`, reasoning effort `medium`. Case 1 ran first; cases 2--4 ran
concurrently (peak Codex concurrency: 3).

| Case | ID / node | Expected | Existing pipeline | Codex | Independent check |
|---|---|---|---|---|---|
| 1 | `aime_2025/10` | reject false positive | accepted | reject | semantic evidence reviewed |
| 2 | `MATH-500/test/prealgebra/1139.json` | reject | rejected | reject | semantic evidence reviewed |
| 3 | `cmimc_2025/9::P_no_small_prime_divisor` | prove negation | negative probe failed | negation proved | Lean exit 0 |
| 4 | `brumo_2025/6::product_k1_eq` | prove negation | negative probe succeeded | negation proved | Lean exit 0 |

Selected-set accuracy: existing pipeline 2/4 (50%); Codex 4/4 (100%). This is
a deliberately stratified diagnostic set, not an unbiased estimate over all 76
examples.

| Case | Input | Cached input | Uncached input | Output | Reasoning output | Estimated credits |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 31,305 | 0 | 31,305 | 579 | 112 | 4.347375 |
| 2 | 29,954 | 13,056 | 16,898 | 1,207 | 586 | 3.180700 |
| 3 | 358,064 | 303,104 | 54,960 | 2,634 | 860 | 12.634300 |
| 4 | 454,170 | 416,768 | 37,402 | 3,175 | 1,322 | 12.266100 |
| **Total** | **873,493** | **732,928** | **140,565** | **7,595** | **2,880** | **32.428475** |

Credit estimate uses the 2026-08-13 GPT-5.6 Sol rate card: 125 credits per
million uncached input tokens, 12.5 per million cached input tokens, and 750 per
million output tokens. `reasoning_output_tokens` is a breakdown of output and is
not added a second time.

Raw artifacts are under
`/ssd/czx/czx_work/cot_blueprint_refine/codex_exec_wrong76_probe4_20260813/`.
Each case has a final JSON result, a JSONL event stream, and stderr log.

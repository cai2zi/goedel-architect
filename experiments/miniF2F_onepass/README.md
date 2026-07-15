# miniF2F One-Pass

Runs one blueprint extraction and one DAG proving pass for miniF2F rows. It calls
`run_phase1()` and `run_phase2()` only, and does not run refinement.

Example:

```bash
export OPENAI_BASE_URL=https://poloai.top/v1
export GOEDEL_OPENAI_BASE_URL=https://poloai.top/v1
export GOEDEL_BLUEPRINT_MAX_TOKENS=262144
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897

python experiments/miniF2F_onepass/run_minif2f_onepass.py \
  --model deepseek-v4-flash \
  --split test \
  --limit 2
```

Outputs are written by default to:

```text
czx_work/goedel-architect/miniF2F_onepass/deepseek_v4_flash/<split>/
```


export GEMINI_API_KEY=
export GEMINI_API_URL="https://poloai.top/v1beta/models/gemini-2.5-flash:generateContent"

python goedel-architect/experiments/robustpa_refine/evaluate_stmt_sc.py \
  --input /ssd/czx/czx_work/robustpa_eval/qwen3_5_397b_MiniF2F_orig_reExp/stmt_sc_inputs.jsonl \
  --workers 256
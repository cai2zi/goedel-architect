#!/usr/bin/env bash
set -e

# global_original
python experiments/robustpa_refine/run_robustpa_refine.py --exp-name qwen3_5_397b_MiniF2F_orig_reExp44 --model Qwen3.5-397B-A17B-FP8 --openai-base-url http://127.0.0.1:8001/v1 --split miniF2F --subset global_original
python experiments/robustpa_refine/run_robustpa_refine.py --exp-name qwen3_5_397b_math500_orig_reExp44 --model Qwen3.5-397B-A17B-FP8 --openai-base-url http://127.0.0.1:8001/v1 --split MATH500 --subset global_original

# global_gemini_rephrase
python experiments/robustpa_refine/run_robustpa_refine.py --exp-name qwen3_5_397b_MiniF2F_global_gemini_rephrase_reExp44 --model Qwen3.5-397B-A17B-FP8 --openai-base-url http://127.0.0.1:8001/v1 --split miniF2F --subset global_gemini_rephrase
python experiments/robustpa_refine/run_robustpa_refine.py --exp-name qwen3_5_397b_math500_global_gemini_rephrase_reExp44 --model Qwen3.5-397B-A17B-FP8 --openai-base-url http://127.0.0.1:8001/v1 --split MATH500 --subset global_gemini_rephrase

# global_gemini_step
python experiments/robustpa_refine/run_robustpa_refine.py --exp-name qwen3_5_397b_MiniF2F_global_gemini_step_reExp44 --model Qwen3.5-397B-A17B-FP8 --openai-base-url http://127.0.0.1:8001/v1 --split miniF2F --subset global_gemini_step
python experiments/robustpa_refine/run_robustpa_refine.py --exp-name qwen3_5_397b_math500_global_gemini_step_reExp44 --model Qwen3.5-397B-A17B-FP8 --openai-base-url http://127.0.0.1:8001/v1 --split MATH500 --subset global_gemini_step

# global_qwen3_rephrase
python experiments/robustpa_refine/run_robustpa_refine.py --exp-name qwen3_5_397b_MiniF2F_global_qwen3_rephrase_reExp44 --model Qwen3.5-397B-A17B-FP8 --openai-base-url http://127.0.0.1:8001/v1 --split miniF2F --subset global_qwen3_rephrase
python experiments/robustpa_refine/run_robustpa_refine.py --exp-name qwen3_5_397b_math500_global_qwen3_rephrase_reExp44 --model Qwen3.5-397B-A17B-FP8 --openai-base-url http://127.0.0.1:8001/v1 --split MATH500 --subset global_qwen3_rephrase

# global_qwen3_step
python experiments/robustpa_refine/run_robustpa_refine.py --exp-name qwen3_5_397b_MiniF2F_global_qwen3_step_reExp44 --model Qwen3.5-397B-A17B-FP8 --openai-base-url http://127.0.0.1:8001/v1 --split miniF2F --subset global_qwen3_step
python experiments/robustpa_refine/run_robustpa_refine.py --exp-name qwen3_5_397b_math500_global_qwen3_step_reExp44 --model Qwen3.5-397B-A17B-FP8 --openai-base-url http://127.0.0.1:8001/v1 --split MATH500 --subset global_qwen3_step

# local_number_edit_proof
python experiments/robustpa_refine/run_robustpa_refine.py --exp-name qwen3_5_397b_MiniF2F_local_number_edit_proof_reExp44 --model Qwen3.5-397B-A17B-FP8 --openai-base-url http://127.0.0.1:8001/v1 --split miniF2F --subset local_number_edit_proof
python experiments/robustpa_refine/run_robustpa_refine.py --exp-name qwen3_5_397b_math500_local_number_edit_proof_reExp44 --model Qwen3.5-397B-A17B-FP8 --openai-base-url http://127.0.0.1:8001/v1 --split MATH500 --subset local_number_edit_proof

# local_number_edit_statement
python experiments/robustpa_refine/run_robustpa_refine.py --exp-name qwen3_5_397b_MiniF2F_local_number_edit_statement_reExp44 --model Qwen3.5-397B-A17B-FP8 --openai-base-url http://127.0.0.1:8001/v1 --split miniF2F --subset local_number_edit_statement
python experiments/robustpa_refine/run_robustpa_refine.py --exp-name qwen3_5_397b_math500_local_number_edit_statement_reExp44 --model Qwen3.5-397B-A17B-FP8 --openai-base-url http://127.0.0.1:8001/v1 --split MATH500 --subset local_number_edit_statement

# local_step_delete
python experiments/robustpa_refine/run_robustpa_refine.py --exp-name qwen3_5_397b_MiniF2F_local_step_delete_reExp44 --model Qwen3.5-397B-A17B-FP8 --openai-base-url http://127.0.0.1:8001/v1 --split miniF2F --subset local_step_delete
python experiments/robustpa_refine/run_robustpa_refine.py --exp-name qwen3_5_397b_math500_local_step_delete_reExp44 --model Qwen3.5-397B-A17B-FP8 --openai-base-url http://127.0.0.1:8001/v1 --split MATH500 --subset local_step_delete

# local_symbol_edit_proof
python experiments/robustpa_refine/run_robustpa_refine.py --exp-name qwen3_5_397b_MiniF2F_local_symbol_edit_proof_reExp44 --model Qwen3.5-397B-A17B-FP8 --openai-base-url http://127.0.0.1:8001/v1 --split miniF2F --subset local_symbol_edit_proof
python experiments/robustpa_refine/run_robustpa_refine.py --exp-name qwen3_5_397b_math500_local_symbol_edit_proof_reExp44 --model Qwen3.5-397B-A17B-FP8 --openai-base-url http://127.0.0.1:8001/v1 --split MATH500 --subset local_symbol_edit_proof

# local_symbol_edit_statement
python experiments/robustpa_refine/run_robustpa_refine.py --exp-name qwen3_5_397b_MiniF2F_local_symbol_edit_statement_reExp44 --model Qwen3.5-397B-A17B-FP8 --openai-base-url http://127.0.0.1:8001/v1 --split miniF2F --subset local_symbol_edit_statement
python experiments/robustpa_refine/run_robustpa_refine.py --exp-name qwen3_5_397b_math500_local_symbol_edit_statement_reExp44 --model Qwen3.5-397B-A17B-FP8 --openai-base-url http://127.0.0.1:8001/v1 --split MATH500 --subset local_symbol_edit_statement

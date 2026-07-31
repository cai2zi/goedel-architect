#!/usr/bin/env bash
set -euo pipefail

OUT="/ssd/czx/robustpa_refine_proofrate_context.md"
REPO_ROOT="/ssd/czx/goedel-architect"

cd "${REPO_ROOT}"
mkdir -p "$(dirname "${OUT}")"

fence_lang() {
  case "$1" in
    *.py) printf 'python' ;;
    *.sh) printf 'bash' ;;
    *.yaml|*.yml) printf 'yaml' ;;
    *.md) printf 'markdown' ;;
    *.diff) printf 'diff' ;;
    *) printf 'text' ;;
  esac
}

append_command() {
  local title="$1"
  shift
  {
    printf '\n## %s\n\n' "${title}"
    printf '```text\n'
    "$@" || true
    printf '```\n'
  } >> "${OUT}"
}

append_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    {
      printf '\n## File: %s\n\n' "${path}"
      printf '```text\n'
      printf 'MISSING: %s\n' "${path}"
      printf '```\n'
    } >> "${OUT}"
    return
  fi

  {
    printf '\n## File: %s\n\n' "${path}"
    printf '```%s\n' "$(fence_lang "${path}")"
    sed -n '1,$p' "${path}"
    printf '\n```\n'
  } >> "${OUT}"
}

FILES=(
  "experiments/robustpa_refine/run_robustpa_refine.py"
  "experiments/robustpa_refine/run_all_robustpa_reexp.sh"
  "experiments/robustpa_refine/configs/base.yaml"
  "experiments/robustpa_refine/configs/phase2_context_debug.yaml"
  "src/pipeline.py"
  "src/orchestrator.py"
  "src/blueprint.py"
  "src/blueprint_text.py"
  "src/refinement.py"
  "src/prover.py"
  "src/checkpoint.py"
  "src/tracer.py"
  "src/lean_compiler.py"
  "src/kimina_lean_compiler.py"
  "src/mathlib_retrieval.py"
  "src/llm_client.py"
  "src/goedel_prompts.py"
  "experiments/shared/lean_runtime.py"
  "experiments/shared/io_utils.py"
)

PROMPTS=(
  "prompts/robustpa_blueprint_system.md"
  "prompts/robustpa_blueprint_user.md"
  "prompts/blueprint_system.md"
  "prompts/blueprint_user.md"
  "prompts/prover_system.md"
  "prompts/prover_user.md"
  "prompts/refinement_system.md"
  "prompts/refinement_user.md"
)

: > "${OUT}"

{
  printf '# RobustPA Refine Proof-Rate Context\n\n'
  printf 'Generated at: %s\n\n' "$(date -Is)"
  printf 'Repository: %s\n\n' "${REPO_ROOT}"
  printf 'Purpose: full-code context bundle for analyzing and improving robustpa_refine overall proof rate.\n'
} >> "${OUT}"

append_command "Git Status" git status --short
append_command "Git Diff Stat" git diff --stat
append_command "Recent Commits" git log --oneline -n 12

append_command \
  "Proof-Rate Related Search Hits" \
  rg -n "max_retries|node_max_prove|negation|infra_error|timeout|success|done|proved_cache|failed_nodes|refine|contract_check|parallel_tool_calls|tool_choice|reasoning_effort|GOEDEL_" \
    src experiments/robustpa_refine prompts

append_command \
  "Main Import Graph Hints" \
  rg -n "^(from|import) " \
    experiments/robustpa_refine/run_robustpa_refine.py \
    src/pipeline.py \
    src/orchestrator.py \
    src/blueprint.py \
    src/refinement.py \
    src/prover.py

{
  printf '\n# Full Files\n'
  printf '\nThe files below are included in full, not snippet-extracted.\n'
} >> "${OUT}"

for file in "${FILES[@]}"; do
  append_file "${file}"
done

{
  printf '\n# Prompt Files\n'
  printf '\nPrompt files below are included in full because prompt wording directly affects proof rate.\n'
} >> "${OUT}"

for file in "${PROMPTS[@]}"; do
  append_file "${file}"
done

{
  printf '\n# Local Diff\n\n'
  printf '```diff\n'
  git diff || true
  printf '```\n'
} >> "${OUT}"

printf '[context] wrote %s\n' "${OUT}"

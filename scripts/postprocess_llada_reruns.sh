#!/usr/bin/env bash
# Post-process the corrected LLaDA re-runs (qwen HE/MBPP, deepseek MBPP).
# Run this after the three gen_remask jobs finish.
#
# Jobs being waited on:
#   GPU1: qwen_llada_remask_humaneval_t0.9.jsonl
#   GPU7: qwen_llada_remask_mbpp_t0.9.jsonl
#   GPU6: deepseek_llada_remask_mbpp_t0.9.jsonl
#
# Usage: bash scripts/postprocess_llada_reruns.sh

set -euo pipefail
ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

source /home/tteng/miniconda3/etc/profile.d/conda.sh
conda activate code
export PYTHONPATH="${ROOT}/src"

OUT="outputs/base_tuteng"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
wait_for_lock_clear() {
  local jsonl="$1"
  local lock="${jsonl}.lock"
  echo "[wait] waiting for $lock to clear..."
  while [[ -f "$lock" ]]; do
    sleep 30
  done
  echo "[wait] $lock cleared."
}

sanitize_and_eval() {
  local dataset="$1"
  local jsonl="$2"
  local summary_out="$3"
  echo "[sanitize+eval] $jsonl"
  python -m coder.scripts.postprocess_evalplus \
    --dataset "$dataset" \
    --samples "$jsonl"
  local sanitized="${jsonl%.jsonl}-sanitized.jsonl"
  python -m coder.scripts.eval_evalplus \
    --backend local \
    --dataset "$dataset" \
    --samples "$sanitized" \
    --summary_out "$summary_out"
}

# ---------------------------------------------------------------------------
# Wait for each job, then sanitize+eval
# ---------------------------------------------------------------------------

echo "=== Waiting for qwen+llada HE ==="
wait_for_lock_clear "${OUT}/qwen_llada_remask_humaneval_t0.9.jsonl"
sanitize_and_eval humaneval \
  "${OUT}/qwen_llada_remask_humaneval_t0.9.jsonl" \
  "${OUT}/qwen_llada_remask_humaneval_t0.9_summary.json"

echo "=== Waiting for qwen+llada MBPP ==="
wait_for_lock_clear "${OUT}/qwen_llada_remask_mbpp_t0.9.jsonl"
sanitize_and_eval mbpp \
  "${OUT}/qwen_llada_remask_mbpp_t0.9.jsonl" \
  "${OUT}/qwen_llada_remask_mbpp_t0.9_summary.json"

echo "=== Waiting for deepseek+llada MBPP ==="
wait_for_lock_clear "${OUT}/deepseek_llada_remask_mbpp_t0.9.jsonl"
sanitize_and_eval mbpp \
  "${OUT}/deepseek_llada_remask_mbpp_t0.9.jsonl" \
  "${OUT}/deepseek_llada_remask_mbpp_t0.9_summary.json"

# ---------------------------------------------------------------------------
# Rebuild model_pairs JSON and results.md
# ---------------------------------------------------------------------------
echo "=== Regenerating model_pairs_all_t0.9.json ==="
python -m coder.scripts.model_pairs_evalplus

echo "=== Regenerating results.md ==="
python -m coder.scripts.gen_results_table

echo "=== All done ==="

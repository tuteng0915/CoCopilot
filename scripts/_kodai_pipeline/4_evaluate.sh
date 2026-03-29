#!/usr/bin/env bash
ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

# 4_evaluate.sh — Run EvalPlus evaluation on all sanitized outputs.
# Requires sanitized .jsonl files from step 3.
# Usage: THRESHOLDS="0.7 0.8 0.9" bash scripts/4_evaluate.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}

THRESHOLDS=${THRESHOLDS:-"0.7 0.8 0.9"}

# Evaluate DeepSeek sanitized outputs
for DATASET in humaneval mbpp; do
  echo "=== Evaluating: deepseek_${DATASET} ==="
  python -m coder.scripts.eval_evalplus \
    --backend local \
    --dataset "${DATASET}" \
    --samples "outputs/deepseek_${DATASET}-sanitized.jsonl"
done

# Evaluate remasked sanitized outputs
for THRESH in ${THRESHOLDS}; do
  for DATASET in humaneval mbpp; do
    echo "=== Evaluating: remask_${DATASET}_t${THRESH} ==="
    python -m coder.scripts.eval_evalplus \
      --backend local \
      --dataset "${DATASET}" \
      --samples "outputs/remask_${DATASET}_t${THRESH}-sanitized.jsonl" \
      --summary_out "outputs/remask_${DATASET}_t${THRESH}_summary.json"
  done
done

echo "=== Evaluation done ==="

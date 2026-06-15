#!/usr/bin/env bash
ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

# 3_sanitize.sh — Syntax-check and sanitize all .jsonl outputs for EvalPlus evaluation.
# Sanitizes DeepSeek outputs and all remasked outputs matching outputs/remask_*.jsonl.
# Usage: THRESHOLDS="0.7 0.8 0.9" bash scripts/3_sanitize.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}

THRESHOLDS=${THRESHOLDS:-"0.7 0.8 0.9"}

# Sanitize DeepSeek outputs
for DATASET in humaneval mbpp; do
  echo "=== Sanitizing: deepseek_${DATASET} ==="
  python -m coder.scripts.postprocess_evalplus \
    --dataset "${DATASET}" \
    --samples "outputs/deepseek_${DATASET}.jsonl"
done

# Sanitize remasked outputs
for THRESH in ${THRESHOLDS}; do
  for DATASET in humaneval mbpp; do
    echo "=== Sanitizing: remask_${DATASET}_t${THRESH} ==="
    python -m coder.scripts.postprocess_evalplus \
      --dataset "${DATASET}" \
      --samples "outputs/remask_${DATASET}_t${THRESH}.jsonl"
  done
done

echo "=== Sanitization done ==="

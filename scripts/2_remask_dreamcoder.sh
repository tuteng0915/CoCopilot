#!/usr/bin/env bash
# 2_remask_dreamcoder.sh — Refine DeepSeek drafts with DreamCoder remasking.
# Requires: outputs/deepseek_{humaneval,mbpp}.jsonl from step 1.
# Usage: THRESHOLDS="0.7 0.8 0.9" bash scripts/2_remask_dreamcoder.sh
set -euo pipefail

export PYTHONPATH=/home/kodai/CoCopilot/src
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}

THRESHOLDS=${THRESHOLDS:-"0.7 0.8 0.9"}

for THRESH in ${THRESHOLDS}; do
  for DATASET in humaneval mbpp; do
    echo "=== Remasking: ${DATASET}, threshold=${THRESH} ==="
    python scripts/gen_remask.py \
      --input "outputs/deepseek_${DATASET}.jsonl" \
      --out   "outputs/remask_${DATASET}_t${THRESH}.jsonl" \
      --confidence_threshold "${THRESH}" \
      --temperature 0.0 --top_p 1.0 --seed 3407 \
      --resume
  done
done

echo "=== Remasking done ==="

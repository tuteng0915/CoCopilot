#!/usr/bin/env bash
ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

# 5_pipeline.sh — Full pipeline: DeepSeek generation → DreamCoder remasking → sanitize → evaluate.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}

THRESHOLDS=${THRESHOLDS:-"0.93 0.95 0.97 0.99"}
SCRIPTS=$(dirname "$0")

echo "========================================"
echo " Step 1: DeepSeek generation"
echo "========================================"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} bash "${SCRIPTS}/1_generate_deepseek.sh"

echo "========================================"
echo " Step 2: DreamCoder remasking"
echo "========================================"
THRESHOLDS="${THRESHOLDS}" CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} bash "${SCRIPTS}/2_remask_dreamcoder.sh"

echo "========================================"
echo " Step 3: Sanitize"
echo "========================================"
THRESHOLDS="${THRESHOLDS}" bash "${SCRIPTS}/3_sanitize.sh"

echo "========================================"
echo " Step 4: Evaluate"
echo "========================================"
THRESHOLDS="${THRESHOLDS}" bash "${SCRIPTS}/4_evaluate.sh"

echo "========================================"
echo " Pipeline complete!"
echo "========================================"

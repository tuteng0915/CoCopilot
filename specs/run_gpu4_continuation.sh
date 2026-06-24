#!/bin/bash
# GPU 4 continuation: extra DeepSeek MBPP tau + CodeLlama MBPP tau ablation

set -e
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cocoder

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=4
cd "${ROOT_DIR}"

LOG="outputs/tau_rerun/gpu4_continuation.log"
exec > >(tee -a "${LOG}") 2>&1
echo "=== GPU4 Continuation start: $(date) ==="

# ── DeepSeek extra MBPP tau (fill in the 0.5→0.7 transition zone) ───────────
echo "--- DeepSeek MBPP extra tau: 0.1 0.2 0.3 0.4 0.6 ---"
for THRESH in 0.1 0.2 0.3 0.4 0.6; do
  echo "  tau=${THRESH}: $(date)"
  python -m coder.scripts.gen_remask \
    --refiner dream \
    --input  outputs/base_tuteng/deepseek_mbpp.jsonl \
    --out    outputs/tau_rerun/remask_mbpp_t${THRESH}.jsonl \
    --confidence_threshold ${THRESH} \
    --seed 3407 --resume

  python -m coder.scripts.postprocess_evalplus \
    --dataset mbpp \
    --samples outputs/tau_rerun/remask_mbpp_t${THRESH}.jsonl

  python -m coder.scripts.eval_evalplus \
    --backend local \
    --dataset mbpp \
    --samples outputs/tau_rerun/remask_mbpp_t${THRESH}-sanitized.jsonl \
    --summary_out outputs/tau_rerun/remask_mbpp_t${THRESH}_summary.json
done

# ── CodeLlama MBPP tau ablation ─────────────────────────────────────────────
echo ""
echo "--- CodeLlama MBPP tau: 0.5 0.7 0.9 ---"
for THRESH in 0.5 0.7 0.9; do
  echo "  tau=${THRESH}: $(date)"
  python -m coder.scripts.gen_remask \
    --refiner dream \
    --input  outputs/base_tuteng/codellama_mbpp.jsonl \
    --out    outputs/tau_rerun/codellama_remask_mbpp_t${THRESH}.jsonl \
    --confidence_threshold ${THRESH} \
    --seed 3407 --resume

  python -m coder.scripts.postprocess_evalplus \
    --dataset mbpp \
    --samples outputs/tau_rerun/codellama_remask_mbpp_t${THRESH}.jsonl

  python -m coder.scripts.eval_evalplus \
    --backend local \
    --dataset mbpp \
    --samples outputs/tau_rerun/codellama_remask_mbpp_t${THRESH}-sanitized.jsonl \
    --summary_out outputs/tau_rerun/codellama_remask_mbpp_t${THRESH}_summary.json
done

echo ""
echo "=== GPU4 Continuation DONE: $(date) ==="

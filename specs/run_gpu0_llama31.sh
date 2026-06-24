#!/bin/bash
# GPU 0: Llama-3.1 CoCoder tau ablation -- HumanEval then MBPP

set -e
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cocoder

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
cd "${ROOT_DIR}"

LOG="outputs/tau_rerun/gpu0_llama31.log"
exec > >(tee -a "${LOG}") 2>&1
echo "=== GPU0 Llama-3.1 start: $(date) ==="

# ── Llama-3.1 HumanEval ─────────────────────────────────────────────────────
echo "--- Llama-3.1 HumanEval tau: 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 ---"
for THRESH in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
  echo "  tau=${THRESH}: $(date)"
  python -m coder.scripts.gen_remask \
    --refiner dream \
    --input  outputs/base_tuteng/llama31_humaneval.jsonl \
    --out    outputs/tau_rerun/llama31_remask_humaneval_t${THRESH}.jsonl \
    --confidence_threshold ${THRESH} \
    --seed 3407 --resume

  python -m coder.scripts.postprocess_evalplus \
    --dataset humaneval \
    --samples outputs/tau_rerun/llama31_remask_humaneval_t${THRESH}.jsonl

  python -m coder.scripts.eval_evalplus \
    --backend local \
    --dataset humaneval \
    --samples outputs/tau_rerun/llama31_remask_humaneval_t${THRESH}-sanitized.jsonl \
    --summary_out outputs/tau_rerun/llama31_remask_humaneval_t${THRESH}_summary.json
done

# ── Llama-3.1 MBPP ──────────────────────────────────────────────────────────
echo ""
echo "--- Llama-3.1 MBPP tau: 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 ---"
for THRESH in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
  echo "  tau=${THRESH}: $(date)"
  python -m coder.scripts.gen_remask \
    --refiner dream \
    --input  outputs/base_tuteng/llama31_mbpp.jsonl \
    --out    outputs/tau_rerun/llama31_remask_mbpp_t${THRESH}.jsonl \
    --confidence_threshold ${THRESH} \
    --seed 3407 --resume

  python -m coder.scripts.postprocess_evalplus \
    --dataset mbpp \
    --samples outputs/tau_rerun/llama31_remask_mbpp_t${THRESH}.jsonl

  python -m coder.scripts.eval_evalplus \
    --backend local \
    --dataset mbpp \
    --samples outputs/tau_rerun/llama31_remask_mbpp_t${THRESH}-sanitized.jsonl \
    --summary_out outputs/tau_rerun/llama31_remask_mbpp_t${THRESH}_summary.json
done

echo ""
echo "=== GPU0 Llama-3.1 DONE: $(date) ==="

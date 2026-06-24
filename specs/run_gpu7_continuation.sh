#!/bin/bash
# GPU 7 continuation: Qwen MBPP tau ablation + Qwen & CodeLlama HumanEval/MBPP broader tau

set -e
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cocoder

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=7
cd "${ROOT_DIR}"

LOG="outputs/tau_rerun/gpu7_continuation.log"
exec > >(tee -a "${LOG}") 2>&1
echo "=== GPU7 Continuation start: $(date) ==="

# ── Qwen MBPP tau ablation ──────────────────────────────────────────────────
echo "--- Qwen MBPP tau: 0.5 0.7 0.9 ---"
for THRESH in 0.5 0.7 0.9; do
  echo "  tau=${THRESH}: $(date)"
  python -m coder.scripts.gen_remask \
    --refiner dream \
    --input  outputs/base_tuteng/qwen_mbpp.jsonl \
    --out    outputs/tau_rerun/qwen_remask_mbpp_t${THRESH}.jsonl \
    --confidence_threshold ${THRESH} \
    --seed 3407 --resume

  python -m coder.scripts.postprocess_evalplus \
    --dataset mbpp \
    --samples outputs/tau_rerun/qwen_remask_mbpp_t${THRESH}.jsonl

  python -m coder.scripts.eval_evalplus \
    --backend local \
    --dataset mbpp \
    --samples outputs/tau_rerun/qwen_remask_mbpp_t${THRESH}-sanitized.jsonl \
    --summary_out outputs/tau_rerun/qwen_remask_mbpp_t${THRESH}_summary.json
done

# ── Broader tau for CodeLlama HumanEval (fill transition zone) ──────────────
echo ""
echo "--- CodeLlama HumanEval extra tau: 0.1 0.2 0.3 0.4 0.6 ---"
for THRESH in 0.1 0.2 0.3 0.4 0.6; do
  echo "  tau=${THRESH}: $(date)"
  python -m coder.scripts.gen_remask \
    --refiner dream \
    --input  outputs/base_tuteng/codellama_humaneval.jsonl \
    --out    outputs/tau_rerun/codellama_remask_humaneval_t${THRESH}.jsonl \
    --confidence_threshold ${THRESH} \
    --seed 3407 --resume

  python -m coder.scripts.postprocess_evalplus \
    --dataset humaneval \
    --samples outputs/tau_rerun/codellama_remask_humaneval_t${THRESH}.jsonl

  python -m coder.scripts.eval_evalplus \
    --backend local \
    --dataset humaneval \
    --samples outputs/tau_rerun/codellama_remask_humaneval_t${THRESH}-sanitized.jsonl \
    --summary_out outputs/tau_rerun/codellama_remask_humaneval_t${THRESH}_summary.json
done

echo ""
echo "=== GPU7 Continuation DONE: $(date) ==="

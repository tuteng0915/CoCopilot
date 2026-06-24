#!/bin/bash
# GPU 1 continuation: extra DeepSeek HumanEval tau + CodeLlama & Qwen HumanEval tau

set -e
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cocoder

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=1
cd "${ROOT_DIR}"

LOG="outputs/tau_rerun/gpu1_continuation.log"
exec > >(tee -a "${LOG}") 2>&1
echo "=== GPU1 Continuation start: $(date) ==="

# ── DeepSeek extra tau (fill in the 0.5→0.7 transition zone) ────────────────
echo "--- DeepSeek HumanEval extra tau: 0.1 0.2 0.3 0.4 0.6 ---"
for THRESH in 0.1 0.2 0.3 0.4 0.6; do
  echo "  tau=${THRESH}: $(date)"
  python -m coder.scripts.gen_remask \
    --refiner dream \
    --input  outputs/base_tuteng/deepseek_humaneval.jsonl \
    --out    outputs/tau_rerun/remask_humaneval_t${THRESH}.jsonl \
    --confidence_threshold ${THRESH} \
    --seed 3407 --resume

  python -m coder.scripts.postprocess_evalplus \
    --dataset humaneval \
    --samples outputs/tau_rerun/remask_humaneval_t${THRESH}.jsonl

  python -m coder.scripts.eval_evalplus \
    --backend local \
    --dataset humaneval \
    --samples outputs/tau_rerun/remask_humaneval_t${THRESH}-sanitized.jsonl \
    --summary_out outputs/tau_rerun/remask_humaneval_t${THRESH}_summary.json
done

# ── CodeLlama HumanEval tau ablation ────────────────────────────────────────
echo ""
echo "--- CodeLlama HumanEval tau: 0.5 0.7 0.9 ---"
for THRESH in 0.5 0.7 0.9; do
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

# ── Qwen HumanEval tau ablation ─────────────────────────────────────────────
echo ""
echo "--- Qwen HumanEval tau: 0.5 0.7 0.9 ---"
for THRESH in 0.5 0.7 0.9; do
  echo "  tau=${THRESH}: $(date)"
  python -m coder.scripts.gen_remask \
    --refiner dream \
    --input  outputs/base_tuteng/qwen_humaneval.jsonl \
    --out    outputs/tau_rerun/qwen_remask_humaneval_t${THRESH}.jsonl \
    --confidence_threshold ${THRESH} \
    --seed 3407 --resume

  python -m coder.scripts.postprocess_evalplus \
    --dataset humaneval \
    --samples outputs/tau_rerun/qwen_remask_humaneval_t${THRESH}.jsonl

  python -m coder.scripts.eval_evalplus \
    --backend local \
    --dataset humaneval \
    --samples outputs/tau_rerun/qwen_remask_humaneval_t${THRESH}-sanitized.jsonl \
    --summary_out outputs/tau_rerun/qwen_remask_humaneval_t${THRESH}_summary.json
done

echo ""
echo "=== GPU1 Continuation DONE: $(date) ==="

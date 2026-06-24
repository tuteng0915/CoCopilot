#!/bin/bash
# GPU 4: fill missing tau points -- CodeLlama MBPP + Qwen HumanEval + Qwen MBPP

set -e
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cocoder

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=4
cd "${ROOT_DIR}"

LOG="outputs/tau_rerun/gpu4_fill.log"
exec > >(tee -a "${LOG}") 2>&1
echo "=== GPU4 fill start: $(date) ==="

# ── CodeLlama MBPP (currently only 0.5, 0.7, 0.9 exist) ─────────────────────
echo "--- CodeLlama MBPP tau fill: 0.1 0.2 0.3 0.4 0.6 0.8 ---"
for THRESH in 0.1 0.2 0.3 0.4 0.6 0.8; do
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

# ── Qwen HumanEval (currently only 0.5, 0.7, 0.9 exist) ─────────────────────
echo ""
echo "--- Qwen HumanEval tau fill: 0.1 0.2 0.3 0.4 0.6 0.8 ---"
for THRESH in 0.1 0.2 0.3 0.4 0.6 0.8; do
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

# ── Qwen MBPP (currently only 0.5, 0.7, 0.9 exist) ──────────────────────────
echo ""
echo "--- Qwen MBPP tau fill: 0.1 0.2 0.3 0.4 0.6 0.8 ---"
for THRESH in 0.1 0.2 0.3 0.4 0.6 0.8; do
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

echo ""
echo "=== GPU4 fill DONE: $(date) ==="

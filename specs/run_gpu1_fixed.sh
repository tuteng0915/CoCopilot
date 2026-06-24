#!/bin/bash
# GPU 1: re-run with fixed build_evalplus_solution (preserves completion imports)

set -e
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cocoder

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=1
cd "${ROOT_DIR}"

LOG="outputs/tau_rerun/gpu1_fixed.log"
exec > >(tee -a "${LOG}") 2>&1
echo "=== GPU1 fixed start: $(date) ==="

# ── Qwen HumanEval τ=0.9 (fixed pipeline) ────────────────────────────────────
echo "--- Qwen HumanEval tau=0.9 fixed: $(date) ---"
python -m coder.scripts.gen_remask \
  --refiner dream \
  --input  outputs/base_tuteng/qwen_humaneval.jsonl \
  --out    outputs/tau_rerun/qwen_remask_humaneval_t0.9_fixed.jsonl \
  --confidence_threshold 0.9 \
  --seed 3407

python -m coder.scripts.postprocess_evalplus \
  --dataset humaneval \
  --samples outputs/tau_rerun/qwen_remask_humaneval_t0.9_fixed.jsonl

python -m coder.scripts.eval_evalplus \
  --backend local \
  --dataset humaneval \
  --samples outputs/tau_rerun/qwen_remask_humaneval_t0.9_fixed-sanitized.jsonl \
  --summary_out outputs/tau_rerun/qwen_remask_humaneval_t0.9_fixed_summary.json

# ── Qwen MBPP τ=0.9 (fixed) ──────────────────────────────────────────────────
echo "--- Qwen MBPP tau=0.9 fixed: $(date) ---"
python -m coder.scripts.gen_remask \
  --refiner dream \
  --input  outputs/base_tuteng/qwen_mbpp.jsonl \
  --out    outputs/tau_rerun/qwen_remask_mbpp_t0.9_fixed.jsonl \
  --confidence_threshold 0.9 \
  --seed 3407

python -m coder.scripts.postprocess_evalplus \
  --dataset mbpp \
  --samples outputs/tau_rerun/qwen_remask_mbpp_t0.9_fixed.jsonl

python -m coder.scripts.eval_evalplus \
  --backend local \
  --dataset mbpp \
  --samples outputs/tau_rerun/qwen_remask_mbpp_t0.9_fixed-sanitized.jsonl \
  --summary_out outputs/tau_rerun/qwen_remask_mbpp_t0.9_fixed_summary.json

echo ""
echo "=== GPU1 fixed DONE: $(date) ==="

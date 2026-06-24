#!/bin/bash
# GPU 3: multi-round CoCoder refinement -- DeepSeek HumanEval + MBPP r2/r3

set -e
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cocoder

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=3
cd "${ROOT_DIR}"

LOG="outputs/tau_rerun/gpu3_multirnd.log"
exec > >(tee -a "${LOG}") 2>&1
echo "=== GPU3 multi-round start: $(date) ==="

# ── DeepSeek HumanEval r2 (input = r1 output) ────────────────────────────────
echo "--- DeepSeek HumanEval r2: $(date) ---"
python -m coder.scripts.gen_remask \
  --refiner dream \
  --input  outputs/tau_rerun/remask_humaneval_t0.9.jsonl \
  --out    outputs/tau_rerun/remask_humaneval_t0.9_r2.jsonl \
  --confidence_threshold 0.9 \
  --seed 3407 --resume

python -m coder.scripts.postprocess_evalplus \
  --dataset humaneval \
  --samples outputs/tau_rerun/remask_humaneval_t0.9_r2.jsonl

python -m coder.scripts.eval_evalplus \
  --backend local \
  --dataset humaneval \
  --samples outputs/tau_rerun/remask_humaneval_t0.9_r2-sanitized.jsonl \
  --summary_out outputs/tau_rerun/remask_humaneval_t0.9_r2_summary.json

# ── DeepSeek HumanEval r3 (input = r2 output) ────────────────────────────────
echo "--- DeepSeek HumanEval r3: $(date) ---"
python -m coder.scripts.gen_remask \
  --refiner dream \
  --input  outputs/tau_rerun/remask_humaneval_t0.9_r2.jsonl \
  --out    outputs/tau_rerun/remask_humaneval_t0.9_r3.jsonl \
  --confidence_threshold 0.9 \
  --seed 3407 --resume

python -m coder.scripts.postprocess_evalplus \
  --dataset humaneval \
  --samples outputs/tau_rerun/remask_humaneval_t0.9_r3.jsonl

python -m coder.scripts.eval_evalplus \
  --backend local \
  --dataset humaneval \
  --samples outputs/tau_rerun/remask_humaneval_t0.9_r3-sanitized.jsonl \
  --summary_out outputs/tau_rerun/remask_humaneval_t0.9_r3_summary.json

# ── DeepSeek MBPP r2 ──────────────────────────────────────────────────────────
echo "--- DeepSeek MBPP r2: $(date) ---"
python -m coder.scripts.gen_remask \
  --refiner dream \
  --input  outputs/tau_rerun/remask_mbpp_t0.9.jsonl \
  --out    outputs/tau_rerun/remask_mbpp_t0.9_r2.jsonl \
  --confidence_threshold 0.9 \
  --seed 3407 --resume

python -m coder.scripts.postprocess_evalplus \
  --dataset mbpp \
  --samples outputs/tau_rerun/remask_mbpp_t0.9_r2.jsonl

python -m coder.scripts.eval_evalplus \
  --backend local \
  --dataset mbpp \
  --samples outputs/tau_rerun/remask_mbpp_t0.9_r2-sanitized.jsonl \
  --summary_out outputs/tau_rerun/remask_mbpp_t0.9_r2_summary.json

# ── DeepSeek MBPP r3 ──────────────────────────────────────────────────────────
echo "--- DeepSeek MBPP r3: $(date) ---"
python -m coder.scripts.gen_remask \
  --refiner dream \
  --input  outputs/tau_rerun/remask_mbpp_t0.9_r2.jsonl \
  --out    outputs/tau_rerun/remask_mbpp_t0.9_r3.jsonl \
  --confidence_threshold 0.9 \
  --seed 3407 --resume

python -m coder.scripts.postprocess_evalplus \
  --dataset mbpp \
  --samples outputs/tau_rerun/remask_mbpp_t0.9_r3.jsonl

python -m coder.scripts.eval_evalplus \
  --backend local \
  --dataset mbpp \
  --samples outputs/tau_rerun/remask_mbpp_t0.9_r3-sanitized.jsonl \
  --summary_out outputs/tau_rerun/remask_mbpp_t0.9_r3_summary.json

echo ""
echo "=== GPU3 multi-round DONE: $(date) ==="

#!/bin/bash
# GPU 4: protect_imports ablation -- Llama-3.1 HumanEval/MBPP + DeepSeek HumanEval

set -e
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cocoder

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=4
cd "${ROOT_DIR}"

LOG="outputs/tau_rerun/gpu4_protect_imports.log"
exec > >(tee -a "${LOG}") 2>&1
echo "=== GPU4 protect_imports start: $(date) ==="

# ── Llama-3.1 HumanEval τ=0.9 with protect_imports ──────────────────────────
echo "--- Llama-3.1 HumanEval tau=0.9 protect_imports: $(date) ---"
python -m coder.scripts.gen_remask \
  --refiner dream \
  --input  outputs/base_tuteng/llama31_humaneval.jsonl \
  --out    outputs/tau_rerun/llama31_remask_humaneval_t0.9_pi.jsonl \
  --confidence_threshold 0.9 \
  --protect_imports \
  --seed 3407 --resume

python -m coder.scripts.postprocess_evalplus \
  --dataset humaneval \
  --samples outputs/tau_rerun/llama31_remask_humaneval_t0.9_pi.jsonl

python -m coder.scripts.eval_evalplus \
  --backend local \
  --dataset humaneval \
  --samples outputs/tau_rerun/llama31_remask_humaneval_t0.9_pi-sanitized.jsonl \
  --summary_out outputs/tau_rerun/llama31_remask_humaneval_t0.9_pi_summary.json

# ── Llama-3.1 MBPP τ=0.9 with protect_imports ───────────────────────────────
echo "--- Llama-3.1 MBPP tau=0.9 protect_imports: $(date) ---"
python -m coder.scripts.gen_remask \
  --refiner dream \
  --input  outputs/base_tuteng/llama31_mbpp.jsonl \
  --out    outputs/tau_rerun/llama31_remask_mbpp_t0.9_pi.jsonl \
  --confidence_threshold 0.9 \
  --protect_imports \
  --seed 3407 --resume

python -m coder.scripts.postprocess_evalplus \
  --dataset mbpp \
  --samples outputs/tau_rerun/llama31_remask_mbpp_t0.9_pi.jsonl

python -m coder.scripts.eval_evalplus \
  --backend local \
  --dataset mbpp \
  --samples outputs/tau_rerun/llama31_remask_mbpp_t0.9_pi-sanitized.jsonl \
  --summary_out outputs/tau_rerun/llama31_remask_mbpp_t0.9_pi_summary.json

# ── DeepSeek HumanEval τ=0.9 with protect_imports (sanity check) ─────────────
echo "--- DeepSeek HumanEval tau=0.9 protect_imports: $(date) ---"
python -m coder.scripts.gen_remask \
  --refiner dream \
  --input  outputs/base_tuteng/deepseek_humaneval.jsonl \
  --out    outputs/tau_rerun/remask_humaneval_t0.9_pi.jsonl \
  --confidence_threshold 0.9 \
  --protect_imports \
  --seed 3407 --resume

python -m coder.scripts.postprocess_evalplus \
  --dataset humaneval \
  --samples outputs/tau_rerun/remask_humaneval_t0.9_pi.jsonl

python -m coder.scripts.eval_evalplus \
  --backend local \
  --dataset humaneval \
  --samples outputs/tau_rerun/remask_humaneval_t0.9_pi-sanitized.jsonl \
  --summary_out outputs/tau_rerun/remask_humaneval_t0.9_pi_summary.json

echo ""
echo "=== GPU4 protect_imports DONE: $(date) ==="

#!/bin/bash
# GPU 4: MBPP tau = 0.5, 0.7, 0.8, 0.9 + AR baseline re-eval

set -e
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cocoder

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=4
cd "${ROOT_DIR}"

LOG="outputs/tau_rerun/gpu4_mbpp_low.log"
exec > >(tee -a "${LOG}") 2>&1
echo "=== GPU4 MBPP-low start: $(date) ==="

# ── Step 0: re-evaluate AR baseline (CPU-only) ──────────────────────────────
echo "--- Re-evaluating AR baseline (MBPP) ---"
python -m coder.scripts.eval_evalplus \
  --backend local \
  --dataset mbpp \
  --samples outputs/base_tuteng/deepseek_mbpp-sanitized.jsonl \
  --summary_out outputs/tau_rerun/ar_mbpp_summary.json

# ── Step 1: remask + sanitize + eval ────────────────────────────────────────
for THRESH in 0.5 0.7 0.8 0.9; do
  echo ""
  echo "--- tau=${THRESH} MBPP: $(date) ---"

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

  echo "    done tau=${THRESH}: $(date)"
done

echo ""
echo "=== GPU4 MBPP-low DONE: $(date) ==="

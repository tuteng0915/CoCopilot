#!/bin/bash
# GPU 1: ALL HumanEval tau values + AR baseline re-eval

set -e
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cocoder

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=1
cd "${ROOT_DIR}"

LOG="outputs/tau_rerun/gpu1_humaneval.log"
exec > >(tee -a "${LOG}") 2>&1
echo "=== GPU1 HumanEval start: $(date) ==="

# ── Step 0: re-evaluate AR baseline (CPU-only) ──────────────────────────────
echo "--- Re-evaluating AR baseline (HumanEval) ---"
python -m coder.scripts.eval_evalplus \
  --backend local \
  --dataset humaneval \
  --samples outputs/base_tuteng/deepseek_humaneval-sanitized.jsonl \
  --summary_out outputs/tau_rerun/ar_humaneval_summary.json

# ── Step 1: remask + sanitize + eval for each tau ───────────────────────────
for THRESH in 0.5 0.7 0.8 0.9 0.93 0.95 0.97 0.99; do
  echo ""
  echo "--- tau=${THRESH} HumanEval: $(date) ---"

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

  echo "    done tau=${THRESH}: $(date)"
done

echo ""
echo "=== GPU1 HumanEval DONE: $(date) ==="

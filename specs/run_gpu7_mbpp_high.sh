#!/bin/bash
# GPU 7: MBPP tau = 0.93, 0.95, 0.97, 0.99

set -e
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cocoder

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=7
cd "${ROOT_DIR}"

LOG="outputs/tau_rerun/gpu7_mbpp_high.log"
exec > >(tee -a "${LOG}") 2>&1
echo "=== GPU7 MBPP-high start: $(date) ==="

for THRESH in 0.93 0.95 0.97 0.99; do
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
echo "=== GPU7 MBPP-high DONE: $(date) ==="

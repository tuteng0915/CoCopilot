ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}

for DATASET in humaneval mbpp; do
  echo "=== DeepSeek generation: ${DATASET} ==="
  python -m coder.scripts.gen_evalplus \
    --model deepseek \
    --dataset "${DATASET}" \
    --out "outputs/deepseek_${DATASET}.jsonl" \
    --temperature 0.0 --top_p 1.0 --seed 3407
done

echo "=== DeepSeek generation done ==="

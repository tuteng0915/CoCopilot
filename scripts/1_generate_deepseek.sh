set -euo pipefail

export PYTHONPATH=/home/kodai/CoCopilot/src
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}

for DATASET in humaneval mbpp; do
  echo "=== DeepSeek generation: ${DATASET} ==="
  python scripts/gen_evalplus.py \
    --model deepseek \
    --dataset "${DATASET}" \
    --out "outputs/deepseek_${DATASET}.jsonl" \
    --temperature 0.0 --top_p 1.0 --seed 3407
done

echo "=== DeepSeek generation done ==="

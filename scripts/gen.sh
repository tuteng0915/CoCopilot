ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=5

python -m coder.scripts.gen_evalplus \
  --model dream \
  --dataset humaneval \
  --out outputs/dream_humaneval.jsonl

python -m coder.scripts.postprocess_evalplus \
  --dataset humaneval \
  --samples outputs/dream_humaneval.jsonl

exit

python -m coder.scripts.gen_evalplus \
  --model dream \
  --dataset humaneval \
  --out outputs/dream_humaneval.jsonl

python -m coder.scripts.gen_evalplus \
  --model deepseek \
  --dataset humaneval \
  --out outputs/deepseek_humaneval.jsonl

python -m coder.scripts.gen_evalplus \
  --model dream \
  --dataset mbpp \
  --out outputs/dream_mbpp.jsonl

python -m coder.scripts.gen_evalplus \
  --model deepseek \
  --dataset mbpp \
  --out outputs/deepseek_mbpp.jsonl




python -m coder.scripts.postprocess_evalplus \
  --dataset humaneval \
  --samples outputs/dream_humaneval.jsonl

python -m coder.scripts.postprocess_evalplus \
  --dataset humaneval \
  --samples outputs/deepseek_humaneval.jsonl

python -m coder.scripts.postprocess_evalplus \
  --dataset mbpp \
  --samples outputs/dream_mbpp.jsonl

python -m coder.scripts.postprocess_evalplus \
  --dataset mbpp \
  --samples outputs/deepseek_mbpp.jsonl

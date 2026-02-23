export PYTHONPATH=src

python scripts/gen_evalplus.py \
  --model dream \
  --dataset humaneval \
  --out outputs/dream_humaneval.jsonl

python scripts/gen_evalplus.py \
  --model deepseek \
  --dataset humaneval \
  --out outputs/deepseek_humaneval.jsonl

python scripts/gen_evalplus.py \
  --model dream \
  --dataset mbpp \
  --out outputs/dream_mbpp.jsonl

python scripts/gen_evalplus.py \
  --model deepseek \
  --dataset mbpp \
  --out outputs/deepseek_mbpp.jsonl




python scripts/postprocess_evalplus.py \
  --dataset humaneval \
  --samples outputs/dream_humaneval.jsonl

python scripts/postprocess_evalplus.py \
  --dataset humaneval \
  --samples outputs/deepseek_humaneval.jsonl

python scripts/postprocess_evalplus.py \
  --dataset mbpp \
  --samples outputs/dream_mbpp.jsonl

python scripts/postprocess_evalplus.py \
  --dataset mbpp \
  --samples outputs/deepseek_mbpp.jsonl

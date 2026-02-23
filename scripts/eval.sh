python scripts/eval_evalplus.py \
  --backend local \
  --dataset humaneval \
  --samples outputs/dream_humaneval-sanitized.jsonl

python scripts/eval_evalplus.py \
  --backend local \
  --dataset humaneval \
  --samples outputs/deepseek_humaneval-sanitized.jsonl

python scripts/eval_evalplus.py \
  --backend local \
  --dataset mbpp \
  --samples outputs/dream_mbpp-sanitized.jsonl

python scripts/eval_evalplus.py \
  --backend local \
  --dataset mbpp \
  --samples outputs/deepseek_mbpp-sanitized.jsonl

python scripts/eval_livebench.py \
  --samples outputs/dream_livebench.jsonl \
  --out_judgments outputs/dream_livebench_judgments.jsonl \
  --out_summary outputs/dream_livebench_summary.json

python scripts/eval_livebench.py \
  --samples outputs/deepseek_livebench.jsonl \
  --out_judgments outputs/deepseek_livebench_judgments.jsonl \
  --out_summary outputs/deepseek_livebench_summary.json

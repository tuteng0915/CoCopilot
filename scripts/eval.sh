python -m coder.scripts.eval_evalplus \
  --backend local \
  --dataset humaneval \
  --samples outputs/dream_humaneval-sanitized.jsonl

python -m coder.scripts.eval_evalplus \
  --backend local \
  --dataset humaneval \
  --samples outputs/deepseek_humaneval-sanitized.jsonl

python -m coder.scripts.eval_evalplus \
  --backend local \
  --dataset mbpp \
  --samples outputs/dream_mbpp-sanitized.jsonl

python -m coder.scripts.eval_evalplus \
  --backend local \
  --dataset mbpp \
  --samples outputs/deepseek_mbpp-sanitized.jsonl

python -m coder.scripts.eval_livebench \
  --samples outputs/dream_livebench.jsonl \
  --out_judgments outputs/dream_livebench_judgments.jsonl \
  --out_summary outputs/dream_livebench_summary.json

python -m coder.scripts.eval_livebench \
  --samples outputs/deepseek_livebench.jsonl \
  --out_judgments outputs/deepseek_livebench_judgments.jsonl \
  --out_summary outputs/deepseek_livebench_summary.json

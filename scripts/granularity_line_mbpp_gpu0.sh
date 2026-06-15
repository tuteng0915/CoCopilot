#!/usr/bin/env bash
set -euo pipefail

cd /home/wjzhang/tt_workspace/model/CoCoder/CoCoder
export PYTHONPATH=src

OUT=outputs/ablation_granularity
RAW="$OUT/deepseek_dream_mbpp_t0.9_gran_line.jsonl"
SAN="$OUT/deepseek_dream_mbpp_t0.9_gran_line-sanitized.jsonl"
SUMMARY="$OUT/deepseek_dream_mbpp_t0.9_gran_line_summary.json"
EVAL="$OUT/deepseek_dream_mbpp_t0.9_gran_line-sanitized_eval_results.json"

mkdir -p "$OUT"

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*"
}

log "generate line granularity MBPP on GPU0"
CUDA_VISIBLE_DEVICES=0 conda run -n elf python -m coder.scripts.gen_remask \
  --input outputs/base_tuteng/deepseek_mbpp.jsonl \
  --out "$RAW" \
  --locator dream \
  --confidence_threshold 0.9 \
  --mask_granularity line \
  --temperature 0.1 --top_p 0.95 --seed 3407 \
  --device cuda:0 \
  --resume

log "verify record count"
conda run -n elf python - <<'PY'
from pathlib import Path

path = Path("outputs/ablation_granularity/deepseek_dream_mbpp_t0.9_gran_line.jsonl")
records = sum(1 for line in path.open() if line.strip())
print(f"records: {records} (expected 378)")
if records != 378:
    raise SystemExit(1)
PY

log "postprocess"
conda run -n elf python -m coder.scripts.postprocess_evalplus \
  --dataset mbpp \
  --samples "$RAW"

log "eval"
conda run -n elf python -m coder.scripts.eval_evalplus \
  --dataset mbpp \
  --samples "$SAN" \
  --backend local \
  --parallel 16 \
  --output_file "$EVAL" \
  --summary_out "$SUMMARY" \
  --summary_model deepseek_dream_mbpp_t0.9_gran_line

log "granularity line MBPP done"

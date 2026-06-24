#!/usr/bin/env bash
# GPU 6 — Granularity ablation expansion.
# Adds llama31 and qwen to the existing deepseek-only granularity results.
# Granularities: line (mask full lines) and span (merge adjacent masked tokens).
# tmux: tmux new -s gpu6_gran && bash scripts/run_gpu6_granularity.sh 2>&1 | tee /tmp/gpu6_granularity.log
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH=src
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export CUDA_VISIBLE_DEVICES=6

PY="/home/wjzhang/miniforge3/envs/cocoder/bin/python"
export PATH="/home/wjzhang/miniforge3/envs/cocoder/bin:$PATH"

BASE="outputs/base_tuteng"
OUT="outputs/ablation_granularity"
DREAM="Dream-org/Dream-Coder-v0-Instruct-7B"

log() { printf '[%(%F %T)T] %s\n' -1 "$*"; }

sanitize_and_eval() {
  local dataset="$1"
  local samples="$2"
  local base="${samples%.jsonl}"
  local sanitized="${base}-sanitized.jsonl"
  local eval_out="${base}-sanitized_eval_results.json"

  if [[ ! -f "$sanitized" ]]; then
    log "sanitize $samples"
    "$PY" -m coder.scripts.postprocess_evalplus \
      --dataset "$dataset" --samples "$samples" --skip_syncheck
  else
    log "skip sanitize (exists): $sanitized"
  fi

  if [[ ! -f "$eval_out" ]]; then
    log "eval $sanitized"
    "$PY" -m coder.scripts.eval_evalplus \
      --backend local --dataset "$dataset" --samples "$sanitized"
  else
    log "skip eval (exists): $eval_out"
  fi
}

gran_job() {
  local ar="$1"
  local ds="$2"
  local gran="$3"   # line or span
  local input="$BASE/${ar}_${ds}.jsonl"
  local out="$OUT/${ar}_dream_${ds}_t0.9_gran_${gran}.jsonl"

  log "=== gran[$gran] [dream] $ar × $ds ==="
  "$PY" -m coder.scripts.gen_remask \
    --refiner dream \
    --model_id "$DREAM" \
    --input "$input" \
    --out "$out" \
    --device cuda:0 \
    --confidence_threshold 0.9 \
    --mask_granularity "$gran" \
    --resume

  sanitize_and_eval "$ds" "$out"
}

# ── 8 jobs: {llama31, qwen} × {humaneval, mbpp} × {line, span} ───────────────
log "===== Granularity expansion (GPU 6) ====="

for AR in llama31 qwen; do
  for DS in humaneval mbpp; do
    for GRAN in line span; do
      gran_job "$AR" "$DS" "$GRAN"
    done
  done
done

log "===== ALL DONE ====="

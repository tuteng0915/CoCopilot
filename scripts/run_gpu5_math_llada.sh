#!/usr/bin/env bash
# GPU 5 — Math-to-code LLaDA refiner expansion.
# Adds LLaDA as dLLM for the 3 existing AR × {gsm8k, math500} combinations.
# tmux: tmux new -s gpu5_math && bash scripts/run_gpu5_math_llada.sh 2>&1 | tee /tmp/gpu5_math_llada.log
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH=src
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export CUDA_VISIBLE_DEVICES=5

PY="/home/wjzhang/miniforge3/envs/cocoder/bin/python"
export PATH="/home/wjzhang/miniforge3/envs/cocoder/bin:$PATH"

MATH="outputs/math_code"
LLADA="GSAI-ML/LLaDA-8B-Instruct"

log() { printf '[%(%F %T)T] %s\n' -1 "$*"; }

math_remask_job() {
  local ar="$1"
  local ds="$2"
  local input="$MATH/${ar}_${ds}_code.jsonl"
  local out="$MATH/${ar}_${ds}_code_llada_t0.9_plnt3.jsonl"
  local eval_out="${out%.jsonl}_eval.json"

  log "=== remask [llada] $ar × $ds ==="
  "$PY" -m coder.scripts.gen_remask \
    --refiner llada \
    --model_id "$LLADA" \
    --input "$input" \
    --out "$out" \
    --device cuda:0 \
    --confidence_threshold 0.9 \
    --protect_last_n_tokens 3 \
    --resume

  if [[ ! -f "$eval_out" ]]; then
    log "eval $ar $ds"
    "$PY" -m coder.scripts.eval_math_code \
      --input "$out" \
      --out "$eval_out" \
      --completion_field raw_completion
  else
    log "skip eval (exists): $eval_out"
  fi
}

# ── 6 jobs: 3 AR × {gsm8k, math500} ──────────────────────────────────────────
log "===== Math-to-code LLaDA refiner (GPU 5) ====="

for AR in deepseek llama31 qwen; do
  for DS in gsm8k math500; do
    math_remask_job "$AR" "$DS"
  done
done

log "===== ALL DONE ====="

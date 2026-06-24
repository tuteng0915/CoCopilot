#!/usr/bin/env bash
# GPU 6 — Continuation of GPU7 tasks + AR rewrite ablation expansion.
#
# Phases:
#   A: AR rewrite ablation: {llama31, qwen} × {humaneval, mbpp} × span  (4 jobs, ~1h)
#   B: DiffuCoder standalone MBPP                                        (1 job,  ~7h)
#   C: DiffuCoder-7B-Instruct refiner, 7 AR × {humaneval, mbpp}         (14 jobs, ~40h est.)
#   D: Stable-DiffCoder-8B refiner,    7 AR × {humaneval, mbpp}         (14 jobs, ~40h est.)
#
# tmux: tmux new -s gpu6_remaining && bash scripts/run_gpu6_remaining.sh 2>&1 | tee /tmp/gpu6_remaining.log
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH=src
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export CUDA_VISIBLE_DEVICES=6

PY="/home/wjzhang/miniforge3/envs/cocoder/bin"
export PATH="$PY:$PATH"
PY="$PY/python"
OUT="outputs/base_tuteng"
GRAN="outputs/ablation_granularity"

log() { printf '[%(%F %T)T] %s\n' -1 "$*"; }

# ── sanitize + eval ────────────────────────────────────────────────────────────
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

# ── gen_remask wrapper ─────────────────────────────────────────────────────────
remask_job() {
  local refiner="$1"
  local model_id="$2"
  local input="$3"
  local out="$4"
  local dataset="$5"
  shift 5

  log "=== remask [$refiner] $(basename "$input") → $(basename "$out") ==="
  "$PY" -m coder.scripts.gen_remask \
    --refiner "$refiner" \
    --model_id "$model_id" \
    --input "$input" \
    --out "$out" \
    --device cuda:0 \
    --resume \
    "$@"

  sanitize_and_eval "$dataset" "$out"
}

# ── AR rewrite wrapper ─────────────────────────────────────────────────────────
ar_rewrite_job() {
  local ar="$1"
  local ds="$2"
  local gran="${3:-span}"
  local input="$OUT/${ar}_${ds}.jsonl"
  local out="$GRAN/${ar}_ar_rewrite_${ds}_t0.9_gran_${gran}.jsonl"
  local base="${out%.jsonl}"

  log "=== ar_rewrite [$ar] $ds gran=$gran ==="
  "$PY" -m coder.scripts.gen_locate_ar_rewrite \
    --ar_model "$ar" \
    --input "$input" \
    --out "$out" \
    --confidence_threshold 0.9 \
    --mask_granularity "$gran" \
    --temperature 0.0 \
    --locator_device cuda:0 \
    --ar_device cuda:0 \
    --resume

  sanitize_and_eval "$ds" "$out"
}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE A — AR rewrite ablation: {llama31, qwen} × {humaneval, mbpp} × span
# (4 jobs; fast ~10-25 min each)
# ══════════════════════════════════════════════════════════════════════════════
log "========== PHASE A: AR rewrite ablation (4 jobs) =========="

for AR in llama31 qwen; do
  for DS in humaneval mbpp; do
    ar_rewrite_job "$AR" "$DS" span
  done
done

# ══════════════════════════════════════════════════════════════════════════════
# PHASE B — DiffuCoder standalone MBPP  (1 job; ~7h)
# ══════════════════════════════════════════════════════════════════════════════
log "========== PHASE B: DiffuCoder standalone MBPP (1 job) =========="

DIFFU_ID="apple/DiffuCoder-7B-Instruct"

for DS in mbpp; do
  out="$OUT/diffucoder_${DS}.jsonl"
  if [[ ! -f "$out" ]] || [[ "$(wc -l < "$out" 2>/dev/null || echo 0)" -lt 100 ]]; then
    log "gen diffucoder $DS"
    "$PY" -m coder.scripts.gen_evalplus \
      --model diffucoder \
      --model_id "$DIFFU_ID" \
      --dataset "$DS" \
      --out "$out" \
      --device cuda:0 \
      --seed 3407
  else
    log "skip gen (exists): $out"
  fi
  sanitize_and_eval "$DS" "$out"
done

# ══════════════════════════════════════════════════════════════════════════════
# PHASE C — DiffuCoder-7B-Instruct refiner  (14 jobs)
# ══════════════════════════════════════════════════════════════════════════════
log "========== PHASE C: DiffuCoder refiner (14 jobs) =========="

for AR in deepseek qwen llama31 codellama mistral starcoder2 seed-coder-instruct; do
  for DS in humaneval mbpp; do
    remask_job diffucoder "$DIFFU_ID" \
      "$OUT/${AR}_${DS}.jsonl" \
      "$OUT/${AR}_diffucoder_remask_${DS}_t0.9.jsonl" \
      "$DS" \
      --confidence_threshold 0.9
  done
done

# ══════════════════════════════════════════════════════════════════════════════
# PHASE D — Stable-DiffCoder-8B refiner  (14 jobs)
# ══════════════════════════════════════════════════════════════════════════════
log "========== PHASE D: Stable-DiffCoder refiner (14 jobs) =========="

SDIFF_ID="ByteDance-Seed/Stable-DiffCoder-8B-Instruct"

for AR in deepseek qwen llama31 codellama mistral starcoder2 seed-coder-instruct; do
  for DS in humaneval mbpp; do
    remask_job seed-diffcoder "$SDIFF_ID" \
      "$OUT/${AR}_${DS}.jsonl" \
      "$OUT/${AR}_seeddiff_remask_${DS}_t0.9.jsonl" \
      "$DS" \
      --confidence_threshold 0.9
  done
done

log "========== ALL PHASES COMPLETE =========="

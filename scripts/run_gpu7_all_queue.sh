#!/usr/bin/env bash
# Master queue for GPU 7: LLaDA plnt3 + DiffuCoder standalone/refiner + Stable-DiffCoder refiner.
# Run in a tmux session so it survives disconnects:
#   tmux new -s gpu7_queue
#   bash scripts/run_gpu7_all_queue.sh 2>&1 | tee /tmp/gpu7_all_queue.log
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH=src
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export CUDA_VISIBLE_DEVICES=7

PY="/home/wjzhang/miniforge3/envs/cocoder/bin/python"
CONDA_ENV_BIN="/home/wjzhang/miniforge3/envs/cocoder/bin"
export PATH="$CONDA_ENV_BIN:$PATH"
OUT="outputs/base_tuteng"
PKG="outputs/base_tuteng/packaging_v2"

log() { printf '[%(%F %T)T] %s\n' -1 "$*"; }

# ── sanitize + eval (--skip_syncheck avoids PATH issue with evalplus.syncheck) ─

sanitize_and_eval() {
  local dataset="$1"
  local samples="$2"
  local base="${samples%.jsonl}"
  local sanitized="${base}-sanitized.jsonl"

  if [[ ! -f "$sanitized" ]]; then
    log "sanitize $samples"
    "$PY" -m coder.scripts.postprocess_evalplus \
      --dataset "$dataset" --samples "$samples" --skip_syncheck
  else
    log "skip sanitize (exists): $sanitized"
  fi

  local eval_out="${base}-sanitized_eval_results.json"
  if [[ ! -f "$eval_out" ]]; then
    log "eval $sanitized"
    "$PY" -m coder.scripts.eval_evalplus \
      --backend local --dataset "$dataset" --samples "$sanitized"
  else
    log "skip eval (exists): $eval_out"
  fi
}

# ── gen_remask wrapper ──────────────────────────────────────────────────────────

remask_job() {
  local refiner="$1"
  local model_id="$2"
  local input="$3"
  local out="$4"
  local dataset="$5"
  shift 5
  # remaining args: extra flags like --confidence_threshold --mask_ratio etc.

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

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — LLaDA plnt3 Group A  (14 AR × {humaneval, mbpp})
# ══════════════════════════════════════════════════════════════════════════════

log "========== PHASE 1: LLaDA plnt3 Group A (14 jobs) =========="

LLADA_ID="GSAI-ML/LLaDA-8B-Instruct"

for AR in deepseek qwen llama31 codellama mistral starcoder2 seed-coder-instruct; do
  for DS in humaneval mbpp; do
    remask_job llada "$LLADA_ID" \
      "$OUT/${AR}_${DS}.jsonl" \
      "$OUT/${AR}_llada_remask_${DS}_t0.9_plnt3.jsonl" \
      "$DS" \
      --confidence_threshold 0.9 --protect_last_n_tokens 3
  done
done

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — LLaDA plnt3 Group B/C  (seed-coder pkgv2 + base maskr0.01 variants)
# ══════════════════════════════════════════════════════════════════════════════

log "========== PHASE 2: LLaDA plnt3 Group B/C (6 jobs) =========="

for DS in humaneval mbpp; do
  # B1: pkgv2 t0.9
  remask_job llada "$LLADA_ID" \
    "$PKG/seed-coder-instruct_${DS}_pkgv2.jsonl" \
    "$PKG/seed-coder-instruct_llada_remask_${DS}_t0.9_plnt3_pkgv2.jsonl" \
    "$DS" \
    --confidence_threshold 0.9 --protect_last_n_tokens 3

  # B2: pkgv2 maskr0.01 gate0.012
  remask_job llada "$LLADA_ID" \
    "$PKG/seed-coder-instruct_${DS}_pkgv2.jsonl" \
    "$PKG/seed-coder-instruct_llada_remask_${DS}_maskr0.01_gate0.012_plnt3_pkgv2.jsonl" \
    "$DS" \
    --mask_ratio 0.01 --gate_min_mask_fraction 0.012 --protect_last_n_tokens 3

  # C: base_tuteng maskr0.01 gate0.012
  remask_job llada "$LLADA_ID" \
    "$OUT/seed-coder-instruct_${DS}.jsonl" \
    "$OUT/seed-coder-instruct_llada_remask_${DS}_maskr0.01_gate0.012_fixed_plnt3.jsonl" \
    "$DS" \
    --mask_ratio 0.01 --gate_min_mask_fraction 0.012 --protect_last_n_tokens 3
done

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — DiffuCoder-7B-Instruct standalone  (2 jobs)
# ══════════════════════════════════════════════════════════════════════════════

log "========== PHASE 3: DiffuCoder standalone (2 jobs) =========="

DIFFU_ID="apple/DiffuCoder-7B-Instruct"

for DS in humaneval mbpp; do
  out="$OUT/diffucoder_${DS}.jsonl"
  if [[ ! -f "$out" ]] || [[ "$(wc -l < "$out" 2>/dev/null || echo 0)" -lt 100 ]]; then
    log "gen diffucoder $DS"
    "$PY" -m coder.scripts.gen_evalplus \
      --model diffucoder \
      --model_id "$DIFFU_ID" \
      --dataset "$DS" \
      --out "$out" \
      --seed 3407
  else
    log "skip gen (exists): $out"
  fi
  sanitize_and_eval "$DS" "$out"
done

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — DiffuCoder-7B-Instruct refiner  (14 AR × dataset)
# ══════════════════════════════════════════════════════════════════════════════

log "========== PHASE 4: DiffuCoder refiner (14 jobs) =========="

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
# PHASE 5 — Stable-DiffCoder-8B refiner  (14 AR × dataset)
# ══════════════════════════════════════════════════════════════════════════════

log "========== PHASE 5: Stable-DiffCoder refiner (14 jobs) =========="

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

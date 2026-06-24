#!/usr/bin/env bash
# Re-run all LLaDA remask experiments with --protect_last_n_tokens 3.
# Fixes systematic trailing-truncation bug caused by LLaDA's low confidence
# on final closing delimiters at tau=0.9.
#
# Usage:
#   GPU=7 bash scripts/run_llada_plnt3_reruns.sh
#   GPU=7 GROUP=A bash scripts/run_llada_plnt3_reruns.sh   # only Group A
#   GPU=7 GROUP=B bash scripts/run_llada_plnt3_reruns.sh   # only Group B/C
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${GPU:-7}"
GROUP="${GROUP:-ALL}"   # A | B | ALL

export PYTHONPATH=src
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

PY="/home/wjzhang/miniforge3/envs/cocoder/bin/python"
OUT="outputs/base_tuteng"
PKG="outputs/base_tuteng/packaging_v2"

log() { printf '[%(%F %T)T] %s\n' -1 "$*"; }

# ─── helpers ───────────────────────────────────────────────────────────────────

sanitize_and_eval() {
  local dataset="$1"
  local samples="$2"
  local base="${samples%.jsonl}"

  log "sanitize $samples"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m coder.scripts.postprocess_evalplus \
    --dataset "$dataset" --samples "$samples"

  local sanitized="${base}-sanitized.jsonl"
  log "eval $sanitized"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m coder.scripts.eval_evalplus \
    --backend local --dataset "$dataset" --samples "$sanitized"
}

run_group_a_job() {
  local ar="$1"
  local dataset="$2"
  local input="${OUT}/${ar}_${dataset}.jsonl"
  local out="${OUT}/${ar}_llada_remask_${dataset}_t0.9_plnt3.jsonl"

  log "=== Group A: $ar + $dataset ==="
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m coder.scripts.gen_remask \
    --refiner llada \
    --input "$input" \
    --out "$out" \
    --confidence_threshold 0.9 \
    --protect_last_n_tokens 3 \
    --device "cuda:0" \
    --resume

  sanitize_and_eval "$dataset" "$out"
}

# ─── Group A: 14 AR × {humaneval,mbpp} combos ──────────────────────────────────

run_group_a() {
  log "====== Starting Group A (14 jobs) ======"

  for ar in deepseek qwen llama31 codellama mistral starcoder2 seed-coder-instruct; do
    for dataset in humaneval mbpp; do
      run_group_a_job "$ar" "$dataset"
    done
  done

  log "====== Group A done ======"
}

# ─── Group B: packaging_v2 seed-coder-instruct LLaDA ───────────────────────────

run_group_b() {
  log "====== Starting Group B (packaging_v2, 4 jobs) ======"

  for dataset in humaneval mbpp; do
    local input="${PKG}/seed-coder-instruct_${dataset}_pkgv2.jsonl"
    local out="${PKG}/seed-coder-instruct_llada_remask_${dataset}_t0.9_plnt3_pkgv2.jsonl"

    log "=== Group B t0.9: seed-coder-instruct + $dataset (pkgv2) ==="
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m coder.scripts.gen_remask \
      --refiner llada \
      --input "$input" \
      --out "$out" \
      --confidence_threshold 0.9 \
      --protect_last_n_tokens 3 \
      --device "cuda:0" \
      --resume

    sanitize_and_eval "$dataset" "$out"
  done

  for dataset in humaneval mbpp; do
    local input="${PKG}/seed-coder-instruct_${dataset}_pkgv2.jsonl"
    local out="${PKG}/seed-coder-instruct_llada_remask_${dataset}_maskr0.01_gate0.012_plnt3_pkgv2.jsonl"

    log "=== Group B maskr0.01: seed-coder-instruct + $dataset (pkgv2) ==="
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m coder.scripts.gen_remask \
      --refiner llada \
      --input "$input" \
      --out "$out" \
      --mask_ratio 0.01 \
      --gate_min_mask_fraction 0.012 \
      --protect_last_n_tokens 3 \
      --device "cuda:0" \
      --resume

    sanitize_and_eval "$dataset" "$out"
  done

  log "====== Group B done ======"
}

# ─── Group C: base_tuteng seed-coder-instruct mask_ratio variants ──────────────

run_group_c() {
  log "====== Starting Group C (base_tuteng maskr0.01, 2 jobs) ======"

  for dataset in humaneval mbpp; do
    local input="${OUT}/seed-coder-instruct_${dataset}.jsonl"
    local out="${OUT}/seed-coder-instruct_llada_remask_${dataset}_maskr0.01_gate0.012_fixed_plnt3.jsonl"

    log "=== Group C maskr0.01: seed-coder-instruct + $dataset ==="
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m coder.scripts.gen_remask \
      --refiner llada \
      --input "$input" \
      --out "$out" \
      --mask_ratio 0.01 \
      --gate_min_mask_fraction 0.012 \
      --protect_last_n_tokens 3 \
      --device "cuda:0" \
      --resume

    sanitize_and_eval "$dataset" "$out"
  done

  log "====== Group C done ======"
}

# ─── Main ──────────────────────────────────────────────────────────────────────

log "Starting llada plnt3 reruns on GPU $GPU (GROUP=$GROUP)"

case "$GROUP" in
  A)   run_group_a ;;
  B)   run_group_b; run_group_c ;;
  ALL) run_group_a; run_group_b; run_group_c ;;
  *)   echo "Unknown GROUP=$GROUP. Use A, B, or ALL."; exit 1 ;;
esac

log "All reruns complete."

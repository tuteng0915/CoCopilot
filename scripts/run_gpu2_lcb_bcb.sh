#!/usr/bin/env bash
# GPU 2 — LCB + BCB gen_remask expansion.
#
# Dream and LLaDA as dLLM refiners for {deepseek, llama31, qwen}.
# LCB:  6 jobs  (outputs/base_tuteng/{AR}_{dream|llada}_livecodebench_t0.9*.jsonl)
# BCB:  6 jobs  (outputs/base_tuteng/{AR}_{dream|llada}_bigcodebench_instruct_full_t0.9*.jsonl)
#
# Eval dependencies:
#   LCB eval: pip install "git+https://github.com/LiveBench/LiveBench.git"
#   BCB eval: requires bigcodebench.evaluate (bcb_iso env on old machine)
#   The eval steps below fail gracefully if packages are missing.
#
# tmux: tmux new -s gpu2_lcb_bcb && bash scripts/run_gpu2_lcb_bcb.sh 2>&1 | tee /tmp/gpu2_lcb_bcb.log
set -uo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH=src
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export CUDA_VISIBLE_DEVICES=2

PY="/home/wjzhang/miniforge3/envs/cocoder/bin/python"
export PATH="/home/wjzhang/miniforge3/envs/cocoder/bin:$PATH"

BASE="outputs/base_tuteng"
DREAM="Dream-org/Dream-Coder-v0-Instruct-7B"
LLADA="GSAI-ML/LLaDA-8B-Instruct"

log() { printf '[%(%F %T)T] %s\n' -1 "$*"; }

# ── LCB eval (soft-fail if livebench not installed) ───────────────────────────
eval_lcb() {
  local samples="$1"
  local stem="${samples%.jsonl}"
  "$PY" -m coder.scripts.eval_livebench \
    --benchmark livecodebench \
    --samples "$samples" \
    --out_judgments "${stem}_judgments.jsonl" \
    --out_summary   "${stem}_summary.json" \
    || log "WARN: LCB eval failed (install livebench: pip install git+https://github.com/LiveBench/LiveBench.git)"
}

# ── BCB eval (soft-fail if bigcodebench.evaluate not in PATH) ────────────────
eval_bcb() {
  local samples="$1"
  local stem="${samples%.jsonl}"
  "$PY" -m coder.scripts.eval_bigcodebench \
    --samples "$samples" \
    --split instruct --subset full \
    --execution local \
    --out_summary "${stem}_summary.json" \
    || log "WARN: BCB eval failed (need bigcodebench.evaluate in PATH)"
}

# ── gen_remask wrapper ────────────────────────────────────────────────────────
remask_job() {
  local refiner="$1"; local model_id="$2"
  local input="$3"; local out="$4"
  shift 4   # remaining: extra flags

  log "=== remask [$refiner] $(basename "$input") → $(basename "$out") ==="
  "$PY" -m coder.scripts.gen_remask \
    --refiner "$refiner" --model_id "$model_id" \
    --input "$input" --out "$out" \
    --device cuda:0 --resume "$@"
}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — LiveCodeBench  (6 jobs: 3 AR × {dream, llada})
# ══════════════════════════════════════════════════════════════════════════════

log "========== PHASE 1: LCB gen_remask + eval =========="

for AR in deepseek llama31 qwen; do
  # Dream refiner
  OUT_D="$BASE/${AR}_dream_livecodebench_t0.9.jsonl"
  remask_job dream "$DREAM" \
    "$BASE/${AR}_livecodebench.jsonl" "$OUT_D" \
    --confidence_threshold 0.9
  eval_lcb "$OUT_D"

  # LLaDA refiner
  OUT_L="$BASE/${AR}_llada_livecodebench_t0.9_plnt3.jsonl"
  remask_job llada "$LLADA" \
    "$BASE/${AR}_livecodebench.jsonl" "$OUT_L" \
    --confidence_threshold 0.9 --protect_last_n_tokens 3
  eval_lcb "$OUT_L"
done

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — BigCodeBench full  (6 jobs: 3 AR × {dream, llada})
# Input: pass1_clean versions (correctly formatted AR solutions)
# ══════════════════════════════════════════════════════════════════════════════

log "========== PHASE 2: BCB gen_remask + eval =========="

for AR in deepseek llama31 qwen; do
  INPUT="$BASE/${AR}_bigcodebench_instruct_full_pass1_clean.jsonl"

  # Dream refiner
  OUT_D="$BASE/${AR}_dream_bigcodebench_instruct_full_t0.9.jsonl"
  remask_job dream "$DREAM" \
    "$INPUT" "$OUT_D" \
    --confidence_threshold 0.9
  eval_bcb "$OUT_D"

  # LLaDA refiner
  OUT_L="$BASE/${AR}_llada_bigcodebench_instruct_full_t0.9_plnt3.jsonl"
  remask_job llada "$LLADA" \
    "$INPUT" "$OUT_L" \
    --confidence_threshold 0.9 --protect_last_n_tokens 3
  eval_bcb "$OUT_L"
done

log "========== ALL PHASES COMPLETE =========="

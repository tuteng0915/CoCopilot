#!/usr/bin/env bash
# GPU 2 — DiffuCoder refiner (Phase C remainder) + Stable-DiffCoder refiner (Phase D).
# Continuation of GPU 6 tasks, queued after run_gpu2_lcb_bcb.sh.
#
# Launch AFTER gpu2_lcb_bcb, or in a new tmux pane:
#   tmux new-window -t gpu2_lcb_bcb "bash scripts/run_gpu2_diffucoder_seeddiff.sh 2>&1 | tee /tmp/gpu2_diffucoder.log"
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH=src
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export CUDA_VISIBLE_DEVICES=2

PY="/home/wjzhang/miniforge3/envs/cocoder/bin"
export PATH="$PY:$PATH"
PY="$PY/python"
OUT="outputs/base_tuteng"

log() { printf '[%(%F %T)T] %s\n' -1 "$*"; }

# ── Wait for LCB/BCB queue to finish ──────────────────────────────────────────
log "Waiting for run_gpu2_lcb_bcb.sh to finish..."
until grep -q "ALL PHASES COMPLETE" /tmp/gpu2_lcb_bcb.log 2>/dev/null; do
    sleep 120
done
log "LCB/BCB queue complete. Starting DiffuCoder/SeedDiff phases."

# ── sanitize + eval ────────────────────────────────────────────────────────────
sanitize_and_eval() {
    local dataset="$1" samples="$2"
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

remask_job() {
    local refiner="$1" model_id="$2" input="$3" out="$4" dataset="$5"
    shift 5
    log "=== remask [$refiner] $(basename "$input") → $(basename "$out") ==="
    "$PY" -m coder.scripts.gen_remask \
        --refiner "$refiner" --model_id "$model_id" \
        --input "$input" --out "$out" \
        --device cuda:0 --resume "$@"
    sanitize_and_eval "$dataset" "$out"
}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE C (remainder) — DiffuCoder refiner: seed-coder-instruct × mbpp
# (all other 13 jobs already done on GPU 6; this one was interrupted at 54/378)
# ══════════════════════════════════════════════════════════════════════════════
log "========== PHASE C remainder: DiffuCoder seed-coder-instruct × mbpp =========="

DIFFU_ID="apple/DiffuCoder-7B-Instruct"

remask_job diffucoder "$DIFFU_ID" \
    "$OUT/seed-coder-instruct_mbpp.jsonl" \
    "$OUT/seed-coder-instruct_diffucoder_remask_mbpp_t0.9.jsonl" \
    mbpp \
    --confidence_threshold 0.9

# ══════════════════════════════════════════════════════════════════════════════
# PHASE D — Stable-DiffCoder-8B refiner  (14 jobs: 7 AR × {humaneval, mbpp})
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

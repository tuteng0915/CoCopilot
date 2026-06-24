#!/usr/bin/env bash
# GPU 2 — Priority reorder: Phase C+D first, then LCB+BCB.
#
# Execution order:
#   Phase C (remainder): DiffuCoder refiner, seed-coder-instruct × mbpp       (~0.5h)
#   Phase D:             Stable-DiffCoder refiner, 7 AR × {humaneval,mbpp}    (~12h)
#   LCB Phase 1:         Dream+LLaDA refiner, {deepseek,llama31,qwen} × LCB   (skips completed, ~50h)
#   BCB Phase 2:         Dream+LLaDA refiner, {deepseek,llama31,qwen} × BCB   (~38h)
#
# tmux: tmux new-window -t gpu2_lcb_bcb "bash scripts/run_gpu2_cd_then_lcb_bcb.sh 2>&1 | tee /tmp/gpu2_cd_lcb_bcb.log"
set -uo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH=src
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export CUDA_VISIBLE_DEVICES=2

PY="/home/wjzhang/miniforge3/envs/cocoder/bin"
export PATH="$PY:$PATH"
PY="$PY/python"
OUT="outputs/base_tuteng"

DREAM_ID="Dream-org/Dream-Coder-v0-Instruct-7B"
LLADA_ID="GSAI-ML/LLaDA-8B-Instruct"
DIFFU_ID="apple/DiffuCoder-7B-Instruct"
SDIFF_ID="ByteDance-Seed/Stable-DiffCoder-8B-Instruct"

log() { printf '[%(%F %T)T] %s\n' -1 "$*"; }

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

eval_lcb() {
    local samples="$1"
    local stem="${samples%.jsonl}"
    "$PY" -m coder.scripts.eval_livebench \
        --benchmark livecodebench \
        --samples "$samples" \
        --out_judgments "${stem}_judgments.jsonl" \
        --out_summary   "${stem}_summary.json" \
        || log "WARN: LCB eval failed"
}

eval_bcb() {
    local samples="$1"
    local stem="${samples%.jsonl}"
    "$PY" -m coder.scripts.eval_bigcodebench \
        --samples "$samples" \
        --split instruct --subset full \
        --execution local \
        --out_summary "${stem}_summary.json" \
        || log "WARN: BCB eval failed"
}

remask_lcb_job() {
    local refiner="$1" model_id="$2" input="$3" out="$4"
    shift 4
    log "=== remask-lcb [$refiner] $(basename "$input") → $(basename "$out") ==="
    "$PY" -m coder.scripts.gen_remask \
        --refiner "$refiner" --model_id "$model_id" \
        --input "$input" --out "$out" \
        --device cuda:0 --resume "$@"
    eval_lcb "$out"
}

remask_bcb_job() {
    local refiner="$1" model_id="$2" input="$3" out="$4"
    shift 4
    log "=== remask-bcb [$refiner] $(basename "$input") → $(basename "$out") ==="
    "$PY" -m coder.scripts.gen_remask \
        --refiner "$refiner" --model_id "$model_id" \
        --input "$input" --out "$out" \
        --device cuda:0 --resume "$@"
    eval_bcb "$out"
}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE C (remainder) — DiffuCoder refiner: seed-coder-instruct × mbpp
# (other 13 jobs done on GPU 6; this one was at 54/378 when interrupted)
# ══════════════════════════════════════════════════════════════════════════════
log "========== PHASE C remainder: DiffuCoder seed-coder-instruct × mbpp =========="

remask_job diffucoder "$DIFFU_ID" \
    "$OUT/seed-coder-instruct_mbpp.jsonl" \
    "$OUT/seed-coder-instruct_diffucoder_remask_mbpp_t0.9.jsonl" \
    mbpp \
    --confidence_threshold 0.9

# ══════════════════════════════════════════════════════════════════════════════
# PHASE D — Stable-DiffCoder-8B refiner  (14 jobs: 7 AR × {humaneval, mbpp})
# ══════════════════════════════════════════════════════════════════════════════
log "========== PHASE D: Stable-DiffCoder refiner (14 jobs) =========="

for AR in deepseek qwen llama31 codellama mistral starcoder2 seed-coder-instruct; do
    for DS in humaneval mbpp; do
        remask_job seed-diffcoder "$SDIFF_ID" \
            "$OUT/${AR}_${DS}.jsonl" \
            "$OUT/${AR}_seeddiff_remask_${DS}_t0.9.jsonl" \
            "$DS" \
            --confidence_threshold 0.9
    done
done

# ══════════════════════════════════════════════════════════════════════════════
# LCB PHASE 1 — LiveCodeBench  (6 jobs: {deepseek,llama31,qwen} × {dream,llada})
# deepseek×dream and deepseek×llada are already done (--resume will skip them).
# llama31×dream was at 388/1055 — resume will continue from there.
# ══════════════════════════════════════════════════════════════════════════════
log "========== LCB Phase 1: gen_remask + eval =========="

for AR in deepseek llama31 qwen; do
    remask_lcb_job dream "$DREAM_ID" \
        "$OUT/${AR}_livecodebench.jsonl" \
        "$OUT/${AR}_dream_livecodebench_t0.9.jsonl" \
        --confidence_threshold 0.9

    remask_lcb_job llada "$LLADA_ID" \
        "$OUT/${AR}_livecodebench.jsonl" \
        "$OUT/${AR}_llada_livecodebench_t0.9_plnt3.jsonl" \
        --confidence_threshold 0.9 --protect_last_n_tokens 3
done

# ══════════════════════════════════════════════════════════════════════════════
# BCB PHASE 2 — BigCodeBench full  (6 jobs: {deepseek,llama31,qwen} × {dream,llada})
# ══════════════════════════════════════════════════════════════════════════════
log "========== BCB Phase 2: gen_remask + eval =========="

for AR in deepseek llama31 qwen; do
    INPUT="$OUT/${AR}_bigcodebench_instruct_full_pass1_clean.jsonl"

    remask_bcb_job dream "$DREAM_ID" \
        "$INPUT" \
        "$OUT/${AR}_dream_bigcodebench_instruct_full_t0.9.jsonl" \
        --confidence_threshold 0.9

    remask_bcb_job llada "$LLADA_ID" \
        "$INPUT" \
        "$OUT/${AR}_llada_bigcodebench_instruct_full_t0.9_plnt3.jsonl" \
        --confidence_threshold 0.9 --protect_last_n_tokens 3
done

log "========== ALL PHASES COMPLETE =========="

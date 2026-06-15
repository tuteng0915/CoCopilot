#!/usr/bin/env bash
set -euo pipefail

cd /home/wjzhang/tt_workspace/model/CoCoder/CoCoder
export PYTHONPATH=src

OUT=outputs/base_tuteng
LOG_DIR="$OUT/random_locator_queue"
mkdir -p "$LOG_DIR"

HE_RAW="$OUT/deepseek_random_locate_dream_rewrite_humaneval.jsonl"
MBPP_RAW="$OUT/deepseek_random_locate_dream_rewrite_mbpp.jsonl"
HE_SAN="$OUT/deepseek_random_locate_dream_rewrite_humaneval-sanitized.jsonl"
MBPP_SAN="$OUT/deepseek_random_locate_dream_rewrite_mbpp-sanitized.jsonl"

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*"
}

run_gen() {
  local dataset="$1"
  local input="$2"
  local out="$3"
  log "generate random-locate + dream-rewrite: $dataset"
  CUDA_VISIBLE_DEVICES=0 conda run -n elf python -m coder.scripts.gen_remask \
    --locator random \
    --locator_model_id 3407 \
    --refiner dream \
    --input "$input" \
    --out "$out" \
    --mask_ratio 0.10 \
    --temperature 0.1 --top_p 0.95 --seed 3407 \
    --device cuda:0 \
    --record_mask_stats \
    --resume
}

run_eval() {
  local dataset="$1"
  local raw="$2"
  local sanitized="$3"
  local stem="$4"
  log "postprocess $dataset $raw"
  conda run -n elf python -m coder.scripts.postprocess_evalplus \
    --dataset "$dataset" \
    --samples "$raw"

  log "eval $dataset $sanitized"
  conda run -n elf python -m coder.scripts.eval_evalplus \
    --backend local \
    --dataset "$dataset" \
    --samples "$sanitized" \
    --summary_model "$stem" \
    --summary_out "$OUT/${stem}_summary.json" \
    --output_file "$OUT/${stem}_eval_results.json"
}

run_gen humaneval "$OUT/deepseek_humaneval.jsonl" "$HE_RAW"
run_eval humaneval "$HE_RAW" "$HE_SAN" deepseek_random_locate_dream_rewrite_humaneval

run_gen mbpp "$OUT/deepseek_mbpp.jsonl" "$MBPP_RAW"
run_eval mbpp "$MBPP_RAW" "$MBPP_SAN" deepseek_random_locate_dream_rewrite_mbpp

touch "$LOG_DIR/done"
log "random locator queue done"

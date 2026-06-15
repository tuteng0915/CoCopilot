#!/usr/bin/env bash
set -euo pipefail

cd /home/wjzhang/tt_workspace/model/CoCoder/CoCoder
export PYTHONPATH=src
OUT=outputs/base_tuteng
STATE="$OUT/table3_queue"
mkdir -p "$STATE" "$STATE/logs"

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*"
}

wait_marker() {
  local marker="$1"
  log "waiting for $marker"
  while [[ ! -f "$marker" ]]; do
    sleep 120
  done
}

run_eval() {
  local dataset="$1"
  local samples="$2"
  local summary_model="$3"
  local summary_out="$4"
  log "eval $dataset $samples"
  conda run -n elf python -m coder.scripts.eval_evalplus \
    --samples "$samples" \
    --dataset "$dataset" \
    --summary_model "$summary_model" \
    --summary_out "$summary_out" \
    --backend local
}

run_locate() {
  local ar_model="$1"
  local model_id="$2"
  local dataset="$3"
  local input="$4"
  local out="$5"
  log "locate+rewrite $ar_model $dataset"
  if [[ -n "$model_id" ]]; then
    CUDA_VISIBLE_DEVICES=0,1 conda run -n elf python -m coder.scripts.gen_locate_ar_rewrite \
      --ar_model "$ar_model" \
      --model_id "$model_id" \
      --ar_device cuda:1 \
      --locator_device cuda:0 \
      --input "$input" \
      --out "$out" \
      --confidence_threshold 0.9 --resume
  else
    CUDA_VISIBLE_DEVICES=0,1 conda run -n elf python -m coder.scripts.gen_locate_ar_rewrite \
      --ar_model "$ar_model" \
      --ar_device cuda:1 \
      --locator_device cuda:0 \
      --input "$input" \
      --out "$out" \
      --confidence_threshold 0.9 --resume
  fi
}

wait_marker "$STATE/gpu0.done"
wait_marker "$STATE/gpu1.done"

run_locate codellama "" humaneval "$OUT/codellama_humaneval.jsonl" "$OUT/codellama_humaneval_locate_ar_rewrite_t0.9.jsonl"
run_eval humaneval "$OUT/codellama_humaneval_locate_ar_rewrite_t0.9.jsonl" \
  codellama_locate_ar_rewrite_t0.9 "$OUT/codellama_humaneval_locate_ar_rewrite_t0.9_summary.json"

run_locate codellama "" mbpp "$OUT/codellama_mbpp.jsonl" "$OUT/codellama_mbpp_locate_ar_rewrite_t0.9.jsonl"
run_eval mbpp "$OUT/codellama_mbpp_locate_ar_rewrite_t0.9.jsonl" \
  codellama_locate_ar_rewrite_t0.9 "$OUT/codellama_mbpp_locate_ar_rewrite_t0.9_summary.json"

run_locate mistral "" humaneval "$OUT/mistral_humaneval.jsonl" "$OUT/mistral_humaneval_locate_ar_rewrite_t0.9.jsonl"
run_eval humaneval "$OUT/mistral_humaneval_locate_ar_rewrite_t0.9.jsonl" \
  mistral_locate_ar_rewrite_t0.9 "$OUT/mistral_humaneval_locate_ar_rewrite_t0.9_summary.json"

run_locate mistral "" mbpp "$OUT/mistral_mbpp.jsonl" "$OUT/mistral_mbpp_locate_ar_rewrite_t0.9.jsonl"
run_eval mbpp "$OUT/mistral_mbpp_locate_ar_rewrite_t0.9.jsonl" \
  mistral_locate_ar_rewrite_t0.9 "$OUT/mistral_mbpp_locate_ar_rewrite_t0.9_summary.json"

run_locate seed-coder ByteDance-Seed/Seed-Coder-8B-Instruct humaneval \
  "$OUT/seed-coder-instruct_humaneval.jsonl" "$OUT/seed-coder-instruct_humaneval_locate_ar_rewrite_t0.9.jsonl"
run_eval humaneval "$OUT/seed-coder-instruct_humaneval_locate_ar_rewrite_t0.9.jsonl" \
  seed_coder_instruct_locate_ar_rewrite_t0.9 "$OUT/seed-coder-instruct_humaneval_locate_ar_rewrite_t0.9_summary.json"

run_locate seed-coder ByteDance-Seed/Seed-Coder-8B-Instruct mbpp \
  "$OUT/seed-coder-instruct_mbpp.jsonl" "$OUT/seed-coder-instruct_mbpp_locate_ar_rewrite_t0.9.jsonl"
run_eval mbpp "$OUT/seed-coder-instruct_mbpp_locate_ar_rewrite_t0.9.jsonl" \
  seed_coder_instruct_locate_ar_rewrite_t0.9 "$OUT/seed-coder-instruct_mbpp_locate_ar_rewrite_t0.9_summary.json"

touch "$STATE/locate.done"
log "locate queue done"

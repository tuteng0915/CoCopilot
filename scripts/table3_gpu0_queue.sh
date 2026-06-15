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

wait_jsonl_count() {
  local file="$1"
  local expected="$2"
  log "waiting for $file to reach $expected lines"
  while true; do
    local n=0
    if [[ -f "$file" ]]; then
      n=$(wc -l < "$file")
    fi
    if [[ "$n" -ge "$expected" ]]; then
      log "$file has $n lines"
      break
    fi
    sleep 60
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

run_gen_gpu0() {
  log "run: $*"
  CUDA_VISIBLE_DEVICES=0 conda run -n elf python -m "$@"
}

# Current non-tmux CodeLlama MBPP Reflexion may already be running.
wait_jsonl_count "$OUT/codellama_mbpp_reflexion_feedback_r1.jsonl" 378
run_eval mbpp "$OUT/codellama_mbpp_reflexion_feedback_r1.jsonl" \
  codellama_reflexion_feedback_r1 "$OUT/codellama_mbpp_reflexion_feedback_r1_summary.json"

run_gen_gpu0 coder.scripts.gen_self_refine \
  --model seed-coder \
  --model_id ByteDance-Seed/Seed-Coder-8B-Instruct \
  --input "$OUT/seed-coder-instruct_humaneval.jsonl" \
  --out "$OUT/seed-coder-instruct_humaneval_selfrefine_r1.jsonl" \
  --device cuda:0 --resume
run_eval humaneval "$OUT/seed-coder-instruct_humaneval_selfrefine_r1.jsonl" \
  seed_coder_instruct_selfrefine_r1 "$OUT/seed-coder-instruct_humaneval_selfrefine_r1_summary.json"

run_gen_gpu0 coder.scripts.gen_self_refine \
  --model seed-coder \
  --model_id ByteDance-Seed/Seed-Coder-8B-Instruct \
  --input "$OUT/seed-coder-instruct_mbpp.jsonl" \
  --out "$OUT/seed-coder-instruct_mbpp_selfrefine_r1.jsonl" \
  --device cuda:0 --resume
run_eval mbpp "$OUT/seed-coder-instruct_mbpp_selfrefine_r1.jsonl" \
  seed_coder_instruct_selfrefine_r1 "$OUT/seed-coder-instruct_mbpp_selfrefine_r1_summary.json"

run_gen_gpu0 coder.scripts.gen_reflexion \
  --model seed-coder \
  --model_id ByteDance-Seed/Seed-Coder-8B-Instruct \
  --input "$OUT/seed-coder-instruct_humaneval-sanitized.jsonl" \
  --raw_input "$OUT/seed-coder-instruct_humaneval.jsonl" \
  --out "$OUT/seed-coder-instruct_humaneval_reflexion_feedback_r1.jsonl" \
  --feedback_key eval.error --device cuda:0 --resume
run_eval humaneval "$OUT/seed-coder-instruct_humaneval_reflexion_feedback_r1.jsonl" \
  seed_coder_instruct_reflexion_feedback_r1 "$OUT/seed-coder-instruct_humaneval_reflexion_feedback_r1_summary.json"

run_gen_gpu0 coder.scripts.gen_reflexion \
  --model seed-coder \
  --model_id ByteDance-Seed/Seed-Coder-8B-Instruct \
  --input "$OUT/seed-coder-instruct_mbpp-sanitized.jsonl" \
  --raw_input "$OUT/seed-coder-instruct_mbpp.jsonl" \
  --out "$OUT/seed-coder-instruct_mbpp_reflexion_feedback_r1.jsonl" \
  --feedback_key eval.error --device cuda:0 --resume
run_eval mbpp "$OUT/seed-coder-instruct_mbpp_reflexion_feedback_r1.jsonl" \
  seed_coder_instruct_reflexion_feedback_r1 "$OUT/seed-coder-instruct_mbpp_reflexion_feedback_r1_summary.json"

run_gen_gpu0 coder.scripts.gen_rerank \
  --model seed-coder \
  --model_id ByteDance-Seed/Seed-Coder-8B-Instruct \
  --dataset humaneval \
  --out "$OUT/seed-coder-instruct_humaneval_rerank_logprob_k8.jsonl" \
  --num_samples 8 --score_mode logprob --device cuda:0 --resume
run_eval humaneval "$OUT/seed-coder-instruct_humaneval_rerank_logprob_k8.jsonl" \
  seed_coder_instruct_rerank_logprob_k8 "$OUT/seed-coder-instruct_humaneval_rerank_logprob_k8_summary.json"

run_gen_gpu0 coder.scripts.gen_rerank \
  --model seed-coder \
  --model_id ByteDance-Seed/Seed-Coder-8B-Instruct \
  --dataset mbpp \
  --out "$OUT/seed-coder-instruct_mbpp_rerank_logprob_k8.jsonl" \
  --num_samples 8 --score_mode logprob --device cuda:0 --resume
run_eval mbpp "$OUT/seed-coder-instruct_mbpp_rerank_logprob_k8.jsonl" \
  seed_coder_instruct_rerank_logprob_k8 "$OUT/seed-coder-instruct_mbpp_rerank_logprob_k8_summary.json"

touch "$STATE/gpu0.done"
log "gpu0 queue done"

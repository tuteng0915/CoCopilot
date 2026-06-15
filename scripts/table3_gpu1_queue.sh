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

run_gen_gpu1() {
  log "run: $*"
  CUDA_VISIBLE_DEVICES=1 conda run -n elf python -m "$@"
}

# Current non-tmux Mistral MBPP Reflexion may already be running.
wait_jsonl_count "$OUT/mistral_mbpp_reflexion_feedback_r1.jsonl" 378
run_eval mbpp "$OUT/mistral_mbpp_reflexion_feedback_r1.jsonl" \
  mistral_reflexion_feedback_r1 "$OUT/mistral_mbpp_reflexion_feedback_r1_summary.json"

run_gen_gpu1 coder.scripts.gen_rerank \
  --model codellama \
  --dataset humaneval \
  --out "$OUT/codellama_humaneval_rerank_logprob_k8.jsonl" \
  --num_samples 8 --score_mode logprob --device cuda:0 --resume
run_eval humaneval "$OUT/codellama_humaneval_rerank_logprob_k8.jsonl" \
  codellama_rerank_logprob_k8 "$OUT/codellama_humaneval_rerank_logprob_k8_summary.json"

run_gen_gpu1 coder.scripts.gen_rerank \
  --model codellama \
  --dataset mbpp \
  --out "$OUT/codellama_mbpp_rerank_logprob_k8.jsonl" \
  --num_samples 8 --score_mode logprob --device cuda:0 --resume
run_eval mbpp "$OUT/codellama_mbpp_rerank_logprob_k8.jsonl" \
  codellama_rerank_logprob_k8 "$OUT/codellama_mbpp_rerank_logprob_k8_summary.json"

run_gen_gpu1 coder.scripts.gen_rerank \
  --model mistral \
  --dataset humaneval \
  --out "$OUT/mistral_humaneval_rerank_logprob_k8.jsonl" \
  --num_samples 8 --score_mode logprob --device cuda:0 --resume
run_eval humaneval "$OUT/mistral_humaneval_rerank_logprob_k8.jsonl" \
  mistral_rerank_logprob_k8 "$OUT/mistral_humaneval_rerank_logprob_k8_summary.json"

run_gen_gpu1 coder.scripts.gen_rerank \
  --model mistral \
  --dataset mbpp \
  --out "$OUT/mistral_mbpp_rerank_logprob_k8.jsonl" \
  --num_samples 8 --score_mode logprob --device cuda:0 --resume
run_eval mbpp "$OUT/mistral_mbpp_rerank_logprob_k8.jsonl" \
  mistral_rerank_logprob_k8 "$OUT/mistral_mbpp_rerank_logprob_k8_summary.json"

touch "$STATE/gpu1.done"
log "gpu1 queue done"

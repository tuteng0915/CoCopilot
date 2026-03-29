#!/usr/bin/env bash

###############################################################################
# 简单实验命令示例（供手动复制粘贴用）
# 这个文件不会被脚本调用，只是一个「命令备忘录」。
###############################################################################

## 1) 单次 DeepSeek-Coder on HumanEval
# python -m coder.scripts.gen_evalplus \
#   --model deepseek \
#   --dataset humaneval \
#   --out outputs/deepseek_humaneval.jsonl \
#   --max_new_tokens 512 \
#   --temperature 0.0 \
#   --top_p 1.0 \
#   --seed 3407
#
# python -m coder.scripts.postprocess_evalplus \
#   --dataset humaneval \
#   --samples outputs/deepseek_humaneval.jsonl
#
# python -m coder.scripts.eval_evalplus \
#   --backend local \
#   --dataset humaneval \
#   --samples outputs/deepseek_humaneval-sanitized.jsonl


## 2) 多次 roll-out + pass@k（HumanEval）
# # 生成多次 roll-out（不同 seed）
# python -m coder.scripts.gen_evalplus --model deepseek --dataset humaneval \
#   --out outputs/deepseek_humaneval_seed3407.jsonl --max_new_tokens 512 \
#   --temperature 0.7 --top_p 0.95 --seed 3407
# python -m coder.scripts.gen_evalplus --model deepseek --dataset humaneval \
#   --out outputs/deepseek_humaneval_seed3408.jsonl --max_new_tokens 512 \
#   --temperature 0.7 --top_p 0.95 --seed 3408
# python -m coder.scripts.gen_evalplus --model deepseek --dataset humaneval \
#   --out outputs/deepseek_humaneval_seed3409.jsonl --max_new_tokens 512 \
#   --temperature 0.7 --top_p 0.95 --seed 3409
#
# # 合并为一个多样本文件，再做 sanitize + eval（EvalPlus 会自动算 pass@k）
# cat outputs/deepseek_humaneval_seed*.jsonl > outputs/deepseek_multi_humaneval.jsonl
# python -m coder.scripts.postprocess_evalplus \
#   --dataset humaneval \
#   --samples outputs/deepseek_multi_humaneval.jsonl
# python -m coder.scripts.eval_evalplus \
#   --backend local \
#   --dataset humaneval \
#   --samples outputs/deepseek_multi_humaneval-sanitized.jsonl


## 3) LiveBench-Coding DeepSeek → DreamCoder remask
# python -m coder.scripts.gen_livebench \
#   --model deepseek \
#   --out outputs/deepseek_livebench.jsonl \
#   --max_new_tokens 768 \
#   --temperature 0.0 \
#   --top_p 1.0
#
# python -m coder.scripts.gen_remask \
#   --input outputs/deepseek_livebench.jsonl \
#   --out   outputs/remask_livebench_t0.8.jsonl \
#   --confidence_threshold 0.8 \
#   --temperature 0.0 \
#   --top_p 1.0 \
#   --seed 3407
#
# python -m coder.scripts.eval_livebench \
#   --samples outputs/deepseek_livebench.jsonl \
#   --out_judgments outputs/deepseek_livebench_judgments.jsonl \
#   --out_summary   outputs/deepseek_livebench_summary.json
#
# python -m coder.scripts.eval_livebench \
#   --samples outputs/remask_livebench_t0.8.jsonl \
#   --out_judgments outputs/remask_livebench_t0.8_judgments.jsonl \
#   --out_summary   outputs/remask_livebench_t0.8_summary.json


## 4) EvalPlus DeepSeek → Self-Refine（HumanEval）
# python -m coder.scripts.gen_self_refine \
#   --input outputs/deepseek_humaneval.jsonl \
#   --out   outputs/selfrefine_deepseek_humaneval.jsonl \
#   --model deepseek \
#   --max_new_tokens 512 \
#   --temperature 0.2 \
#   --top_p 0.95 \
#   --seed 3407
#
# python -m coder.scripts.postprocess_evalplus \
#   --dataset humaneval \
#   --samples outputs/selfrefine_deepseek_humaneval.jsonl
#
# python -m coder.scripts.eval_evalplus \
#   --backend local \
#   --dataset humaneval \
#   --samples outputs/selfrefine_deepseek_humaneval-sanitized.jsonl


## 5) EvalPlus DeepSeek + Reranking（HumanEval）
# python -m coder.scripts.gen_rerank \
#   --model deepseek \
#   --dataset humaneval \
#   --out outputs/rerank_deepseek_humaneval.jsonl \
#   --num_samples 8 \
#   --max_new_tokens 512 \
#   --temperature 0.7 \
#   --top_p 0.95 \
#   --seed 3407
#
# python -m coder.scripts.postprocess_evalplus \
#   --dataset humaneval \
#   --samples outputs/rerank_deepseek_humaneval.jsonl
#
# python -m coder.scripts.eval_evalplus \
#   --backend local \
#   --dataset humaneval \
#   --samples outputs/rerank_deepseek_humaneval-sanitized.jsonl

#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Example experiment recipes for this repo.
#
# These commands show:
#   1) Single-rollout EvalPlus runs.
#   2) Multi-rollout EvalPlus runs for pass@k (concat JSONL).
#   3) LiveBench-Coding + DreamCoder remask.
#   4) AR + Self-Refine pipeline.
#   5) AR + Reranking pipeline.
#
# Usage:
#   bash scripts/exp_examples.sh evalplus_single
#   bash scripts/exp_examples.sh evalplus_multi
#   bash scripts/exp_examples.sh livebench_remask
#   bash scripts/exp_examples.sh self_refine
#   bash scripts/exp_examples.sh rerank
###############################################################################

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"


evalplus_single() {
  local dataset="${1:-humaneval}"  # humaneval | mbpp

  # DeepSeek-Coder single rollout
  python -m coder.scripts.gen_evalplus \
    --model deepseek \
    --dataset "${dataset}" \
    --out "outputs/deepseek_${dataset}.jsonl" \
    --max_new_tokens 512 \
    --temperature 0.0 \
    --top_p 1.0 \
    --seed 3407

  python -m coder.scripts.postprocess_evalplus \
    --dataset "${dataset}" \
    --samples "outputs/deepseek_${dataset}.jsonl"

  python -m coder.scripts.eval_evalplus \
    --backend local \
    --dataset "${dataset}" \
    --samples "outputs/deepseek_${dataset}-sanitized.jsonl"
}


evalplus_multi() {
  local dataset="${1:-humaneval}"  # humaneval | mbpp
  local base="deepseek_multi_${dataset}"
  local seeds=(3407 3408 3409)

  mkdir -p outputs

  # 1) Generate multiple rollouts with different seeds (one sample / task / seed).
  for s in "${seeds[@]}"; do
    python -m coder.scripts.gen_evalplus \
      --model deepseek \
      --dataset "${dataset}" \
      --out "outputs/${base}_seed${s}.jsonl" \
      --max_new_tokens 512 \
      --temperature 0.7 \
      --top_p 0.95 \
      --seed "${s}"
  done

  # 2) Concatenate to form a multi-sample JSONL.
  cat "outputs/${base}_seed"*.jsonl > "outputs/${base}.jsonl"

  # 3) Sanitize once and evaluate; EvalPlus will compute pass@k from multi-samples.
  python -m coder.scripts.postprocess_evalplus \
    --dataset "${dataset}" \
    --samples "outputs/${base}.jsonl"

  python -m coder.scripts.eval_evalplus \
    --backend local \
    --dataset "${dataset}" \
    --samples "outputs/${base}-sanitized.jsonl"
}


livebench_remask() {
  # 1) Generate LiveBench-Coding answers with DeepSeek-Coder (AR).
  python -m coder.scripts.gen_livebench \
    --model deepseek \
    --out outputs/deepseek_livebench.jsonl \
    --max_new_tokens 768 \
    --temperature 0.0 \
    --top_p 1.0

  # 2) Refine with DreamCoder token remasking (uses prompt/raw_completion from gen_livebench.py).
  python -m coder.scripts.gen_remask \
    --input outputs/deepseek_livebench.jsonl \
    --out   outputs/remask_livebench_t0.8.jsonl \
    --confidence_threshold 0.8 \
    --temperature 0.0 \
    --top_p 1.0 \
    --seed 3407

  # 3) Evaluate both AR and remasked outputs on LiveBench-Coding.
  python -m coder.scripts.eval_livebench \
    --samples outputs/deepseek_livebench.jsonl \
    --out_judgments outputs/deepseek_livebench_judgments.jsonl \
    --out_summary   outputs/deepseek_livebench_summary.json

  python -m coder.scripts.eval_livebench \
    --samples outputs/remask_livebench_t0.8.jsonl \
    --out_judgments outputs/remask_livebench_t0.8_judgments.jsonl \
    --out_summary   outputs/remask_livebench_t0.8_summary.json
}


self_refine() {
  local dataset="${1:-humaneval}"  # humaneval | mbpp

  # Assume DeepSeek-Coder single-rollout EvalPlus samples already exist.
  # If not, you can generate them via evalplus_single.
  local in_path="outputs/deepseek_${dataset}.jsonl"
  local out_path="outputs/selfrefine_deepseek_${dataset}.jsonl"

  python -m coder.scripts.gen_self_refine \
    --input "${in_path}" \
    --out   "${out_path}" \
    --model deepseek \
    --max_new_tokens 512 \
    --temperature 0.2 \
    --top_p 0.95 \
    --seed 3407

  python -m coder.scripts.postprocess_evalplus \
    --dataset "${dataset}" \
    --samples "${out_path}"

  python -m coder.scripts.eval_evalplus \
    --backend local \
    --dataset "${dataset}" \
    --samples "${out_path%-*.jsonl}-sanitized.jsonl"
}


rerank() {
  local dataset="${1:-humaneval}"  # humaneval | mbpp

  python -m coder.scripts.gen_rerank \
    --model deepseek \
    --dataset "${dataset}" \
    --out "outputs/rerank_deepseek_${dataset}.jsonl" \
    --num_samples 8 \
    --max_new_tokens 512 \
    --temperature 0.7 \
    --top_p 0.95 \
    --seed 3407

  python -m coder.scripts.postprocess_evalplus \
    --dataset "${dataset}" \
    --samples "outputs/rerank_deepseek_${dataset}.jsonl"

  python -m coder.scripts.eval_evalplus \
    --backend local \
    --dataset "${dataset}" \
    --samples "outputs/rerank_deepseek_${dataset}-sanitized.jsonl"
}


main() {
  local cmd="${1:-}"
  shift || true

  case "${cmd}" in
    evalplus_single)   evalplus_single "$@";;
    evalplus_multi)    evalplus_multi "$@";;
    livebench_remask)  livebench_remask "$@";;
    self_refine)       self_refine "$@";;
    rerank)            rerank "$@";;
    *)
      echo "Usage: $0 {evalplus_single|evalplus_multi|livebench_remask|self_refine|rerank} [dataset]"
      exit 1
      ;;
  esac
}

main "$@"


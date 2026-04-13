#!/usr/bin/env bash
# Table 4 baseline jobs — DeepSeek timing补跑 + Qwen2.5-Coder 7B 全量基线
#
# 运行前提：
#   - qwen_humaneval.jsonl / qwen_mbpp.jsonl 已存在（accuracy 用途）
#   - qwen_humaneval-sanitized.jsonl / qwen_mbpp-sanitized.jsonl 已存在（reflexion input）
#   - deepseek_humaneval.jsonl / deepseek_mbpp.jsonl 已存在（dream remask input）
#
# GPU 配置（按需修改）：
#   GPU_DS=0        DeepSeek 模型（Rerank timing, ~14GB）
#   GPU_QWEN=2      Qwen 模型（self-refine / reflexion / rerank / locate-ar AR, ~14GB）
#   GPU_LLADA=5     LLaDA 精炼器（remask with LLaDA, ~16GB）
#   GPU_LOC=4       Dream Locator（locate-ar-rewrite 定位器, ~14GB；与 GPU_QWEN 同时加载）
#   GPU_DREAM=7     Dream 精炼器（dream remask timing, 很慢 ~25h）
#
# 分组说明（可分别运行）：
#   Group A  -- DeepSeek Rerank 补 timing（只需 gen，不需 eval）
#   Group B  -- Qwen: self-refine + reflexion + rerank（gen + eval）
#   Group C  -- Qwen: locate-ar-rewrite（gen + eval，需 GPU_LOC + GPU_QWEN 同时可用）
#   Group D  -- LLaDA remask：deepseek+llada MBPP + qwen+llada HE+MBPP（gen + eval）
#   Group E  -- DeepSeek Dream remask 补 timing（只需 gen，很慢）
#
# 用法：
#   bash scripts/gen_table4_baselines.sh [--groups ABCDE] [--dry-run]
#   默认运行全部 groups，并行启动

set -euo pipefail
ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

source /home/tteng/miniconda3/etc/profile.d/conda.sh
conda activate code
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
DRY_RUN=0
RUN_GROUPS="ABCDE"
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --groups=*) RUN_GROUPS="${arg#--groups=}" ;;
    --groups) shift; RUN_GROUPS="$1" ;;
  esac
done

# ---------------------------------------------------------------------------
# GPU 配置（按需修改）
# ---------------------------------------------------------------------------
GPU_DS="${GPU_DS:-0}"
GPU_QWEN="${GPU_QWEN:-2}"
GPU_LLADA="${GPU_LLADA:-5}"
GPU_LOC="${GPU_LOC:-4}"      # Dream locator for locate-ar-rewrite
GPU_DREAM="${GPU_DREAM:-7}"

OUT="outputs/base_tuteng"

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
_run() {
  local phys_gpu="$1"; shift
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[cmd] CUDA_VISIBLE_DEVICES=${phys_gpu} $*"
  else
    CUDA_VISIBLE_DEVICES="${phys_gpu}" "$@"
  fi
}

_run2() {
  # Two-GPU command: locator on GPU1, AR on GPU2
  local gpu1="$1"; local gpu2="$2"; shift 2
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[cmd] CUDA_VISIBLE_DEVICES=${gpu1},${gpu2} $*"
  else
    CUDA_VISIBLE_DEVICES="${gpu1},${gpu2}" "$@"
  fi
}

gen_step() { echo ""; echo ">>> [GEN] $*"; }
eval_step() { echo ""; echo ">>> [EVAL] $*"; }

sanitize_and_eval() {
  # sanitize_and_eval DATASET JSONL_PATH SUMMARY_OUT
  local dataset="$1"
  local jsonl="$2"
  local summary_out="$3"
  # postprocess (sanitize)
  python -m coder.scripts.postprocess_evalplus \
    --dataset "$dataset" \
    --samples "$jsonl"
  # eval
  local sanitized="${jsonl%.jsonl}-sanitized.jsonl"
  python -m coder.scripts.eval_evalplus \
    --backend local \
    --dataset "$dataset" \
    --samples "$sanitized" \
    --summary_out "$summary_out"
}

# ---------------------------------------------------------------------------
# Group A — DeepSeek Rerank logprob k=8（timing only）
# ---------------------------------------------------------------------------
run_group_A() {
  local LOG="${OUT}/table4_groupA.log"
  {
    gen_step "deepseek rerank HE (timing only)"
    _run $GPU_DS python -m coder.scripts.gen_rerank \
      --model deepseek \
      --dataset humaneval \
      --out "${OUT}/deepseek_humaneval_rerank_logprob_k8_timed.jsonl" \
      --num_samples 8 \
      --score_mode logprob \
      --device cuda:0

    gen_step "deepseek rerank MBPP (timing only)"
    _run $GPU_DS python -m coder.scripts.gen_rerank \
      --model deepseek \
      --dataset mbpp \
      --out "${OUT}/deepseek_mbpp_rerank_logprob_k8_timed.jsonl" \
      --num_samples 8 \
      --score_mode logprob \
      --device cuda:0

    echo "[DONE] Group A"
  } 2>&1 | tee "$LOG"
}

# ---------------------------------------------------------------------------
# Group B — Qwen: self-refine + reflexion + rerank（gen + sanitize + eval）
# ---------------------------------------------------------------------------
run_group_B() {
  local LOG="${OUT}/table4_groupB.log"
  {
    # --- Self-Refine ---
    gen_step "qwen self-refine HE"
    _run $GPU_QWEN python -m coder.scripts.gen_self_refine \
      --model qwen \
      --input "${OUT}/qwen_humaneval.jsonl" \
      --out "${OUT}/qwen_humaneval_selfrefine_r1.jsonl" \
      --device cuda:0
    if [[ $DRY_RUN -eq 0 ]]; then
      sanitize_and_eval humaneval \
        "${OUT}/qwen_humaneval_selfrefine_r1.jsonl" \
        "${OUT}/qwen_humaneval_selfrefine_r1_summary.json"
    fi

    gen_step "qwen self-refine MBPP"
    _run $GPU_QWEN python -m coder.scripts.gen_self_refine \
      --model qwen \
      --input "${OUT}/qwen_mbpp.jsonl" \
      --out "${OUT}/qwen_mbpp_selfrefine_r1.jsonl" \
      --device cuda:0
    if [[ $DRY_RUN -eq 0 ]]; then
      sanitize_and_eval mbpp \
        "${OUT}/qwen_mbpp_selfrefine_r1.jsonl" \
        "${OUT}/qwen_mbpp_selfrefine_r1_summary.json"
    fi

    # --- Reflexion ---
    gen_step "qwen reflexion HE"
    _run $GPU_QWEN python -m coder.scripts.gen_reflexion \
      --model qwen \
      --input "${OUT}/qwen_humaneval-sanitized.jsonl" \
      --raw_input "${OUT}/qwen_humaneval.jsonl" \
      --out "${OUT}/qwen_humaneval_reflexion_feedback_r1.jsonl" \
      --device cuda:0
    if [[ $DRY_RUN -eq 0 ]]; then
      sanitize_and_eval humaneval \
        "${OUT}/qwen_humaneval_reflexion_feedback_r1.jsonl" \
        "${OUT}/qwen_humaneval_reflexion_feedback_r1_summary.json"
    fi

    gen_step "qwen reflexion MBPP"
    _run $GPU_QWEN python -m coder.scripts.gen_reflexion \
      --model qwen \
      --input "${OUT}/qwen_mbpp-sanitized.jsonl" \
      --raw_input "${OUT}/qwen_mbpp.jsonl" \
      --out "${OUT}/qwen_mbpp_reflexion_feedback_r1.jsonl" \
      --device cuda:0
    if [[ $DRY_RUN -eq 0 ]]; then
      sanitize_and_eval mbpp \
        "${OUT}/qwen_mbpp_reflexion_feedback_r1.jsonl" \
        "${OUT}/qwen_mbpp_reflexion_feedback_r1_summary.json"
    fi

    # --- Rerank k=8 ---
    gen_step "qwen rerank HE"
    _run $GPU_QWEN python -m coder.scripts.gen_rerank \
      --model qwen \
      --dataset humaneval \
      --out "${OUT}/qwen_humaneval_rerank_logprob_k8.jsonl" \
      --num_samples 8 \
      --score_mode logprob \
      --device cuda:0
    if [[ $DRY_RUN -eq 0 ]]; then
      sanitize_and_eval humaneval \
        "${OUT}/qwen_humaneval_rerank_logprob_k8.jsonl" \
        "${OUT}/qwen_humaneval_rerank_logprob_k8_summary.json"
    fi

    gen_step "qwen rerank MBPP"
    _run $GPU_QWEN python -m coder.scripts.gen_rerank \
      --model qwen \
      --dataset mbpp \
      --out "${OUT}/qwen_mbpp_rerank_logprob_k8.jsonl" \
      --num_samples 8 \
      --score_mode logprob \
      --device cuda:0
    if [[ $DRY_RUN -eq 0 ]]; then
      sanitize_and_eval mbpp \
        "${OUT}/qwen_mbpp_rerank_logprob_k8.jsonl" \
        "${OUT}/qwen_mbpp_rerank_logprob_k8_summary.json"
    fi

    echo "[DONE] Group B"
  } 2>&1 | tee "$LOG"
}

# ---------------------------------------------------------------------------
# Group C — Qwen Locate-AR-Rewrite（Dream locator on GPU_LOC, Qwen AR on GPU_QWEN）
# ---------------------------------------------------------------------------
run_group_C() {
  local LOG="${OUT}/table4_groupC.log"
  {
    gen_step "qwen locate-ar-rewrite HE"
    _run2 $GPU_LOC $GPU_QWEN python -m coder.scripts.gen_locate_ar_rewrite \
      --ar_model qwen \
      --input "${OUT}/qwen_humaneval.jsonl" \
      --out "${OUT}/qwen_humaneval_locate_ar_rewrite_t0.9.jsonl" \
      --confidence_threshold 0.9 \
      --locator_device cuda:0 \
      --ar_device cuda:1
    if [[ $DRY_RUN -eq 0 ]]; then
      sanitize_and_eval humaneval \
        "${OUT}/qwen_humaneval_locate_ar_rewrite_t0.9.jsonl" \
        "${OUT}/qwen_humaneval_locate_ar_rewrite_t0.9_summary.json"
    fi

    gen_step "qwen locate-ar-rewrite MBPP"
    _run2 $GPU_LOC $GPU_QWEN python -m coder.scripts.gen_locate_ar_rewrite \
      --ar_model qwen \
      --input "${OUT}/qwen_mbpp.jsonl" \
      --out "${OUT}/qwen_mbpp_locate_ar_rewrite_t0.9.jsonl" \
      --confidence_threshold 0.9 \
      --locator_device cuda:0 \
      --ar_device cuda:1
    if [[ $DRY_RUN -eq 0 ]]; then
      sanitize_and_eval mbpp \
        "${OUT}/qwen_mbpp_locate_ar_rewrite_t0.9.jsonl" \
        "${OUT}/qwen_mbpp_locate_ar_rewrite_t0.9_summary.json"
    fi

    echo "[DONE] Group C"
  } 2>&1 | tee "$LOG"
}

# ---------------------------------------------------------------------------
# Group D — LLaDA remask: deepseek+llada MBPP + qwen+llada HE+MBPP
# ---------------------------------------------------------------------------
run_group_D() {
  local LOG="${OUT}/table4_groupD.log"
  {
    gen_step "deepseek+llada remask MBPP (补跑)"
    _run $GPU_LLADA python -m coder.scripts.gen_remask \
      --refiner llada \
      --input "${OUT}/deepseek_mbpp.jsonl" \
      --out "${OUT}/deepseek_llada_remask_mbpp_t0.9.jsonl" \
      --confidence_threshold 0.9 \
      --device cuda:0
    if [[ $DRY_RUN -eq 0 ]]; then
      sanitize_and_eval mbpp \
        "${OUT}/deepseek_llada_remask_mbpp_t0.9.jsonl" \
        "${OUT}/deepseek_llada_remask_mbpp_t0.9_summary.json"
    fi

    gen_step "qwen+llada remask HE"
    _run $GPU_LLADA python -m coder.scripts.gen_remask \
      --refiner llada \
      --input "${OUT}/qwen_humaneval.jsonl" \
      --out "${OUT}/qwen_llada_remask_humaneval_t0.9.jsonl" \
      --confidence_threshold 0.9 \
      --device cuda:0
    if [[ $DRY_RUN -eq 0 ]]; then
      sanitize_and_eval humaneval \
        "${OUT}/qwen_llada_remask_humaneval_t0.9.jsonl" \
        "${OUT}/qwen_llada_remask_humaneval_t0.9_summary.json"
    fi

    gen_step "qwen+llada remask MBPP"
    _run $GPU_LLADA python -m coder.scripts.gen_remask \
      --refiner llada \
      --input "${OUT}/qwen_mbpp.jsonl" \
      --out "${OUT}/qwen_llada_remask_mbpp_t0.9.jsonl" \
      --confidence_threshold 0.9 \
      --device cuda:0
    if [[ $DRY_RUN -eq 0 ]]; then
      sanitize_and_eval mbpp \
        "${OUT}/qwen_llada_remask_mbpp_t0.9.jsonl" \
        "${OUT}/qwen_llada_remask_mbpp_t0.9_summary.json"
    fi

    echo "[DONE] Group D"
  } 2>&1 | tee "$LOG"
}

# ---------------------------------------------------------------------------
# Group E — DeepSeek Dream remask timing only（很慢）
# ---------------------------------------------------------------------------
run_group_E() {
  local LOG="${OUT}/table4_groupE.log"
  {
    gen_step "deepseek+dream remask HE (timing only)"
    _run $GPU_DREAM python -m coder.scripts.gen_remask \
      --refiner dream \
      --input "${OUT}/deepseek_humaneval.jsonl" \
      --out "${OUT}/deepseek_dream_remask_humaneval_t0.9_timed.jsonl" \
      --confidence_threshold 0.9 \
      --device cuda:0

    gen_step "deepseek+dream remask MBPP (timing only)"
    _run $GPU_DREAM python -m coder.scripts.gen_remask \
      --refiner dream \
      --input "${OUT}/deepseek_mbpp.jsonl" \
      --out "${OUT}/deepseek_dream_remask_mbpp_t0.9_timed.jsonl" \
      --confidence_threshold 0.9 \
      --device cuda:0

    echo "[DONE] Group E"
  } 2>&1 | tee "$LOG"
}

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
echo "Running groups: ${RUN_GROUPS}"
echo "GPU_DS=${GPU_DS}  GPU_QWEN=${GPU_QWEN}  GPU_LLADA=${GPU_LLADA}  GPU_LOC=${GPU_LOC}  GPU_DREAM=${GPU_DREAM}"
echo ""

PIDS=()

[[ "$RUN_GROUPS" == *A* ]] && { run_group_A & PIDS+=($!); echo "[launched] Group A → pid ${PIDS[-1]}  log: ${OUT}/table4_groupA.log"; }
[[ "$RUN_GROUPS" == *B* ]] && { run_group_B & PIDS+=($!); echo "[launched] Group B → pid ${PIDS[-1]}  log: ${OUT}/table4_groupB.log"; }
# Group C sequentially after B (needs Qwen model); launch separately when B done or GPU available
[[ "$RUN_GROUPS" == *C* ]] && { run_group_C & PIDS+=($!); echo "[launched] Group C → pid ${PIDS[-1]}  log: ${OUT}/table4_groupC.log"; }
[[ "$RUN_GROUPS" == *D* ]] && { run_group_D & PIDS+=($!); echo "[launched] Group D → pid ${PIDS[-1]}  log: ${OUT}/table4_groupD.log"; }
[[ "$RUN_GROUPS" == *E* ]] && { run_group_E & PIDS+=($!); echo "[launched] Group E → pid ${PIDS[-1]}  log: ${OUT}/table4_groupE.log"; }

if [[ $DRY_RUN -eq 0 ]] && [[ ${#PIDS[@]} -gt 0 ]]; then
  echo ""
  echo "Waiting for all groups to complete..."
  wait "${PIDS[@]}"
  echo ""
  echo "=== All done. Regenerating results.md ==="
  python -m coder.scripts.gen_results_table
fi

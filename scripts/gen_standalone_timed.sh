#!/usr/bin/env bash
# Re-run all standalone models with timing instrumentation.
# Outputs go to outputs/base_tuteng/{model}_{dataset}_timed.jsonl
# + auto-generated *.timing_summary.json files.
#
# GPU assignments (based on free memory as of 2026-04-10):
#   GPU 0 (~25GB): DeepSeek, Mistral
#   GPU 2 (~23GB): Qwen, StarCoder2
#   GPU 4 (~28GB): Seed-Coder, Seed-DiffCoder
#   GPU 5 (~26GB): Llama31, LLaDA
#   GPU 7 (~36GB): Dream (slow: ~163s/sample)
#
# Usage:
#   bash scripts/gen_standalone_timed.sh [--dry-run]
#
# Each group runs sequentially within its GPU (nohup background).
# Logs: outputs/base_tuteng/timed_{group}.log

set -euo pipefail
ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

# Activate conda environment with evalplus + coder deps
source /home/tteng/miniconda3/etc/profile.d/conda.sh
conda activate code

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  echo "[dry-run] will print commands only"
fi

OUT="outputs/base_tuteng"
mkdir -p "$OUT"

run_gen() {
  local model="$1"
  local dataset="$2"
  local phys_gpu="$3"   # physical GPU index (e.g. 0, 2, 7)
  local extra="${4:-}"
  local out_file="${OUT}/${model}_${dataset}_timed.jsonl"
  # With CUDA_VISIBLE_DEVICES=N, the process always sees a single GPU as cuda:0
  local cmd="python -m coder.scripts.gen_evalplus \
    --model ${model} \
    --dataset ${dataset} \
    --device cuda:0 \
    --out ${out_file} \
    ${extra}"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[cmd] CUDA_VISIBLE_DEVICES=${phys_gpu} $cmd"
  else
    echo "[START] ${model} ${dataset} on GPU ${phys_gpu}"
    CUDA_VISIBLE_DEVICES="${phys_gpu}" eval $cmd
    echo "[DONE]  ${model} ${dataset}"
  fi
}

# ---------------------------------------------------------------------------
# GPU 0: DeepSeek + Mistral
# ---------------------------------------------------------------------------
(
  LOGFILE="${OUT}/timed_gpu0.log"
  {
    run_gen deepseek humaneval 0
    run_gen deepseek mbpp     0
    run_gen mistral  humaneval 0
    run_gen mistral  mbpp     0
  } 2>&1 | tee "$LOGFILE"
) &
PID_GPU0=$!
echo "[launched] GPU0 group → pid $PID_GPU0  log: ${OUT}/timed_gpu0.log"

# ---------------------------------------------------------------------------
# GPU 2: Qwen + StarCoder2
# ---------------------------------------------------------------------------
(
  LOGFILE="${OUT}/timed_gpu2.log"
  {
    run_gen qwen       humaneval 2
    run_gen qwen       mbpp      2
    run_gen starcoder2 humaneval 2
    run_gen starcoder2 mbpp      2
  } 2>&1 | tee "$LOGFILE"
) &
PID_GPU2=$!
echo "[launched] GPU2 group → pid $PID_GPU2  log: ${OUT}/timed_gpu2.log"

# ---------------------------------------------------------------------------
# GPU 4: Seed-Coder + Seed-DiffCoder
# ---------------------------------------------------------------------------
(
  LOGFILE="${OUT}/timed_gpu4.log"
  {
    run_gen seed-coder     humaneval 4 "--model_id ByteDance-Seed/Seed-Coder-8B-Base"
    run_gen seed-coder     mbpp      4 "--model_id ByteDance-Seed/Seed-Coder-8B-Base"
    run_gen seed-diffcoder humaneval 4 "--model_id ByteDance-Seed/Stable-DiffCoder-8B-Instruct"
    run_gen seed-diffcoder mbpp      4 "--model_id ByteDance-Seed/Stable-DiffCoder-8B-Instruct"
  } 2>&1 | tee "$LOGFILE"
) &
PID_GPU4=$!
echo "[launched] GPU4 group → pid $PID_GPU4  log: ${OUT}/timed_gpu4.log"

# ---------------------------------------------------------------------------
# GPU 5: Llama31 + LLaDA
# ---------------------------------------------------------------------------
(
  LOGFILE="${OUT}/timed_gpu5.log"
  {
    run_gen llama31 humaneval 5
    run_gen llama31 mbpp      5
    run_gen llada   humaneval 5
    run_gen llada   mbpp      5
  } 2>&1 | tee "$LOGFILE"
) &
PID_GPU5=$!
echo "[launched] GPU5 group → pid $PID_GPU5  log: ${OUT}/timed_gpu5.log"

# ---------------------------------------------------------------------------
# GPU 7: Dream (slow — ~163s/sample × 164 HE + 378 MBPP ≈ 25h)
# ---------------------------------------------------------------------------
(
  LOGFILE="${OUT}/timed_gpu7.log"
  {
    run_gen dream humaneval 7
    run_gen dream mbpp      7
  } 2>&1 | tee "$LOGFILE"
) &
PID_GPU7=$!
echo "[launched] GPU7 group → pid $PID_GPU7  log: ${OUT}/timed_gpu7.log"

if [[ $DRY_RUN -eq 0 ]]; then
  echo ""
  echo "All groups launched. Waiting for all to complete..."
  wait $PID_GPU0 $PID_GPU2 $PID_GPU4 $PID_GPU5 $PID_GPU7
  echo ""
  echo "=== All done. Regenerating results.md ==="
  python -m coder.scripts.gen_results_table
fi

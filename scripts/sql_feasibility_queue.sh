#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH=src
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

PY="${PY:-/home/wjzhang/miniforge3/envs/cocoder/bin/python}"
SQL_OUT="${SQL_OUT:-outputs/sql_feasibility}"
SPIDER_DIR="${SPIDER_DIR:-${SQL_OUT}/spider}"
N_SAMPLES="${N_SAMPLES:-200}"
MAX_LOCATOR_SAMPLES="${MAX_LOCATOR_SAMPLES:-100}"
RUN_REMASK="${RUN_REMASK:-1}"
RUN_LLAMA31="${RUN_LLAMA31:-0}"

mkdir -p "${SQL_OUT}/queue_logs"

ratio_ge_3() {
  local path="$1"
  "${PY}" - "$path" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    ratio = json.load(open(path, encoding="utf-8")).get("fault_detection_ratio")
except FileNotFoundError:
    ratio = None
if ratio is not None and float(ratio) >= 3.0:
    raise SystemExit(0)
raise SystemExit(1)
PY
}

run_remask_if_passed() {
  local model="$1"
  local refiner="$2"
  local gpu="$3"
  local ratio_summary="$4"

  if [[ "${RUN_REMASK}" != "1" ]]; then
    echo "[skip] ${model}/${refiner}: RUN_REMASK=${RUN_REMASK}"
    return 0
  fi
  if ! ratio_ge_3 "${ratio_summary}"; then
    echo "[skip] ${model}/${refiner}: locator ratio < 3.0"
    return 0
  fi

  local out="${SQL_OUT}/${model}_${refiner}_remask_spider_dev.jsonl"
  local eval_out="${SQL_OUT}/${model}_${refiner}_remask_spider_dev_eval.jsonl"
  local summary_out="${SQL_OUT}/${model}_${refiner}_remask_spider_dev_eval_summary.json"

  echo "[remask] ${model}/${refiner} on gpu ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -m coder.scripts.gen_remask \
    --locator dream \
    --refiner "${refiner}" \
    --input "${SQL_OUT}/${model}_spider_dev.jsonl" \
    --out "${out}" \
    --confidence_threshold 0.9 \
    --device cuda:0 \
    --resume \
    --record_mask_stats

  "${PY}" -m coder.scripts.sql_eval \
    --pred "${out}" \
    --spider_dir "${SPIDER_DIR}" \
    --out "${eval_out}" \
    --summary_out "${summary_out}"
}

run_lane() {
  local model="$1"
  local gpu="$2"
  local ar_model_id="$3"

  export CUDA_VISIBLE_DEVICES="${gpu}"
  local pred="${SQL_OUT}/${model}_spider_dev.jsonl"
  local eval_out="${SQL_OUT}/${model}_spider_dev_eval.jsonl"
  local eval_summary="${SQL_OUT}/${model}_spider_dev_eval_summary.json"

  echo "===== ${model}: start on gpu ${gpu} ====="
  date -Is

  "${PY}" -m coder.scripts.gen_sql_ar \
    --spider_dir "${SPIDER_DIR}" \
    --out "${pred}" \
    --model "${model}" \
    --n_samples "${N_SAMPLES}" \
    --device cuda:0 \
    --resume

  "${PY}" -m coder.scripts.sql_eval \
    --pred "${pred}" \
    --spider_dir "${SPIDER_DIR}" \
    --out "${eval_out}" \
    --summary_out "${eval_summary}"

  for locator in dream llada; do
    local loc_summary="${SQL_OUT}/${model}_sql_locator_analysis_${locator}_summary.json"
    echo "[locator] ${model}/${locator}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -m coder.analysis.sql_locator_analysis \
      --eval_jsonl "${eval_out}" \
      --locator "${locator}" \
      --device cuda:0 \
      --max_samples "${MAX_LOCATOR_SAMPLES}" \
      --summary_out "${loc_summary}" \
      > "${SQL_OUT}/${model}_sql_locator_analysis_${locator}.txt"
  done

  echo "[locator] ${model}/ar"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -m coder.analysis.sql_locator_analysis \
    --eval_jsonl "${eval_out}" \
    --locator ar \
    --locator_model_id "${ar_model_id}" \
    --device cuda:0 \
    --max_samples "${MAX_LOCATOR_SAMPLES}" \
    --summary_out "${SQL_OUT}/${model}_sql_locator_analysis_ar_summary.json" \
    > "${SQL_OUT}/${model}_sql_locator_analysis_ar.txt"

  run_remask_if_passed "${model}" dream "${gpu}" "${SQL_OUT}/${model}_sql_locator_analysis_dream_summary.json"
  run_remask_if_passed "${model}" llada "${gpu}" "${SQL_OUT}/${model}_sql_locator_analysis_llada_summary.json"

  echo "===== ${model}: done ====="
  date -Is
}

run_lane qwen 3 "Qwen/Qwen2.5-Coder-7B-Instruct" \
  > "${SQL_OUT}/queue_logs/qwen.log" 2>&1 &
if [[ "${RUN_LLAMA31}" == "1" ]]; then
  run_lane llama31 4 "meta-llama/Llama-3.1-8B-Instruct" \
    > "${SQL_OUT}/queue_logs/llama31.log" 2>&1 &
else
  echo "[skip] llama31: RUN_LLAMA31=${RUN_LLAMA31}; cache missing in current environment" \
    > "${SQL_OUT}/queue_logs/llama31.log"
fi
run_lane starcoder2 7 "bigcode/starcoder2-7b" \
  > "${SQL_OUT}/queue_logs/starcoder2.log" 2>&1 &
run_lane mistral 4 "mistralai/Mistral-7B-Instruct-v0.3" \
  > "${SQL_OUT}/queue_logs/mistral.log" 2>&1 &
run_lane codellama 1 "codellama/CodeLlama-7b-Instruct-hf" \
  > "${SQL_OUT}/queue_logs/codellama.log" 2>&1 &

wait

"${PY}" -m coder.scripts.gen_results_table
echo "[done] sql feasibility queue complete"

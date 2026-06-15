#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH=src
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

PY="${PY:-/home/wjzhang/miniforge3/envs/cocoder/bin/python}"
SQL_OUT="${SQL_OUT:-outputs/sql_feasibility}"
SPIDER_DIR="${SPIDER_DIR:-${SQL_OUT}/spider}"
GPU="${GPU:-3}"
MAX_LOCATOR_SAMPLES="${MAX_LOCATOR_SAMPLES:-100}"
RUN_REMASK="${RUN_REMASK:-1}"
WAIT_TMUX="${WAIT_TMUX:-sql_more_queue}"
WAIT_SECONDS="${WAIT_SECONDS:-300}"

mkdir -p "${SQL_OUT}/queue_logs"

if [[ -n "${WAIT_TMUX}" ]]; then
  while tmux has-session -t "${WAIT_TMUX}" 2>/dev/null; do
    echo "[wait] ${WAIT_TMUX} still running; sleeping ${WAIT_SECONDS}s"
    sleep "${WAIT_SECONDS}"
  done
fi

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

run_llada_remask_if_passed() {
  local model="$1"
  local ratio_summary="$2"

  if [[ "${RUN_REMASK}" != "1" ]]; then
    echo "[skip] ${model}/llada: RUN_REMASK=${RUN_REMASK}"
    return 0
  fi
  if ! ratio_ge_3 "${ratio_summary}"; then
    echo "[skip] ${model}/llada: locator ratio < 3.0"
    return 0
  fi

  local out="${SQL_OUT}/${model}_llada_remask_spider_dev.jsonl"
  local eval_out="${SQL_OUT}/${model}_llada_remask_spider_dev_eval.jsonl"
  local summary_out="${SQL_OUT}/${model}_llada_remask_spider_dev_eval_summary.json"

  echo "[remask] ${model}/llada on gpu ${GPU}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -m coder.scripts.gen_remask \
    --locator dream \
    --refiner llada \
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

run_llada_locator() {
  local model="$1"
  local eval_out="$2"
  local text_out="$3"
  local summary_out="$4"

  if [[ -s "${summary_out}" ]]; then
    echo "[skip] ${model}/llada locator exists: ${summary_out}"
  else
    echo "[locator] ${model}/llada on gpu ${GPU}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -m coder.analysis.sql_locator_analysis \
      --eval_jsonl "${eval_out}" \
      --locator llada \
      --device cuda:0 \
      --max_samples "${MAX_LOCATOR_SAMPLES}" \
      --summary_out "${summary_out}" \
      > "${text_out}"
  fi

  run_llada_remask_if_passed "${model}" "${summary_out}"
}

date -Is
run_llada_locator \
  deepseek \
  "${SQL_OUT}/deepseek_spider_dev_eval.jsonl" \
  "${SQL_OUT}/sql_locator_analysis_llada.txt" \
  "${SQL_OUT}/sql_locator_analysis_llada_summary.json"

run_llada_locator \
  seed-coder \
  "${SQL_OUT}/seed-coder_spider_dev_eval.jsonl" \
  "${SQL_OUT}/seed-coder_sql_locator_analysis_llada.txt" \
  "${SQL_OUT}/seed-coder_sql_locator_analysis_llada_summary.json"

"${PY}" -m coder.scripts.gen_results_table
date -Is
echo "[done] SQL LLaDA catch-up complete"

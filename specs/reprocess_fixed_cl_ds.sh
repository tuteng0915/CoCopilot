#!/bin/bash
# Reprocess CodeLlama + DeepSeek with fixed build_evalplus_solution (no GPU needed).

set -e
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cocoder

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
cd "${ROOT_DIR}"

LOG="outputs/tau_rerun/reprocess_fixed_cl_ds.log"
exec > >(tee -a "${LOG}") 2>&1
echo "=== reprocess_fixed_cl_ds start: $(date) ==="

process_one() {
    local dataset="$1"
    local jsonl="$2"
    local base="${jsonl%.jsonl}_fixed"

    echo "--- Processing: $(basename ${jsonl}) ---"

    python -m coder.scripts.normalize_evalplus_packaging \
        --input  "${jsonl}" \
        --out    "${base}.jsonl"

    python -m coder.scripts.postprocess_evalplus \
        --dataset "${dataset}" \
        --samples "${base}.jsonl"

    python -m coder.scripts.eval_evalplus \
        --backend local \
        --dataset "${dataset}" \
        --samples "${base}-sanitized.jsonl" \
        --summary_out "${base}_summary.json"
}

# ── CodeLlama HumanEval ───────────────────────────────────────────────────────
for t in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.9; do  # 0.8 missing
    f="outputs/tau_rerun/codellama_remask_humaneval_t${t}.jsonl"
    [ -f "${f}" ] && process_one humaneval "${f}"
done

# ── CodeLlama MBPP ────────────────────────────────────────────────────────────
for t in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    f="outputs/tau_rerun/codellama_remask_mbpp_t${t}.jsonl"
    [ -f "${f}" ] && process_one mbpp "${f}"
done

# ── DeepSeek HumanEval ────────────────────────────────────────────────────────
for t in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.93 0.95 0.97 0.99; do
    f="outputs/tau_rerun/remask_humaneval_t${t}.jsonl"
    [ -f "${f}" ] && process_one humaneval "${f}"
done

# ── DeepSeek MBPP ─────────────────────────────────────────────────────────────
for t in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.93 0.95 0.97 0.99; do
    f="outputs/tau_rerun/remask_mbpp_t${t}.jsonl"
    [ -f "${f}" ] && process_one mbpp "${f}"
done

echo ""
echo "=== reprocess_fixed_cl_ds DONE: $(date) ==="

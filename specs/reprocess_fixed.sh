#!/bin/bash
# Reprocess existing jsonl files with fixed build_evalplus_solution (no GPU needed).
# Writes output with _fixed suffix to avoid overwriting old results.

set -e
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cocoder

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
cd "${ROOT_DIR}"

LOG="outputs/tau_rerun/reprocess_fixed.log"
exec > >(tee -a "${LOG}") 2>&1
echo "=== reprocess_fixed start: $(date) ==="

process_one() {
    local dataset="$1"   # humaneval or mbpp
    local jsonl="$2"     # path to existing remask output
    local base="${jsonl%.jsonl}_fixed"

    echo "--- Processing: $(basename ${jsonl}) ---"

    # Step 1: re-apply build_evalplus_solution
    python -m coder.scripts.normalize_evalplus_packaging \
        --input  "${jsonl}" \
        --out    "${base}.jsonl"

    # Step 2: syncheck + sanitize
    python -m coder.scripts.postprocess_evalplus \
        --dataset "${dataset}" \
        --samples "${base}.jsonl"

    # Step 3: evaluate
    python -m coder.scripts.eval_evalplus \
        --backend local \
        --dataset "${dataset}" \
        --samples "${base}-sanitized.jsonl" \
        --summary_out "${base}_summary.json"
}

TAUS="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"

# ── Qwen HumanEval ────────────────────────────────────────────────────────────
for t in $TAUS; do
    process_one humaneval "outputs/tau_rerun/qwen_remask_humaneval_t${t}.jsonl"
done

# ── Qwen MBPP ────────────────────────────────────────────────────────────────
for t in $TAUS; do
    process_one mbpp "outputs/tau_rerun/qwen_remask_mbpp_t${t}.jsonl"
done

# ── Llama-3.1 HumanEval ──────────────────────────────────────────────────────
for t in $TAUS; do
    process_one humaneval "outputs/tau_rerun/llama31_remask_humaneval_t${t}.jsonl"
done

# ── Llama-3.1 MBPP ───────────────────────────────────────────────────────────
for t in $TAUS; do
    process_one mbpp "outputs/tau_rerun/llama31_remask_mbpp_t${t}.jsonl"
done

echo ""
echo "=== reprocess_fixed DONE: $(date) ==="

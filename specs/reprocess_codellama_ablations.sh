#!/bin/bash
# Normalize CodeLlama ablation methods to fixed build_evalplus_solution pipeline.
# No GPU needed — pure CPU postprocessing.
set -e
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cocoder
cd /home/wjzhang/tt_workspace/model/CoCoder/CoCoder
export PYTHONPATH=src

LOG="outputs/tau_rerun/codellama_ablations_normalized.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== codellama_ablations_normalized start: $(date) ==="

process_one() {
    local dataset="$1"
    local src="$2"
    local out_base="outputs/tau_rerun/$(basename ${src%.jsonl})_normalized"

    echo "--- $dataset: $(basename $src) ---"
    python -m coder.scripts.normalize_evalplus_packaging \
        --input  "$src" \
        --out    "${out_base}.jsonl"

    python -m coder.scripts.postprocess_evalplus \
        --dataset "$dataset" \
        --samples "${out_base}.jsonl"

    python -m coder.scripts.eval_evalplus \
        --backend local \
        --dataset "$dataset" \
        --samples "${out_base}-sanitized.jsonl" \
        --summary_out "${out_base}_summary.json"
}

# HumanEval
for method in selfrefine_r1 reflexion_feedback_r1 rerank_logprob_k8 locate_ar_rewrite_t0.9; do
    process_one humaneval "outputs/base_tuteng/codellama_humaneval_${method}.jsonl"
done

# MBPP
for method in selfrefine_r1 reflexion_feedback_r1 rerank_logprob_k8 locate_ar_rewrite_t0.9; do
    process_one mbpp "outputs/base_tuteng/codellama_mbpp_${method}.jsonl"
done

# Also normalize MBPP baseline for consistency
process_one mbpp "outputs/base_tuteng/codellama_mbpp.jsonl"

echo "=== codellama_ablations_normalized DONE: $(date) ==="

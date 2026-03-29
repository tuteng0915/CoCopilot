ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=4

# Step 1: Generate DeepSeek drafts (saves prompt + raw_completion)
#python -m coder.scripts.gen_evalplus \
  #--model deepseek \
  #--dataset humaneval \
  #--out outputs/deepseek_humaneval.jsonl

#python -m coder.scripts.gen_evalplus \
  #--model deepseek \
  #--dataset mbpp \
  #--out outputs/deepseek_mbpp.jsonl



# Step 2: Refine drafts with DreamCoder remasking

for THRESH in 0.93 0.95 0.97 ; do

python -m coder.scripts.gen_remask \
  --input outputs/deepseek_humaneval.jsonl \
  --out   outputs/remask_humaneval_t${THRESH}.jsonl \
  --confidence_threshold ${THRESH} \
  --temperature 0.0 --top_p 1.0 --seed 3407

python -m coder.scripts.gen_remask \
  --input outputs/deepseek_mbpp.jsonl \
  --out   outputs/remask_mbpp_t${THRESH}.jsonl \
  --confidence_threshold ${THRESH} \
  --temperature 0.0 --top_p 1.0 --seed 3407

done

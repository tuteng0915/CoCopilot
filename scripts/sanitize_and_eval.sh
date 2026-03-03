export PYTHONPATH=/home/kodai/CoCopilot/src
export CUDA_VISIBLE_DEVICES=4

# Step 1: Sanitize the generated outputs

## Sanitize DeepSeek outputs
#python scripts/postprocess_evalplus.py \
  #--dataset humaneval \
  #--samples outputs/deepseek_humaneval.jsonl

#python scripts/postprocess_evalplus.py \
  #--dataset mbpp \
  #--samples outputs/deepseek_mbpp.jsonl

## Sanitize remasked outputs (thresholds 0.7, 0.8, 0.9)
#for THRESH in 0.7 0.8 0.9; do
  #python scripts/postprocess_evalplus.py \
    #--dataset humaneval \
    #--samples outputs/remask_humaneval_t${THRESH}.jsonl

  #python scripts/postprocess_evalplus.py \
    #--dataset mbpp \
    #--samples outputs/remask_mbpp_t${THRESH}.jsonl
#done

# Step 2: Evaluate all sanitized outputs

# Evaluate DeepSeek sanitized outputs
python scripts/eval_evalplus.py \
  --backend local \
  --dataset humaneval \
  --samples outputs/deepseek_humaneval-sanitized.jsonl

python scripts/eval_evalplus.py \
  --backend local \
  --dataset mbpp \
  --samples outputs/deepseek_mbpp-sanitized.jsonl

# Evaluate remasked sanitized outputs (thresholds 0.7, 0.8, 0.9)
for THRESH in 0.7 0.8 0.9 ; do
  python scripts/eval_evalplus.py \
    --backend local \
    --dataset humaneval \
    --samples outputs/remask_humaneval_t${THRESH}-sanitized.jsonl \
    --summary_out outputs/remask_humaneval_t${THRESH}_summary.json

  python scripts/eval_evalplus.py \
    --backend local \
    --dataset mbpp \
    --samples outputs/remask_mbpp_t${THRESH}-sanitized.jsonl \
    --summary_out outputs/remask_mbpp_t${THRESH}_summary.json
done



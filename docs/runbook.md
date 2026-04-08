## 目标

把本项目跑 benchmark 时的**环境、命令模板、产物位置/命名**写死，减少“记错/跑错/被跳过”的失败率。

## 环境约定

- **`code` conda 环境**：EvalPlus（HumanEval/MBPP）+ LiveCodeBench/LiveBench-Coding 的默认环境
- **`bcb_iso` conda 环境**：BigCodeBench 专用（避免依赖污染）

通用环境变量（建议每次跑前都设）：

- `export PYTHONPATH="/model/tteng/CoCoder/src:${PYTHONPATH:-}"`
- BigCodeBench 跑模型时常加：
  - `export TRANSFORMERS_NO_TF=1`
  - `export TRANSFORMERS_NO_FLAX=1`

## 产物目录

统一在：

- `outputs/base_tuteng/`

## 数学任务（GSM8K / MATH-500）

为验证方案的泛化能力，数学任务与代码任务共用同一套 `gen_math` / `eval_math` 脚本，入口风格与代码脚本对齐（`--samples`、`--out_summary`、`--resume`、`--num_shards` 等参数均相同）。

典型 pipeline（以 `MODEL=dream`、`DATASET=gsm8k` 为例）：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate code
cd /model/tteng/CoCoder
export PYTHONPATH="/model/tteng/CoCoder/src:${PYTHONPATH:-}"

# 生成
python -m coder.scripts.gen_math \
  --model "$MODEL" --dataset "$DATASET" \
  --out "outputs/base_tuteng/${MODEL}_${DATASET}.jsonl"

# 评测（内置 answer 提取，无需额外 sanitize 步骤）
python -m coder.scripts.eval_math \
  --samples "outputs/base_tuteng/${MODEL}_${DATASET}.jsonl" \
  --out_summary "outputs/base_tuteng/${MODEL}_${DATASET}_summary.json"
```

MATH-500 可以加 `--per_subject --per_level` 按学科和难度细分：

```bash
python -m coder.scripts.eval_math \
  --samples "outputs/base_tuteng/${MODEL}_math500.jsonl" \
  --out_summary "outputs/base_tuteng/${MODEL}_math500_summary.json" \
  --per_subject --per_level
```

常见产物：

- `outputs/base_tuteng/${MODEL}_gsm8k.jsonl`
- `outputs/base_tuteng/${MODEL}_gsm8k.jsonl.timing_summary.json`
- `outputs/base_tuteng/${MODEL}_gsm8k_summary.json`
- `outputs/base_tuteng/${MODEL}_math500...` 同理

断点续跑（任务中断后接着跑）：

```bash
python -m coder.scripts.gen_math --model "$MODEL" --dataset "$DATASET" \
  --out "outputs/base_tuteng/${MODEL}_${DATASET}.jsonl" --resume
```

多卡分片并行：

```bash
# GPU 0 跑前一半，GPU 1 跑后一半
CUDA_VISIBLE_DEVICES=0 python -m coder.scripts.gen_math ... --num_shards 2 --shard_idx 0 &
CUDA_VISIBLE_DEVICES=1 python -m coder.scripts.gen_math ... --num_shards 2 --shard_idx 1 &
```

## EvalPlus（HumanEval / MBPP）

典型 pipeline（以 `MODEL=deepseek`、`DATASET=humaneval` 为例）：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate code
cd /model/tteng/CoCoder
export PYTHONPATH="/model/tteng/CoCoder/src:${PYTHONPATH:-}"

python -m coder.scripts.gen_evalplus --model "$MODEL" --dataset "$DATASET" --out "outputs/base_tuteng/${MODEL}_${DATASET}.jsonl"
python -m coder.scripts.postprocess_evalplus --dataset "$DATASET" --samples "outputs/base_tuteng/${MODEL}_${DATASET}.jsonl"
python -m coder.scripts.eval_evalplus --backend local --dataset "$DATASET" --samples "outputs/base_tuteng/${MODEL}_${DATASET}-sanitized.jsonl"
```

常见产物：

- `outputs/base_tuteng/${MODEL}_humaneval.jsonl`
- `outputs/base_tuteng/${MODEL}_humaneval-sanitized.jsonl`
- `outputs/base_tuteng/${MODEL}_humaneval-sanitized_eval_results.json`
- `outputs/base_tuteng/${MODEL}_humaneval_summary.json`
- `outputs/base_tuteng/${MODEL}_mbpp...` 同理

### EvalPlus 失败反馈 -> Reflexion（真实反馈版本）

以 `MODEL=deepseek`、`DATASET=humaneval` 为例：

```bash
# 1) 先有 baseline samples + sanitize + eval（见上面的 EvalPlus 典型 pipeline）

# 2) 从 evalplus 评测结果提取每题失败摘要
python -m coder.analysis.evalplus_feedback \
  --eval_results "outputs/base_tuteng/${MODEL}_${DATASET}-sanitized_eval_results.json" \
  --out_feedback "outputs/base_tuteng/${MODEL}_${DATASET}.evalplus_feedback.jsonl"

# 3) 将 feedback 挂到 Reflexion prompt 的 [Feedback] 段，生成修订版 samples
python -m coder.scripts.gen_reflexion \
  --input "outputs/base_tuteng/${MODEL}_${DATASET}-sanitized.jsonl" \
  --feedback_file "outputs/base_tuteng/${MODEL}_${DATASET}.evalplus_feedback.jsonl" \
  --feedback_field failure_summary \
  --model "$MODEL" \
  --rounds 1 \
  --out "outputs/base_tuteng/${MODEL}_${DATASET}_reflexion_feedback.jsonl"

# 4) 对修订版重复 sanitize + eval
python -m coder.scripts.postprocess_evalplus \
  --dataset "$DATASET" \
  --samples "outputs/base_tuteng/${MODEL}_${DATASET}_reflexion_feedback.jsonl"
python -m coder.scripts.eval_evalplus \
  --backend local \
  --dataset "$DATASET" \
  --samples "outputs/base_tuteng/${MODEL}_${DATASET}_reflexion_feedback-sanitized.jsonl"
```

建议命名约定：

- 反馈文件：`outputs/base_tuteng/${MODEL}_${DATASET}.evalplus_feedback.jsonl`
- Reflexion 修订输出：`outputs/base_tuteng/${MODEL}_${DATASET}_reflexion_feedback.jsonl`
- timing：`<reflexion_out>.timing_summary.json`

### Reranking（AR logprob 打分）

```bash
python -m coder.scripts.gen_rerank \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --num_samples 8 \
  --score_mode logprob \
  --logprob_norm avg \
  --out "outputs/base_tuteng/${MODEL}_${DATASET}_rerank_logprob_k8.jsonl"
```

后续流程同样是 `postprocess_evalplus -> eval_evalplus`。

## LiveCodeBench / LiveBench-Coding

注意：脚本名是 `gen_livebench.py / eval_livebench.py`，但通过参数区分 benchmark。

生成（livecodebench）：

```bash
python -m coder.scripts.gen_livebench --benchmark livecodebench --model "$MODEL" --out "outputs/base_tuteng/${MODEL}_livecodebench.jsonl"
```

评测（livecodebench）：

```bash
python -m coder.scripts.eval_livebench --benchmark livecodebench \
  --samples "outputs/base_tuteng/${MODEL}_livecodebench.jsonl" \
  --out_judgments "outputs/base_tuteng/${MODEL}_livecodebench_judgments.jsonl" \
  --out_summary "outputs/base_tuteng/${MODEL}_livecodebench_summary.json"
```

## BigCodeBench

推荐用 `bcb_iso`：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bcb_iso
cd /model/tteng/CoCoder
export PYTHONPATH="/model/tteng/CoCoder/src:${PYTHONPATH:-}"
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1
export CUDA_VISIBLE_DEVICES=3
```

生成：

```bash
python -m coder.scripts.gen_bigcodebench --model qwen --split instruct --subset full --out outputs/base_tuteng/qwen_bigcodebench_instruct_full.jsonl
```

评测（命令行工具 + 包装 summary）：

```bash
bigcodebench.evaluate instruct full --execution local \
  --samples "/model/tteng/CoCoder/outputs/base_tuteng/qwen_bigcodebench_instruct_full.jsonl" \
  --pass_k 1,5,10 --save_pass_rate

python -m coder.scripts.eval_bigcodebench --samples outputs/base_tuteng/qwen_bigcodebench_instruct_full.jsonl \
  --split instruct --subset full --execution local \
  --out_summary outputs/base_tuteng/qwen_bigcodebench_instruct_full_summary.json
```


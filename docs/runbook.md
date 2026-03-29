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


# Spec: Seed-Coder-8B-Instruct 实验

## 目标

测试 `ByteDance-Seed/Seed-Coder-8B-Instruct` 作为新的 AR drafter，验证 CoCoder 是否对 DeepSeek-Coder 以外的 code-specialized 模型同样有效。

**背景**：当前 Table 3 中，Llama-3.1 8B 虽然 HE+ baseline（57.9%）与 DeepSeek 相近，但 CoCoder 对其无效（-0.6pp）。原因是 Llama-3.1 是通用模型，coding 错误更深层；而 DeepSeek-Coder 是 code-specialized 模型，错误集中在边界条件等局部可修正区域。Seed-Coder-8B-Instruct 同属 code-specialized instruct 模型，是最合适的验证对象。

**模型已下载**：`ByteDance-Seed/Seed-Coder-8B-Instruct` 在本地 HF cache 中。

---

## 前提条件

- conda env: `code`
- 项目根目录: `/model/tteng/CoCoder`
- PYTHONPATH 需设为 `src`

```bash
source /home/tteng/miniconda3/etc/profile.d/conda.sh
conda activate code
cd /model/tteng/CoCoder
export PYTHONPATH=src
```

---

## Phase 1：生成 AR 草稿 + Standalone 评测

### Step 1a: 生成 HumanEval 草稿

```bash
CUDA_VISIBLE_DEVICES=<空闲GPU> python -m coder.scripts.gen_evalplus \
  --model seed-coder \
  --model_id ByteDance-Seed/Seed-Coder-8B-Instruct \
  --out outputs/base_tuteng/seed-coder-instruct_humaneval.jsonl \
  --dataset humaneval \
  --device cuda:0
```

产物：
- `outputs/base_tuteng/seed-coder-instruct_humaneval.jsonl`（164 条）
- `outputs/base_tuteng/seed-coder-instruct_humaneval.jsonl.timing_summary.json`

### Step 1b: 生成 MBPP 草稿

```bash
CUDA_VISIBLE_DEVICES=<空闲GPU> python -m coder.scripts.gen_evalplus \
  --model seed-coder \
  --model_id ByteDance-Seed/Seed-Coder-8B-Instruct \
  --out outputs/base_tuteng/seed-coder-instruct_mbpp.jsonl \
  --dataset mbpp \
  --device cuda:0
```

产物：
- `outputs/base_tuteng/seed-coder-instruct_mbpp.jsonl`（378 条）
- `outputs/base_tuteng/seed-coder-instruct_mbpp.jsonl.timing_summary.json`

### Step 1c: 评测 HumanEval

```bash
python -m coder.scripts.eval_evalplus \
  --samples outputs/base_tuteng/seed-coder-instruct_humaneval.jsonl \
  --dataset humaneval \
  --model seed-coder-instruct
```

产物：
- `outputs/base_tuteng/seed-coder-instruct_humaneval_summary.json`

### Step 1d: 评测 MBPP

```bash
python -m coder.scripts.eval_evalplus \
  --samples outputs/base_tuteng/seed-coder-instruct_mbpp.jsonl \
  --dataset mbpp \
  --model seed-coder-instruct
```

产物：
- `outputs/base_tuteng/seed-coder-instruct_mbpp_summary.json`

### Step 1e: 验证 baseline（Go/No-Go）

```bash
python3 -c "
import json
he = json.load(open('outputs/base_tuteng/seed-coder-instruct_humaneval_summary.json'))
mb = json.load(open('outputs/base_tuteng/seed-coder-instruct_mbpp_summary.json'))
s = he['summary']
print('HumanEval+ plus%:', round(s['n_plus_pass']/s['n_tasks']*100, 1))
print('HumanEval+ base%:', round(s['n_base_pass']/s['n_tasks']*100, 1))
s = mb['summary']
print('MBPP+ plus%:', round(s['n_plus_pass']/s['n_tasks']*100, 1))
print('MBPP+ base%:', round(s['n_base_pass']/s['n_tasks']*100, 1))
"
```

**Go/No-Go 判断**：
- HE+ plus% 在 50–70% → 继续 Phase 2（CoCoder 有可能帮助）
- HE+ plus% > 70% → 考虑只跑 Dream 配对看效果，也可继续
- HE+ plus% < 50% → 模型偏弱，可能不是 DeepSeek 级别，参见 Fallback

---

## Phase 2：CoCoder 配对实验（remask）

### Step 2a: seed-coder-instruct + Dream HumanEval

```bash
CUDA_VISIBLE_DEVICES=<空闲GPU> python -m coder.scripts.gen_remask \
  --refiner dream \
  --input outputs/base_tuteng/seed-coder-instruct_humaneval.jsonl \
  --out outputs/base_tuteng/seed-coder-instruct_dream_remask_humaneval_t0.9.jsonl \
  --confidence_threshold 0.9 \
  --device cuda:0
```

### Step 2b: seed-coder-instruct + Dream MBPP

```bash
CUDA_VISIBLE_DEVICES=<空闲GPU> python -m coder.scripts.gen_remask \
  --refiner dream \
  --input outputs/base_tuteng/seed-coder-instruct_mbpp.jsonl \
  --out outputs/base_tuteng/seed-coder-instruct_dream_remask_mbpp_t0.9.jsonl \
  --confidence_threshold 0.9 \
  --device cuda:0
```

### Step 2c: seed-coder-instruct + LLaDA HumanEval

```bash
CUDA_VISIBLE_DEVICES=<空闲GPU> python -m coder.scripts.gen_remask \
  --refiner llada \
  --input outputs/base_tuteng/seed-coder-instruct_humaneval.jsonl \
  --out outputs/base_tuteng/seed-coder-instruct_llada_remask_humaneval_t0.9.jsonl \
  --confidence_threshold 0.9 \
  --device cuda:0
```

### Step 2d: seed-coder-instruct + LLaDA MBPP

```bash
CUDA_VISIBLE_DEVICES=<空闲GPU> python -m coder.scripts.gen_remask \
  --refiner llada \
  --input outputs/base_tuteng/seed-coder-instruct_mbpp.jsonl \
  --out outputs/base_tuteng/seed-coder-instruct_llada_remask_mbpp_t0.9.jsonl \
  --confidence_threshold 0.9 \
  --device cuda:0
```

### Step 2e: 评测所有 collab 产物

```bash
for slug in \
  seed-coder-instruct_dream_remask_humaneval_t0.9 \
  seed-coder-instruct_dream_remask_mbpp_t0.9 \
  seed-coder-instruct_llada_remask_humaneval_t0.9 \
  seed-coder-instruct_llada_remask_mbpp_t0.9; do

  dataset=$(echo $slug | grep -oP "(humaneval|mbpp)")
  python -m coder.scripts.eval_evalplus \
    --samples outputs/base_tuteng/${slug}.jsonl \
    --dataset $dataset \
    --model $slug
done
```

---

## Phase 3：注册进 gen_results_table.py 和 model_pairs_evalplus.py

### Step 3a: 在 gen_results_table.py 中添加 Standalone 条目

文件：`src/coder/scripts/gen_results_table.py`

在 `_STANDALONE_ENTRIES` 列表中（约第 147 行），在 `Seed-Coder 8B` 条目后添加：

```python
(
    "Seed-Coder-Instruct 8B",
    "seed-coder-instruct_humaneval_summary.json",
    "seed-coder-instruct_mbpp_summary.json",
    None,  # no LCB
    None,  # no BCB
    "seed-coder-instruct_humaneval.jsonl.timing_summary.json",
    "seed-coder-instruct_mbpp.jsonl.timing_summary.json",
),
```

### Step 3b: 在 `_PAIR_TIMING` 中添加 timing 条目（约第 298 行）

```python
# HumanEval
"seed_coder_instruct_dream_humaneval_t0.9":
    OUTPUTS / "seed-coder-instruct_dream_remask_humaneval_t0.9.jsonl.timing_summary.json",
"seed_coder_instruct_llada_humaneval_t0.9":
    OUTPUTS / "seed-coder-instruct_llada_remask_humaneval_t0.9.jsonl.timing_summary.json",
# MBPP
"seed_coder_instruct_dream_mbpp_t0.9":
    OUTPUTS / "seed-coder-instruct_dream_remask_mbpp_t0.9.jsonl.timing_summary.json",
"seed_coder_instruct_llada_mbpp_t0.9":
    OUTPUTS / "seed-coder-instruct_llada_remask_mbpp_t0.9.jsonl.timing_summary.json",
```

### Step 3c: 在 model_pairs_evalplus.py 中添加 PairConfig 条目

文件：`src/coder/scripts/model_pairs_evalplus.py`

在现有 PAIR_CONFIGS 末尾添加（参考已有 llama31/starcoder2 条目格式）：

```python
# Seed-Coder-Instruct + Dream
PairConfig(
    slug="seed_coder_instruct_dream_humaneval_t0.9",
    dllm_label="Dream-Coder 7B",
    ar_label="Seed-Coder-Instruct 8B",
    dataset="humaneval",
    ar_input="outputs/base_tuteng/seed-coder-instruct_humaneval.jsonl",
    ar_summary="outputs/base_tuteng/seed-coder-instruct_humaneval_summary.json",
    collab_jsonl="outputs/base_tuteng/seed-coder-instruct_dream_remask_humaneval_t0.9.jsonl",
    collab_summary="outputs/base_tuteng/seed-coder-instruct_dream_remask_humaneval_t0.9_summary.json",
),
PairConfig(
    slug="seed_coder_instruct_dream_mbpp_t0.9",
    dllm_label="Dream-Coder 7B",
    ar_label="Seed-Coder-Instruct 8B",
    dataset="mbpp",
    ar_input="outputs/base_tuteng/seed-coder-instruct_mbpp.jsonl",
    ar_summary="outputs/base_tuteng/seed-coder-instruct_mbpp_summary.json",
    collab_jsonl="outputs/base_tuteng/seed-coder-instruct_dream_remask_mbpp_t0.9.jsonl",
    collab_summary="outputs/base_tuteng/seed-coder-instruct_dream_remask_mbpp_t0.9_summary.json",
),
# Seed-Coder-Instruct + LLaDA
PairConfig(
    slug="seed_coder_instruct_llada_humaneval_t0.9",
    dllm_label="LLaDA 8B",
    ar_label="Seed-Coder-Instruct 8B",
    dataset="humaneval",
    ar_input="outputs/base_tuteng/seed-coder-instruct_humaneval.jsonl",
    ar_summary="outputs/base_tuteng/seed-coder-instruct_humaneval_summary.json",
    collab_jsonl="outputs/base_tuteng/seed-coder-instruct_llada_remask_humaneval_t0.9.jsonl",
    collab_summary="outputs/base_tuteng/seed-coder-instruct_llada_remask_humaneval_t0.9_summary.json",
),
PairConfig(
    slug="seed_coder_instruct_llada_mbpp_t0.9",
    dllm_label="LLaDA 8B",
    ar_label="Seed-Coder-Instruct 8B",
    dataset="mbpp",
    ar_input="outputs/base_tuteng/seed-coder-instruct_mbpp.jsonl",
    ar_summary="outputs/base_tuteng/seed-coder-instruct_mbpp_summary.json",
    collab_jsonl="outputs/base_tuteng/seed-coder-instruct_llada_remask_mbpp_t0.9.jsonl",
    collab_summary="outputs/base_tuteng/seed-coder-instruct_llada_remask_mbpp_t0.9_summary.json",
),
```

### Step 3d: 重新生成 model_pairs JSON 和 results.md

```bash
cd /model/tteng/CoCoder
PYTHONPATH=src python -m coder.scripts.model_pairs_evalplus
PYTHONPATH=src python -m coder.scripts.gen_results_table
```

---

## 验收标准

- `seed-coder-instruct_humaneval_summary.json` 中 `n_tasks=164`
- `seed-coder-instruct_mbpp_summary.json` 中 `n_tasks=378`
- Table 3 中出现 `Seed-Coder-Instruct 8B` 行（Dream + LLaDA × HE + MBPP = 4 行）
- Standalone Models 表中出现 `Seed-Coder-Instruct 8B` 行

---

## Fallback：若 Seed-Coder-Instruct 偏强（HE+ > 70%）

如果结果类似 Qwen（CoCoder Δ ≈ 0），可考虑下载以下模型（DeepSeek 同系列，更可能有相似的置信度校准）：

```bash
# 选项 A：Magicoder-S-DS-6.7B（DeepSeek-Coder 骨架 + OSS-Instruct 微调）
# HuggingFace: ise-uiuc/Magicoder-S-DS-6.7B
# 预计 HE+: ~60-65%

# 选项 B：deepseek-coder-1.3b-instruct（同系列，更小）
# HuggingFace: deepseek-ai/deepseek-coder-1.3b-instruct
# 预计 HE+: ~40-48%（偏低）
```

如需下载，在 gen_evalplus.py 中 `seed-coder` 条目已存在，可用 `--model_id` 指定任意 HF model id。

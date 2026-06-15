# Spec: 补全 tab:combined（Table 3）空白数据

> 对应论文 `NeurIPS26-CoCoder/docs/todo.md` 条目 **J**（高优先级，rebuttal 承诺）

## 目标

补全 `section/05_experiments.tex` 中 `tab:combined` 表格里三组 AR 模型的所有 `---` 占位符：

| AR 模型 | 已有 | 缺失 |
|---------|------|------|
| CodeLlama 7B | AR-only、Dream remask HE+/MBPP+、LLaDA remask HE+/MBPP+ | Self-Refine、Reflexion、Reranking、Locate-AR-Rewrite 全行；三组现有行的 HE(base)、MBPP(base)、Overhead |
| Mistral 7B | AR-only、Dream remask HE+/MBPP+、LLaDA remask HE+/MBPP+ | 同上 |
| Seed-Coder-Instruct 8B | AR-only、Dream remask HE+/MBPP+、LLaDA remask HE+/MBPP+ | 同上 |

完成后，在 `section/05_experiments.tex` 的 `tab:combined` 中用真实数字替换所有 `---`，并在 `docs/results.md` 中新增对应 section。

---

## 前提条件

```bash
source /home/tteng/miniconda3/etc/profile.d/conda.sh
conda activate code
cd /model/tteng/CoCoder
export PYTHONPATH=src
export OUTPUTS=outputs/base_tuteng
```

---

## Phase 0：从现有产物提取缺失的 HE(base) / MBPP(base) / Overhead

这些数字不需要重新跑 GPU，直接从已有 sanitized 评测结果和 timing 文件读取。

### Step 0a：提取 HumanEval(base)、HumanEval+、MBPP(base)、MBPP+ 和 Overhead

对以下六个模型 × remask 组合（共 12 行），运行：

```python
import json, os

OUTPUTS = "outputs/base_tuteng"

configs = [
    # (label, he_summary, mbpp_summary, timing_he, timing_mbpp)
    ("codellama + Dream τ=0.9",
     "codellama_dream_remask_humaneval_t0.9_summary.json",
     "codellama_dream_remask_mbpp_t0.9_summary.json",
     "codellama_dream_remask_humaneval_t0.9.jsonl.timing_summary.json",
     "codellama_dream_remask_mbpp_t0.9.jsonl.timing_summary.json"),
    ("codellama + LLaDA τ=0.9",
     "codellama_llada_remask_humaneval_t0.9_summary.json",
     "codellama_llada_remask_mbpp_t0.9_summary.json",
     "codellama_llada_remask_humaneval_t0.9.jsonl.timing_summary.json",
     "codellama_llada_remask_mbpp_t0.9.jsonl.timing_summary.json"),
    ("mistral + Dream τ=0.9",
     "mistral_dream_remask_humaneval_t0.9_summary.json",
     "mistral_dream_remask_mbpp_t0.9_summary.json",
     "mistral_dream_remask_humaneval_t0.9.jsonl.timing_summary.json",
     "mistral_dream_remask_mbpp_t0.9.jsonl.timing_summary.json"),
    ("mistral + LLaDA τ=0.9",
     "mistral_llada_remask_humaneval_t0.9_summary.json",
     "mistral_llada_remask_mbpp_t0.9_summary.json",
     "mistral_llada_remask_humaneval_t0.9.jsonl.timing_summary.json",
     "mistral_llada_remask_mbpp_t0.9.jsonl.timing_summary.json"),
    ("seed-coder-instruct + Dream τ=0.9",
     "seed-coder-instruct_dream_remask_humaneval_t0.9_summary.json",
     "seed-coder-instruct_dream_remask_mbpp_t0.9_summary.json",
     "seed-coder-instruct_dream_remask_humaneval_t0.9.jsonl.timing_summary.json",
     "seed-coder-instruct_dream_remask_mbpp_t0.9.jsonl.timing_summary.json"),
    ("seed-coder-instruct + LLaDA τ=0.9",
     "seed-coder-instruct_llada_remask_humaneval_t0.9_summary.json",
     "seed-coder-instruct_llada_remask_mbpp_t0.9_summary.json",
     "seed-coder-instruct_llada_remask_humaneval_t0.9.jsonl.timing_summary.json",
     "seed-coder-instruct_llada_remask_mbpp_t0.9.jsonl.timing_summary.json"),
]

print(f"{'Label':<40} HE  HE+  MBPP  MBPP+  OHE(s)  OMBPP(s)")
for label, hef, mbpf, the, tmb in configs:
    he  = json.load(open(f"{OUTPUTS}/{hef}"))["summary"]
    mb  = json.load(open(f"{OUTPUTS}/{mbpf}"))["summary"]
    t_he  = json.load(open(f"{OUTPUTS}/{the}"))
    t_mb  = json.load(open(f"{OUTPUTS}/{tmb}"))
    he_base  = round(he["n_base_pass"] / he["n_tasks"] * 100, 1)
    he_plus  = round(he["n_plus_pass"] / he["n_tasks"] * 100, 1)
    mb_base  = round(mb["n_base_pass"] / mb["n_tasks"] * 100, 1)
    mb_plus  = round(mb["n_plus_pass"] / mb["n_tasks"] * 100, 1)
    # timing key: timing.remask_generate_s_avg (gen_remask) or timing.total_s / n
    t_he_s = t_he.get("timing", t_he).get("remask_generate_s_avg") or \
             (t_he.get("timing", {}).get("total_s", 0) / he["n_tasks"])
    t_mb_s = t_mb.get("timing", t_mb).get("remask_generate_s_avg") or \
             (t_mb.get("timing", {}).get("total_s", 0) / mb["n_tasks"])
    print(f"{label:<40} {he_base}  {he_plus}  {mb_base}  {mb_plus}  {t_he_s:.1f}  {t_mb_s:.1f}")
```

**记录结果**：把输出填入下方空表，后续 Phase 4 直接用这些数字更新 LaTeX。

| 行 | HE | HE+ | MBPP | MBPP+ | Overhead HE (s) | Overhead MBPP (s) |
|----|-----|------|------|-------|-----------------|-------------------|
| CodeLlama + Dream | | 34.1 | | 43.4 | | |
| CodeLlama + LLaDA | | 32.3 | | 39.2 | | |
| Mistral + Dream | | 32.3 | | 42.6 | | |
| Mistral + LLaDA | | 31.1 | | 36.2 | | |
| Seed-Coder-Instruct + Dream | | 75.6 | | 72.2 | | |
| Seed-Coder-Instruct + LLaDA | | 72.6 | | 64.6 | | |

> Overhead 列：tab:combined 表头说明 "Overhead (s/sample on HumanEval) is measured **separately from** the initial AR generation"，故直接取 remask 生成耗时（`remask_generate_s_avg`），不含 AR 草稿生成时间。

---

## Phase 1：代码修改（无需 GPU，必须先于 Phase 2）

当前各脚本对三个新模型的支持缺口：

| 脚本 | codellama | mistral | seed-coder(-instruct) |
|------|-----------|---------|----------------------|
| `gen_self_refine.py` | ❌ | ❌ | ❌ |
| `gen_rerank.py` | ❌ | ❌ | ❌ |
| `gen_reflexion.py` | ❌ | ✅ | ✅ (`seed-coder`) |
| `gen_locate_ar_rewrite.py` | ❌ | ✅ | ✅ (`seed-coder`) |

### Step 1a：扩展 `gen_self_refine.py`

文件：`src/coder/scripts/gen_self_refine.py`

**在文件顶部 import 区**（约第 1–10 行）中补充：

```python
from coder.models import CodeLlamaCoder, MistralCoder, SeedCoder
```

**在 `build_model()` 的 `raise ValueError(...)` 前**（约第 47 行）添加：

```python
if name in ["codellama", "codellama_coder"]:
    return CodeLlamaCoder(
        model_id=model_id or "codellama/CodeLlama-7b-Instruct-hf",
        device=device,
    )
if name in ["mistral", "mistral_coder"]:
    return MistralCoder(
        model_id=model_id or "mistralai/Mistral-7B-Instruct-v0.3",
        device=device,
    )
if name in ["seed-coder", "seed_coder", "seedcoder"]:
    return SeedCoder(
        model_id=model_id,
        device=device,
    )
```

> `SeedCoder` 需要 `--model_id`；Seed-Coder-Instruct 用 `--model seed-coder --model_id ByteDance-Seed/Seed-Coder-8B-Instruct`。

### Step 1b：扩展 `gen_rerank.py`

文件：`src/coder/scripts/gen_rerank.py`

**在文件顶部 import 区**（约第 20 行）修改为：

```python
from coder.models import (
    DeepSeekCoder, QwenCoder, Llama31Coder, StarCoder2Coder,
    CodeLlamaCoder, MistralCoder, SeedCoder, CoderModel,
)
```

**在 `build_model()` 的 `raise ValueError(...)` 前**（约第 46 行）添加：

```python
if name in ["codellama", "codellama_coder"]:
    return CodeLlamaCoder(
        model_id=model_id or "codellama/CodeLlama-7b-Instruct-hf",
        device=device,
    )
if name in ["mistral", "mistral_coder"]:
    return MistralCoder(
        model_id=model_id or "mistralai/Mistral-7B-Instruct-v0.3",
        device=device,
    )
if name in ["seed-coder", "seed_coder", "seedcoder"]:
    return SeedCoder(
        model_id=model_id,
        device=device,
    )
```

### Step 1c：扩展 `gen_reflexion.py`（仅 codellama 缺失）

文件：`src/coder/scripts/gen_reflexion.py`

**在文件顶部 import 区**（约第 1–10 行）中补充：

```python
from coder.models import CodeLlamaCoder
```

**在 `build_model()` 的 `raise ValueError(...)` 前**（约第 96 行）添加：

```python
if name in ["codellama", "codellama_coder"]:
    return CodeLlamaCoder(
        model_id=model_id or "codellama/CodeLlama-7b-Instruct-hf",
        device=device,
    )
```

### Step 1d：扩展 `gen_locate_ar_rewrite.py`（仅 codellama 缺失）

文件：`src/coder/scripts/gen_locate_ar_rewrite.py`

**在 `build_model()` 的 `raise ValueError(...)` 前**（约第 78 行）添加：

```python
if name in ["codellama", "codellama_coder"]:
    return CodeLlamaCoder(
        model_id=model_id or "codellama/CodeLlama-7b-Instruct-hf",
        device=device,
    )
```

`CodeLlamaCoder` 已在该文件 import 区中，检查是否需要补充。

---

## Phase 2：CodeLlama 7B 四种 baseline 方法

AR 草稿文件已存在：`outputs/base_tuteng/codellama_humaneval.jsonl`（164条）、`codellama_mbpp.jsonl`（378条）。
Sanitized 文件已存在：`codellama_humaneval-sanitized.jsonl`、`codellama_mbpp-sanitized.jsonl`（如不存在先跑 `postprocess_evalplus`）。

### Step 2.1：Self-Refine

```bash
# HumanEval
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_self_refine \
  --model codellama \
  --input $OUTPUTS/codellama_humaneval.jsonl \
  --out $OUTPUTS/codellama_humaneval_selfrefine_r1.jsonl \
  --device cuda:0 --resume

# MBPP
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_self_refine \
  --model codellama \
  --input $OUTPUTS/codellama_mbpp.jsonl \
  --out $OUTPUTS/codellama_mbpp_selfrefine_r1.jsonl \
  --device cuda:0 --resume
```

评测：

```bash
python -m coder.scripts.eval_evalplus \
  --samples $OUTPUTS/codellama_humaneval_selfrefine_r1.jsonl \
  --dataset humaneval --model codellama_selfrefine_r1

python -m coder.scripts.eval_evalplus \
  --samples $OUTPUTS/codellama_mbpp_selfrefine_r1.jsonl \
  --dataset mbpp --model codellama_selfrefine_r1
```

产物：`codellama_humaneval_selfrefine_r1_summary.json`、`codellama_mbpp_selfrefine_r1_summary.json`

### Step 2.2：Reflexion (w/ feedback)

```bash
# HumanEval（--input 用 sanitized JSONL，其中含 eval.error 字段）
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_reflexion \
  --model codellama \
  --input $OUTPUTS/codellama_humaneval-sanitized.jsonl \
  --raw_input $OUTPUTS/codellama_humaneval.jsonl \
  --out $OUTPUTS/codellama_humaneval_reflexion_feedback_r1.jsonl \
  --feedback_key eval.error \
  --device cuda:0 --resume

# MBPP
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_reflexion \
  --model codellama \
  --input $OUTPUTS/codellama_mbpp-sanitized.jsonl \
  --raw_input $OUTPUTS/codellama_mbpp.jsonl \
  --out $OUTPUTS/codellama_mbpp_reflexion_feedback_r1.jsonl \
  --feedback_key eval.error \
  --device cuda:0 --resume
```

评测：

```bash
python -m coder.scripts.eval_evalplus \
  --samples $OUTPUTS/codellama_humaneval_reflexion_feedback_r1.jsonl \
  --dataset humaneval --model codellama_reflexion_feedback_r1

python -m coder.scripts.eval_evalplus \
  --samples $OUTPUTS/codellama_mbpp_reflexion_feedback_r1.jsonl \
  --dataset mbpp --model codellama_reflexion_feedback_r1
```

> **注意**：CodeLlama instruct 跟随能力偏弱，Reflexion 效果可能很差甚至退化，属正常负面结果。

### Step 2.3：Reranking (k=8, logprob)

> `gen_rerank.py` 默认 `--score_mode self_judge`，但论文已有结果均用 `logprob` 模式，必须显式传入。

```bash
# HumanEval
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_rerank \
  --model codellama \
  --dataset humaneval \
  --out $OUTPUTS/codellama_humaneval_rerank_logprob_k8.jsonl \
  --num_samples 8 \
  --score_mode logprob \
  --device cuda:0

# MBPP
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_rerank \
  --model codellama \
  --dataset mbpp \
  --out $OUTPUTS/codellama_mbpp_rerank_logprob_k8.jsonl \
  --num_samples 8 \
  --score_mode logprob \
  --device cuda:0
```

评测：

```bash
python -m coder.scripts.eval_evalplus \
  --samples $OUTPUTS/codellama_humaneval_rerank_logprob_k8.jsonl \
  --dataset humaneval --model codellama_rerank_logprob_k8

python -m coder.scripts.eval_evalplus \
  --samples $OUTPUTS/codellama_mbpp_rerank_logprob_k8.jsonl \
  --dataset mbpp --model codellama_rerank_logprob_k8
```

> Reranking 需要 8 次完整生成，CodeLlama（14GB bfloat16）在单张 A100 80G 上应能跑完。

### Step 2.4：dLLM-locate + AR-rewrite

```bash
# HumanEval（Dream locator on cuda:0，CodeLlama rewriter on cuda:0 或 cuda:1）
CUDA_VISIBLE_DEVICES=<GPU0>,<GPU1> python -m coder.scripts.gen_locate_ar_rewrite \
  --ar_model codellama \
  --ar_device cuda:1 \
  --locator_device cuda:0 \
  --input $OUTPUTS/codellama_humaneval.jsonl \
  --out $OUTPUTS/codellama_humaneval_locate_ar_rewrite_t0.9.jsonl \
  --confidence_threshold 0.9 --resume

# MBPP
CUDA_VISIBLE_DEVICES=<GPU0>,<GPU1> python -m coder.scripts.gen_locate_ar_rewrite \
  --ar_model codellama \
  --ar_device cuda:1 \
  --locator_device cuda:0 \
  --input $OUTPUTS/codellama_mbpp.jsonl \
  --out $OUTPUTS/codellama_mbpp_locate_ar_rewrite_t0.9.jsonl \
  --confidence_threshold 0.9 --resume
```

评测：

```bash
python -m coder.scripts.eval_evalplus \
  --samples $OUTPUTS/codellama_humaneval_locate_ar_rewrite_t0.9.jsonl \
  --dataset humaneval --model codellama_locate_ar_rewrite_t0.9

python -m coder.scripts.eval_evalplus \
  --samples $OUTPUTS/codellama_mbpp_locate_ar_rewrite_t0.9.jsonl \
  --dataset mbpp --model codellama_locate_ar_rewrite_t0.9
```

> 若单卡 OOM，Dream locator 约 15GB + CodeLlama 约 14GB = 29GB；两张 A100 各放一个即可。

---

## Phase 3：Mistral 7B 四种 baseline 方法

AR 草稿文件已存在：`mistral_humaneval.jsonl`、`mistral_mbpp.jsonl`。
Sanitized 文件已存在：`mistral_humaneval-sanitized.jsonl`、`mistral_mbpp-sanitized.jsonl`（如不存在先跑 `postprocess_evalplus`）。

### Step 3.1：Self-Refine

```bash
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_self_refine \
  --model mistral \
  --input $OUTPUTS/mistral_humaneval.jsonl \
  --out $OUTPUTS/mistral_humaneval_selfrefine_r1.jsonl \
  --device cuda:0 --resume

CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_self_refine \
  --model mistral \
  --input $OUTPUTS/mistral_mbpp.jsonl \
  --out $OUTPUTS/mistral_mbpp_selfrefine_r1.jsonl \
  --device cuda:0 --resume
```

评测同 CodeLlama 格式，`--model mistral_selfrefine_r1`。

### Step 3.2：Reflexion (w/ feedback)

```bash
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_reflexion \
  --model mistral \
  --input $OUTPUTS/mistral_humaneval-sanitized.jsonl \
  --raw_input $OUTPUTS/mistral_humaneval.jsonl \
  --out $OUTPUTS/mistral_humaneval_reflexion_feedback_r1.jsonl \
  --feedback_key eval.error \
  --device cuda:0 --resume

CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_reflexion \
  --model mistral \
  --input $OUTPUTS/mistral_mbpp-sanitized.jsonl \
  --raw_input $OUTPUTS/mistral_mbpp.jsonl \
  --out $OUTPUTS/mistral_mbpp_reflexion_feedback_r1.jsonl \
  --feedback_key eval.error \
  --device cuda:0 --resume
```

评测同上，`--model mistral_reflexion_feedback_r1`。

### Step 3.3：Reranking (k=8, logprob)

```bash
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_rerank \
  --model mistral \
  --dataset humaneval \
  --out $OUTPUTS/mistral_humaneval_rerank_logprob_k8.jsonl \
  --num_samples 8 --score_mode logprob --device cuda:0

CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_rerank \
  --model mistral \
  --dataset mbpp \
  --out $OUTPUTS/mistral_mbpp_rerank_logprob_k8.jsonl \
  --num_samples 8 --score_mode logprob --device cuda:0
```

评测同上，`--model mistral_rerank_logprob_k8`。

### Step 3.4：dLLM-locate + AR-rewrite

```bash
CUDA_VISIBLE_DEVICES=<GPU0>,<GPU1> python -m coder.scripts.gen_locate_ar_rewrite \
  --ar_model mistral \
  --ar_device cuda:1 \
  --locator_device cuda:0 \
  --input $OUTPUTS/mistral_humaneval.jsonl \
  --out $OUTPUTS/mistral_humaneval_locate_ar_rewrite_t0.9.jsonl \
  --confidence_threshold 0.9 --resume

CUDA_VISIBLE_DEVICES=<GPU0>,<GPU1> python -m coder.scripts.gen_locate_ar_rewrite \
  --ar_model mistral \
  --ar_device cuda:1 \
  --locator_device cuda:0 \
  --input $OUTPUTS/mistral_mbpp.jsonl \
  --out $OUTPUTS/mistral_mbpp_locate_ar_rewrite_t0.9.jsonl \
  --confidence_threshold 0.9 --resume
```

---

## Phase 4：Seed-Coder-Instruct 8B 四种 baseline 方法

AR 草稿文件已存在：`seed-coder-instruct_humaneval.jsonl`、`seed-coder-instruct_mbpp.jsonl`。
Sanitized 文件：检查 `seed-coder-instruct_humaneval-sanitized.jsonl`，如不存在先跑：

```bash
python -m coder.scripts.postprocess_evalplus \
  --dataset humaneval --samples $OUTPUTS/seed-coder-instruct_humaneval.jsonl

python -m coder.scripts.postprocess_evalplus \
  --dataset mbpp --samples $OUTPUTS/seed-coder-instruct_mbpp.jsonl
```

### Step 4.1：Self-Refine

```bash
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_self_refine \
  --model seed-coder \
  --model_id ByteDance-Seed/Seed-Coder-8B-Instruct \
  --input $OUTPUTS/seed-coder-instruct_humaneval.jsonl \
  --out $OUTPUTS/seed-coder-instruct_humaneval_selfrefine_r1.jsonl \
  --device cuda:0 --resume

CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_self_refine \
  --model seed-coder \
  --model_id ByteDance-Seed/Seed-Coder-8B-Instruct \
  --input $OUTPUTS/seed-coder-instruct_mbpp.jsonl \
  --out $OUTPUTS/seed-coder-instruct_mbpp_selfrefine_r1.jsonl \
  --device cuda:0 --resume
```

评测：

```bash
python -m coder.scripts.eval_evalplus \
  --samples $OUTPUTS/seed-coder-instruct_humaneval_selfrefine_r1.jsonl \
  --dataset humaneval --model seed_coder_instruct_selfrefine_r1

python -m coder.scripts.eval_evalplus \
  --samples $OUTPUTS/seed-coder-instruct_mbpp_selfrefine_r1.jsonl \
  --dataset mbpp --model seed_coder_instruct_selfrefine_r1
```

### Step 4.2：Reflexion (w/ feedback)

```bash
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_reflexion \
  --model seed-coder \
  --model_id ByteDance-Seed/Seed-Coder-8B-Instruct \
  --input $OUTPUTS/seed-coder-instruct_humaneval-sanitized.jsonl \
  --raw_input $OUTPUTS/seed-coder-instruct_humaneval.jsonl \
  --out $OUTPUTS/seed-coder-instruct_humaneval_reflexion_feedback_r1.jsonl \
  --feedback_key eval.error \
  --device cuda:0 --resume

CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_reflexion \
  --model seed-coder \
  --model_id ByteDance-Seed/Seed-Coder-8B-Instruct \
  --input $OUTPUTS/seed-coder-instruct_mbpp-sanitized.jsonl \
  --raw_input $OUTPUTS/seed-coder-instruct_mbpp.jsonl \
  --out $OUTPUTS/seed-coder-instruct_mbpp_reflexion_feedback_r1.jsonl \
  --feedback_key eval.error \
  --device cuda:0 --resume
```

评测同上，`--model seed_coder_instruct_reflexion_feedback_r1`。

### Step 4.3：Reranking (k=8, logprob)

```bash
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_rerank \
  --model seed-coder \
  --model_id ByteDance-Seed/Seed-Coder-8B-Instruct \
  --dataset humaneval \
  --out $OUTPUTS/seed-coder-instruct_humaneval_rerank_logprob_k8.jsonl \
  --num_samples 8 --score_mode logprob --device cuda:0

CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_rerank \
  --model seed-coder \
  --model_id ByteDance-Seed/Seed-Coder-8B-Instruct \
  --dataset mbpp \
  --out $OUTPUTS/seed-coder-instruct_mbpp_rerank_logprob_k8.jsonl \
  --num_samples 8 --score_mode logprob --device cuda:0
```

评测，`--model seed_coder_instruct_rerank_logprob_k8`。

### Step 4.4：dLLM-locate + AR-rewrite

```bash
CUDA_VISIBLE_DEVICES=<GPU0>,<GPU1> python -m coder.scripts.gen_locate_ar_rewrite \
  --ar_model seed-coder \
  --model_id ByteDance-Seed/Seed-Coder-8B-Instruct \
  --ar_device cuda:1 \
  --locator_device cuda:0 \
  --input $OUTPUTS/seed-coder-instruct_humaneval.jsonl \
  --out $OUTPUTS/seed-coder-instruct_humaneval_locate_ar_rewrite_t0.9.jsonl \
  --confidence_threshold 0.9 --resume

CUDA_VISIBLE_DEVICES=<GPU0>,<GPU1> python -m coder.scripts.gen_locate_ar_rewrite \
  --ar_model seed-coder \
  --model_id ByteDance-Seed/Seed-Coder-8B-Instruct \
  --ar_device cuda:1 \
  --locator_device cuda:0 \
  --input $OUTPUTS/seed-coder-instruct_mbpp.jsonl \
  --out $OUTPUTS/seed-coder-instruct_mbpp_locate_ar_rewrite_t0.9.jsonl \
  --confidence_threshold 0.9 --resume
```

> **注意 `gen_locate_ar_rewrite.py` 的 `--model_id` flag**：该脚本当前的 `build_model()` 可能用 `--ar_model` 而不支持 `--model_id`。检查脚本参数，必要时在 Phase 1 Step 1d 中同时添加 `--model_id` 参数透传。

---

## Phase 5：更新 `section/05_experiments.tex`

所有数字收集完成后（Phase 0 + Phases 2–4 评测结果），将 `tab:combined` 中三组模型的 `---` 替换为真实数字。

定位方式（在 `section/05_experiments.tex` 中 `\multirow{7}{*}{CodeLlama 7B}` 等块内）：

```latex
\multirow{7}{*}{CodeLlama 7B}
& AR-only                           & 29.3 & 25.6 & 38.4 & 30.7 & --- \\
& + Self-Refine (1 round)           & X.X  & X.X  & X.X  & X.X  & X.X \\
& + Reflexion (1 round)             & X.X  & X.X  & X.X  & X.X  & X.X \\
& + Reranking ($k{=}8$)             & X.X  & X.X  & X.X  & X.X  & X.X \\
& + dLLM-locate + AR-rewrite        & X.X  & X.X  & X.X  & X.X  & X.X \\
& + LLaDA remask ($\tau{=}0.9$)     & X.X  & X.X  & X.X  & X.X  & X.X \\
& + Dream remask ($\tau{=}0.9$, \textbf{ours}) & X.X & X.X & X.X & X.X & X.X \\
```

列顺序：HumanEval, HumanEval+, MBPP, MBPP+, Overhead(s)。

**注意**：
- AR-only 的 Overhead 列填 `---`（表头已说明 overhead 是方法本身的额外耗时）
- 若某个方法结果很差（如 Self-Refine 对 StarCoder2 的 7.9%），保留原始数字，不过滤
- Overhead 取 HumanEval 上的 s/sample（表头已注明是 HumanEval overhead）

---

## Phase 6：验收检查

```bash
# 检查各 summary 文件存在且 n_tasks 正确
python3 -c "
import json, pathlib
OUTPUTS = pathlib.Path('outputs/base_tuteng')
checks = [
    ('codellama_humaneval_selfrefine_r1_summary.json', 164),
    ('codellama_mbpp_selfrefine_r1_summary.json', 378),
    ('codellama_humaneval_reflexion_feedback_r1_summary.json', 164),
    ('codellama_mbpp_reflexion_feedback_r1_summary.json', 378),
    ('codellama_humaneval_rerank_logprob_k8_summary.json', 164),
    ('codellama_mbpp_rerank_logprob_k8_summary.json', 378),
    ('codellama_humaneval_locate_ar_rewrite_t0.9_summary.json', 164),
    ('codellama_mbpp_locate_ar_rewrite_t0.9_summary.json', 378),
    ('mistral_humaneval_selfrefine_r1_summary.json', 164),
    ('mistral_mbpp_selfrefine_r1_summary.json', 378),
    ('mistral_humaneval_reflexion_feedback_r1_summary.json', 164),
    ('mistral_mbpp_reflexion_feedback_r1_summary.json', 378),
    ('mistral_humaneval_rerank_logprob_k8_summary.json', 164),
    ('mistral_mbpp_rerank_logprob_k8_summary.json', 378),
    ('mistral_humaneval_locate_ar_rewrite_t0.9_summary.json', 164),
    ('mistral_mbpp_locate_ar_rewrite_t0.9_summary.json', 378),
    ('seed-coder-instruct_humaneval_selfrefine_r1_summary.json', 164),
    ('seed-coder-instruct_mbpp_selfrefine_r1_summary.json', 378),
    ('seed-coder-instruct_humaneval_reflexion_feedback_r1_summary.json', 164),
    ('seed-coder-instruct_mbpp_reflexion_feedback_r1_summary.json', 378),
    ('seed-coder-instruct_humaneval_rerank_logprob_k8_summary.json', 164),
    ('seed-coder-instruct_mbpp_rerank_logprob_k8_summary.json', 378),
    ('seed-coder-instruct_humaneval_locate_ar_rewrite_t0.9_summary.json', 164),
    ('seed-coder-instruct_mbpp_locate_ar_rewrite_t0.9_summary.json', 378),
]
for fname, expected_n in checks:
    p = OUTPUTS / fname
    if not p.exists():
        print(f'MISSING: {fname}')
        continue
    n = json.load(open(p))['summary']['n_tasks']
    status = '✅' if n == expected_n else f'❌ got {n}'
    print(f'{status} {fname}')
"
```

---

## 注意事项

1. **sanitized JSONL 前置依赖**：Reflexion 的 `--input` 需要 sanitized JSONL（含 `eval.error` 字段）。若某模型的 sanitized 文件不存在，先跑：
   ```bash
   python -m coder.scripts.postprocess_evalplus --dataset humaneval --samples $OUTPUTS/<model>_humaneval.jsonl
   python -m coder.scripts.postprocess_evalplus --dataset mbpp --samples $OUTPUTS/<model>_mbpp.jsonl
   ```

2. **显存估算**：
   - Self-Refine / Reflexion / Reranking：单模型推理，单张 A100 80G 足够
   - Locate-AR-Rewrite：Dream-Coder locator (~15GB) + AR model (~14GB)；建议两卡，`locator_device cuda:0`，`ar_device cuda:1`

3. **输出文件命名约定**：
   - Seed-Coder-Instruct 使用连字符前缀 `seed-coder-instruct_`（保持与现有产物一致）
   - CodeLlama 使用 `codellama_`，Mistral 使用 `mistral_`（与现有产物一致）

4. **Reranking score_mode**：`gen_rerank.py` 默认 `--score_mode self_judge`，但论文已有结果均采用 `logprob` 模式。所有 Reranking 步骤均已在命令中写入 `--score_mode logprob`，请勿省略。

---

## 最终执行结果（2026-06-04）

本 spec 的新增实验已全部生成并评测完成。以下路径均相对于仓库根目录；`Raw JSONL` 是原始完整生成结果，`Summary` 是 `eval_evalplus` 写出的汇总结果。

| AR Model | Method | HE | HE+ | MBPP | MBPP+ | Raw JSONL | Summary |
|----------|--------|----|-----|------|-------|-----------|---------|
| CodeLlama 7B | Self-Refine | 36.6% | 32.3% | 50.3% | 43.1% | `outputs/base_tuteng/codellama_humaneval_selfrefine_r1.jsonl`; `outputs/base_tuteng/codellama_mbpp_selfrefine_r1.jsonl` | `outputs/base_tuteng/codellama_humaneval_selfrefine_r1_summary.json`; `outputs/base_tuteng/codellama_mbpp_selfrefine_r1_summary.json` |
| CodeLlama 7B | Reflexion | 11.0% | 7.3% | 19.0% | 12.7% | `outputs/base_tuteng/codellama_humaneval_reflexion_feedback_r1.jsonl`; `outputs/base_tuteng/codellama_mbpp_reflexion_feedback_r1.jsonl` | `outputs/base_tuteng/codellama_humaneval_reflexion_feedback_r1_summary.json`; `outputs/base_tuteng/codellama_mbpp_reflexion_feedback_r1_summary.json` |
| CodeLlama 7B | Reranking (`k=8`, logprob) | 37.2% | 31.7% | 51.1% | 41.8% | `outputs/base_tuteng/codellama_humaneval_rerank_logprob_k8.jsonl`; `outputs/base_tuteng/codellama_mbpp_rerank_logprob_k8.jsonl` | `outputs/base_tuteng/codellama_humaneval_rerank_logprob_k8_summary.json`; `outputs/base_tuteng/codellama_mbpp_rerank_logprob_k8_summary.json` |
| CodeLlama 7B | dLLM-locate + AR-rewrite | 20.1% | 17.1% | 28.0% | 23.5% | `outputs/base_tuteng/codellama_humaneval_locate_ar_rewrite_t0.9.jsonl`; `outputs/base_tuteng/codellama_mbpp_locate_ar_rewrite_t0.9.jsonl` | `outputs/base_tuteng/codellama_humaneval_locate_ar_rewrite_t0.9_summary.json`; `outputs/base_tuteng/codellama_mbpp_locate_ar_rewrite_t0.9_summary.json` |
| Mistral 7B | Self-Refine | 31.7% | 23.8% | 42.6% | 37.6% | `outputs/base_tuteng/mistral_humaneval_selfrefine_r1.jsonl`; `outputs/base_tuteng/mistral_mbpp_selfrefine_r1.jsonl` | `outputs/base_tuteng/mistral_humaneval_selfrefine_r1_summary.json`; `outputs/base_tuteng/mistral_mbpp_selfrefine_r1_summary.json` |
| Mistral 7B | Reflexion | 29.3% | 23.2% | 38.9% | 29.1% | `outputs/base_tuteng/mistral_humaneval_reflexion_feedback_r1.jsonl`; `outputs/base_tuteng/mistral_mbpp_reflexion_feedback_r1.jsonl` | `outputs/base_tuteng/mistral_humaneval_reflexion_feedback_r1_summary.json`; `outputs/base_tuteng/mistral_mbpp_reflexion_feedback_r1_summary.json` |
| Mistral 7B | Reranking (`k=8`, logprob) | 42.1% | 34.8% | 51.1% | 44.2% | `outputs/base_tuteng/mistral_humaneval_rerank_logprob_k8.jsonl`; `outputs/base_tuteng/mistral_mbpp_rerank_logprob_k8.jsonl` | `outputs/base_tuteng/mistral_humaneval_rerank_logprob_k8_summary.json`; `outputs/base_tuteng/mistral_mbpp_rerank_logprob_k8_summary.json` |
| Mistral 7B | dLLM-locate + AR-rewrite | 32.9% | 28.7% | 49.7% | 43.4% | `outputs/base_tuteng/mistral_humaneval_locate_ar_rewrite_t0.9.jsonl`; `outputs/base_tuteng/mistral_mbpp_locate_ar_rewrite_t0.9.jsonl` | `outputs/base_tuteng/mistral_humaneval_locate_ar_rewrite_t0.9_summary.json`; `outputs/base_tuteng/mistral_mbpp_locate_ar_rewrite_t0.9_summary.json` |
| Seed-Coder-Instruct 8B | Self-Refine | 78.7% | 74.4% | 81.0% | 69.0% | `outputs/base_tuteng/seed-coder-instruct_humaneval_selfrefine_r1.jsonl`; `outputs/base_tuteng/seed-coder-instruct_mbpp_selfrefine_r1.jsonl` | `outputs/base_tuteng/seed-coder-instruct_humaneval_selfrefine_r1_summary.json`; `outputs/base_tuteng/seed-coder-instruct_mbpp_selfrefine_r1_summary.json` |
| Seed-Coder-Instruct 8B | Reflexion | 67.1% | 57.3% | 66.9% | 49.2% | `outputs/base_tuteng/seed-coder-instruct_humaneval_reflexion_feedback_r1.jsonl`; `outputs/base_tuteng/seed-coder-instruct_mbpp_reflexion_feedback_r1.jsonl` | `outputs/base_tuteng/seed-coder-instruct_humaneval_reflexion_feedback_r1_summary.json`; `outputs/base_tuteng/seed-coder-instruct_mbpp_reflexion_feedback_r1_summary.json` |
| Seed-Coder-Instruct 8B | Reranking (`k=8`, logprob) | 82.3% | 77.4% | 86.0% | 73.8% | `outputs/base_tuteng/seed-coder-instruct_humaneval_rerank_logprob_k8.jsonl`; `outputs/base_tuteng/seed-coder-instruct_mbpp_rerank_logprob_k8.jsonl` | `outputs/base_tuteng/seed-coder-instruct_humaneval_rerank_logprob_k8_summary.json`; `outputs/base_tuteng/seed-coder-instruct_mbpp_rerank_logprob_k8_summary.json` |
| Seed-Coder-Instruct 8B | dLLM-locate + AR-rewrite | 81.7% | 76.2% | 82.8% | 69.8% | `outputs/base_tuteng/seed-coder-instruct_humaneval_locate_ar_rewrite_t0.9.jsonl`; `outputs/base_tuteng/seed-coder-instruct_mbpp_locate_ar_rewrite_t0.9.jsonl` | `outputs/base_tuteng/seed-coder-instruct_humaneval_locate_ar_rewrite_t0.9_summary.json`; `outputs/base_tuteng/seed-coder-instruct_mbpp_locate_ar_rewrite_t0.9_summary.json` |

### 完整性检查

| Artifact group | Expected | Status |
|----------------|----------|--------|
| CodeLlama HumanEval locate+rewrite | 164 records | complete |
| CodeLlama MBPP locate+rewrite | 378 records | complete |
| Mistral HumanEval locate+rewrite | 164 records | complete |
| Mistral MBPP locate+rewrite | 378 records | complete |
| Seed-Coder-Instruct HumanEval locate+rewrite | 164 records | complete |
| Seed-Coder-Instruct MBPP locate+rewrite | 378 records | complete |

队列日志：`outputs/base_tuteng/table3_queue/logs/locate.log`，完成时间 `2026-06-04 21:55:55 UTC`。

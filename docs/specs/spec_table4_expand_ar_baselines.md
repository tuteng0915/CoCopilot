# Spec: 扩展 Table 4 baseline 对比到其他 AR 模型

## 目标

将 Table 4（DeepSeek-Coder baselines）的方法对比格式，复制到 **Llama-3.1 8B** 和 **StarCoder2 7B** 上，
使每个 AR 模型都有一张独立的 baseline 对比表（对应论文 Table 4c / Table 4d）。

每张表的行结构与 Table 4 完全一致：

| 方法 | HE+ plus% | HE+ base% | MBPP+ plus% | MBPP+ base% | s/sample (HE) | s/sample (MBPP) |
|---|---|---|---|---|---|---|
| {AR} baseline | — | — | — | — | — | — |
| + Self-Refine | | | | | | |
| + Reflexion (w/ feedback) | | | | | | |
| + Rerank logprob k=8 | | | | | | |
| + Locate-AR-Rewrite | | | | | | |
| + LLaDA remask τ=0.9 | | | | | | |
| + Dream remask τ=0.9 (ours) | | | | | | |

---

## 前提条件

```bash
source /home/tteng/miniconda3/etc/profile.d/conda.sh
conda activate code
cd /model/tteng/CoCoder
export PYTHONPATH=src
```

### 已存在产物（无需重新生成）

| 文件 | 说明 |
|---|---|
| `outputs/base_tuteng/llama31_humaneval.jsonl` | Llama-3.1 AR 草稿 (164 条) |
| `outputs/base_tuteng/llama31_mbpp.jsonl` | Llama-3.1 AR 草稿 (378 条) |
| `outputs/base_tuteng/llama31_humaneval_summary.json` | AR baseline 评测结果 |
| `outputs/base_tuteng/llama31_mbpp_summary.json` | |
| `outputs/base_tuteng/llama31_humaneval_timed.jsonl.timing_summary.json` | AR baseline timing |
| `outputs/base_tuteng/llama31_mbpp_timed.jsonl.timing_summary.json` | |
| `outputs/base_tuteng/llama31_dream_remask_humaneval_t0.9.jsonl` | Dream remask 结果 |
| `outputs/base_tuteng/llama31_dream_remask_humaneval_t0.9_summary.json` | |
| `outputs/base_tuteng/llama31_dream_remask_humaneval_t0.9.jsonl.timing_summary.json` | |
| `outputs/base_tuteng/llama31_dream_remask_mbpp_t0.9.jsonl` | |
| `outputs/base_tuteng/llama31_dream_remask_mbpp_t0.9_summary.json` | |
| `outputs/base_tuteng/llama31_dream_remask_mbpp_t0.9.jsonl.timing_summary.json` | |
| `outputs/base_tuteng/llama31_llada_remask_humaneval_t0.9.jsonl` | LLaDA remask 结果 |
| `outputs/base_tuteng/llama31_llada_remask_humaneval_t0.9_summary.json` | |
| `outputs/base_tuteng/llama31_llada_remask_humaneval_t0.9.jsonl.timing_summary.json` | |
| `outputs/base_tuteng/llama31_llada_remask_mbpp_t0.9.jsonl` | |
| `outputs/base_tuteng/llama31_llada_remask_mbpp_t0.9_summary.json` | |
| `outputs/base_tuteng/llama31_llada_remask_mbpp_t0.9.jsonl.timing_summary.json` | |
| `outputs/base_tuteng/starcoder2_humaneval.jsonl` | StarCoder2 AR 草稿 (164 条) |
| `outputs/base_tuteng/starcoder2_mbpp.jsonl` | StarCoder2 AR 草稿 (378 条) |
| `outputs/base_tuteng/starcoder2_humaneval_summary.json` | AR baseline 评测结果 |
| `outputs/base_tuteng/starcoder2_mbpp_summary.json` | |
| `outputs/base_tuteng/starcoder2_humaneval_timed.jsonl.timing_summary.json` | AR baseline timing |
| `outputs/base_tuteng/starcoder2_mbpp_timed.jsonl.timing_summary.json` | |
| `outputs/base_tuteng/starcoder2_dream_remask_humaneval_t0.9.jsonl` | Dream remask 结果 |
| `outputs/base_tuteng/starcoder2_dream_remask_humaneval_t0.9_summary.json` | |
| `outputs/base_tuteng/starcoder2_dream_remask_humaneval_t0.9.jsonl.timing_summary.json` | |
| `outputs/base_tuteng/starcoder2_dream_remask_mbpp_t0.9.jsonl` | |
| `outputs/base_tuteng/starcoder2_dream_remask_mbpp_t0.9_summary.json` | |
| `outputs/base_tuteng/starcoder2_dream_remask_mbpp_t0.9.jsonl.timing_summary.json` | |
| `outputs/base_tuteng/starcoder2_llada_remask_humaneval_t0.9.jsonl` | LLaDA remask 结果 |
| `outputs/base_tuteng/starcoder2_llada_remask_humaneval_t0.9_summary.json` | |
| `outputs/base_tuteng/starcoder2_llada_remask_humaneval_t0.9.jsonl.timing_summary.json` | |
| `outputs/base_tuteng/starcoder2_llada_remask_mbpp_t0.9.jsonl` | |
| `outputs/base_tuteng/starcoder2_llada_remask_mbpp_t0.9_summary.json` | |
| `outputs/base_tuteng/starcoder2_llada_remask_mbpp_t0.9.jsonl.timing_summary.json` | |

---

## Part A：代码修改（必须先做）

### A1：扩展 `gen_self_refine.py` 支持 llama31 / starcoder2

文件：`src/coder/scripts/gen_self_refine.py`

在 `build_model()` 函数（约第 25 行）中，在 `raise ValueError(...)` 前加入：

```python
from coder.models import Llama31Coder, StarCoder2Coder  # 在文件顶部 import 区补充

# 在 build_model() 中添加：
if name in ["llama31", "llama31_coder", "llama3.1"]:
    return Llama31Coder(
        model_id=model_id or "meta-llama/Llama-3.1-8B-Instruct",
        device=device,
    )
if name in ["starcoder2", "starcoder2_coder", "sc2"]:
    return StarCoder2Coder(
        model_id=model_id or "bigcode/starcoder2-7b",
        device=device,
    )
```

### A2：扩展 `gen_rerank.py` 支持 llama31 / starcoder2

文件：`src/coder/scripts/gen_rerank.py`

**Step A2a** — 在 `build_model()` 函数（约第 20 行）顶部 import 区补充：

```python
from coder.models import DeepSeekCoder, QwenCoder, Llama31Coder, StarCoder2Coder, CoderModel
```

**Step A2b** — 在 `build_model()` 的 `raise ValueError(...)` 前添加：

```python
if name in ["llama31", "llama31_coder", "llama3.1"]:
    return Llama31Coder(
        model_id=model_id or "meta-llama/Llama-3.1-8B-Instruct",
        device=device,
    )
if name in ["starcoder2", "starcoder2_coder", "sc2"]:
    return StarCoder2Coder(
        model_id=model_id or "bigcode/starcoder2-7b",
        device=device,
    )
```

**Step A2c** — 更新 `--model` choices（约第 160 行）：

```python
ap.add_argument(
    "--model",
    choices=["deepseek", "qwen", "llama31", "starcoder2"],
    required=True,
)
```

---

## Part B：Llama-3.1 8B baseline 对比实验

### B1：Self-Refine

```bash
# HumanEval
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_self_refine \
  --model llama31 \
  --input outputs/base_tuteng/llama31_humaneval.jsonl \
  --out outputs/base_tuteng/llama31_humaneval_selfrefine_r1.jsonl \
  --device cuda:0

# MBPP
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_self_refine \
  --model llama31 \
  --input outputs/base_tuteng/llama31_mbpp.jsonl \
  --out outputs/base_tuteng/llama31_mbpp_selfrefine_r1.jsonl \
  --device cuda:0
```

评测：

```bash
python -m coder.scripts.eval_evalplus \
  --samples outputs/base_tuteng/llama31_humaneval_selfrefine_r1.jsonl \
  --dataset humaneval --model llama31_selfrefine_r1

python -m coder.scripts.eval_evalplus \
  --samples outputs/base_tuteng/llama31_mbpp_selfrefine_r1.jsonl \
  --dataset mbpp --model llama31_selfrefine_r1
```

产物：`llama31_humaneval_selfrefine_r1_summary.json`，`llama31_mbpp_selfrefine_r1_summary.json`

### B2：Reflexion (w/ feedback)

```bash
# HumanEval
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_reflexion \
  --model llama31 \
  --input outputs/base_tuteng/llama31_humaneval-sanitized.jsonl \
  --raw_input outputs/base_tuteng/llama31_humaneval.jsonl \
  --out outputs/base_tuteng/llama31_humaneval_reflexion_feedback_r1.jsonl \
  --feedback_key eval.error \
  --device cuda:0

# MBPP
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_reflexion \
  --model llama31 \
  --input outputs/base_tuteng/llama31_mbpp-sanitized.jsonl \
  --raw_input outputs/base_tuteng/llama31_mbpp.jsonl \
  --out outputs/base_tuteng/llama31_mbpp_reflexion_feedback_r1.jsonl \
  --feedback_key eval.error \
  --device cuda:0
```

评测：

```bash
python -m coder.scripts.eval_evalplus \
  --samples outputs/base_tuteng/llama31_humaneval_reflexion_feedback_r1.jsonl \
  --dataset humaneval --model llama31_reflexion_feedback_r1

python -m coder.scripts.eval_evalplus \
  --samples outputs/base_tuteng/llama31_mbpp_reflexion_feedback_r1.jsonl \
  --dataset mbpp --model llama31_reflexion_feedback_r1
```

产物：`llama31_humaneval_reflexion_feedback_r1_summary.json`，`llama31_mbpp_reflexion_feedback_r1_summary.json`

### B3：Rerank logprob k=8

```bash
# HumanEval
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_rerank \
  --model llama31 \
  --dataset humaneval \
  --out outputs/base_tuteng/llama31_humaneval_rerank_logprob_k8.jsonl \
  --num_samples 8 \
  --device cuda:0

# MBPP
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_rerank \
  --model llama31 \
  --dataset mbpp \
  --out outputs/base_tuteng/llama31_mbpp_rerank_logprob_k8.jsonl \
  --num_samples 8 \
  --device cuda:0
```

评测：

```bash
python -m coder.scripts.eval_evalplus \
  --samples outputs/base_tuteng/llama31_humaneval_rerank_logprob_k8.jsonl \
  --dataset humaneval --model llama31_rerank_logprob_k8

python -m coder.scripts.eval_evalplus \
  --samples outputs/base_tuteng/llama31_mbpp_rerank_logprob_k8.jsonl \
  --dataset mbpp --model llama31_rerank_logprob_k8
```

产物：`llama31_humaneval_rerank_logprob_k8_summary.json`，`llama31_mbpp_rerank_logprob_k8_summary.json`

### B4：Locate-AR-Rewrite

```bash
# HumanEval（locator=Dream on cuda:0，AR=llama31 on cuda:0，同卡可行；若 OOM 拆两卡）
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_locate_ar_rewrite \
  --ar_model llama31 \
  --ar_device cuda:0 \
  --locator_device cuda:0 \
  --input outputs/base_tuteng/llama31_humaneval.jsonl \
  --out outputs/base_tuteng/llama31_humaneval_locate_ar_rewrite_t0.9.jsonl \
  --confidence_threshold 0.9

# MBPP
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_locate_ar_rewrite \
  --ar_model llama31 \
  --ar_device cuda:0 \
  --locator_device cuda:0 \
  --input outputs/base_tuteng/llama31_mbpp.jsonl \
  --out outputs/base_tuteng/llama31_mbpp_locate_ar_rewrite_t0.9.jsonl \
  --confidence_threshold 0.9
```

评测：

```bash
python -m coder.scripts.eval_evalplus \
  --samples outputs/base_tuteng/llama31_humaneval_locate_ar_rewrite_t0.9.jsonl \
  --dataset humaneval --model llama31_locate_ar_rewrite_t0.9

python -m coder.scripts.eval_evalplus \
  --samples outputs/base_tuteng/llama31_mbpp_locate_ar_rewrite_t0.9.jsonl \
  --dataset mbpp --model llama31_locate_ar_rewrite_t0.9
```

产物：`llama31_humaneval_locate_ar_rewrite_t0.9_summary.json`，`llama31_mbpp_locate_ar_rewrite_t0.9_summary.json`

---

## Part C：StarCoder2 7B baseline 对比实验

### C1：Self-Refine

```bash
# HumanEval
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_self_refine \
  --model starcoder2 \
  --input outputs/base_tuteng/starcoder2_humaneval.jsonl \
  --out outputs/base_tuteng/starcoder2_humaneval_selfrefine_r1.jsonl \
  --device cuda:0

# MBPP
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_self_refine \
  --model starcoder2 \
  --input outputs/base_tuteng/starcoder2_mbpp.jsonl \
  --out outputs/base_tuteng/starcoder2_mbpp_selfrefine_r1.jsonl \
  --device cuda:0
```

评测：

```bash
python -m coder.scripts.eval_evalplus \
  --samples outputs/base_tuteng/starcoder2_humaneval_selfrefine_r1.jsonl \
  --dataset humaneval --model starcoder2_selfrefine_r1

python -m coder.scripts.eval_evalplus \
  --samples outputs/base_tuteng/starcoder2_mbpp_selfrefine_r1.jsonl \
  --dataset mbpp --model starcoder2_selfrefine_r1
```

产物：`starcoder2_humaneval_selfrefine_r1_summary.json`，`starcoder2_mbpp_selfrefine_r1_summary.json`

### C2：Reflexion (w/ feedback)

```bash
# HumanEval
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_reflexion \
  --model starcoder2 \
  --input outputs/base_tuteng/starcoder2_humaneval-sanitized.jsonl \
  --raw_input outputs/base_tuteng/starcoder2_humaneval.jsonl \
  --out outputs/base_tuteng/starcoder2_humaneval_reflexion_feedback_r1.jsonl \
  --feedback_key eval.error \
  --device cuda:0

# MBPP
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_reflexion \
  --model starcoder2 \
  --input outputs/base_tuteng/starcoder2_mbpp-sanitized.jsonl \
  --raw_input outputs/base_tuteng/starcoder2_mbpp.jsonl \
  --out outputs/base_tuteng/starcoder2_mbpp_reflexion_feedback_r1.jsonl \
  --feedback_key eval.error \
  --device cuda:0
```

评测：

```bash
python -m coder.scripts.eval_evalplus \
  --samples outputs/base_tuteng/starcoder2_humaneval_reflexion_feedback_r1.jsonl \
  --dataset humaneval --model starcoder2_reflexion_feedback_r1

python -m coder.scripts.eval_evalplus \
  --samples outputs/base_tuteng/starcoder2_mbpp_reflexion_feedback_r1.jsonl \
  --dataset mbpp --model starcoder2_reflexion_feedback_r1
```

产物：`starcoder2_humaneval_reflexion_feedback_r1_summary.json`，`starcoder2_mbpp_reflexion_feedback_r1_summary.json`

### C3：Rerank logprob k=8

```bash
# HumanEval
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_rerank \
  --model starcoder2 \
  --dataset humaneval \
  --out outputs/base_tuteng/starcoder2_humaneval_rerank_logprob_k8.jsonl \
  --num_samples 8 \
  --device cuda:0

# MBPP
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_rerank \
  --model starcoder2 \
  --dataset mbpp \
  --out outputs/base_tuteng/starcoder2_mbpp_rerank_logprob_k8.jsonl \
  --num_samples 8 \
  --device cuda:0
```

评测：

```bash
python -m coder.scripts.eval_evalplus \
  --samples outputs/base_tuteng/starcoder2_humaneval_rerank_logprob_k8.jsonl \
  --dataset humaneval --model starcoder2_rerank_logprob_k8

python -m coder.scripts.eval_evalplus \
  --samples outputs/base_tuteng/starcoder2_mbpp_rerank_logprob_k8.jsonl \
  --dataset mbpp --model starcoder2_rerank_logprob_k8
```

产物：`starcoder2_humaneval_rerank_logprob_k8_summary.json`，`starcoder2_mbpp_rerank_logprob_k8_summary.json`

### C4：Locate-AR-Rewrite

```bash
# HumanEval
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_locate_ar_rewrite \
  --ar_model starcoder2 \
  --ar_device cuda:0 \
  --locator_device cuda:0 \
  --input outputs/base_tuteng/starcoder2_humaneval.jsonl \
  --out outputs/base_tuteng/starcoder2_humaneval_locate_ar_rewrite_t0.9.jsonl \
  --confidence_threshold 0.9

# MBPP
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_locate_ar_rewrite \
  --ar_model starcoder2 \
  --ar_device cuda:0 \
  --locator_device cuda:0 \
  --input outputs/base_tuteng/starcoder2_mbpp.jsonl \
  --out outputs/base_tuteng/starcoder2_mbpp_locate_ar_rewrite_t0.9.jsonl \
  --confidence_threshold 0.9
```

评测：

```bash
python -m coder.scripts.eval_evalplus \
  --samples outputs/base_tuteng/starcoder2_humaneval_locate_ar_rewrite_t0.9.jsonl \
  --dataset humaneval --model starcoder2_locate_ar_rewrite_t0.9

python -m coder.scripts.eval_evalplus \
  --samples outputs/base_tuteng/starcoder2_mbpp_locate_ar_rewrite_t0.9.jsonl \
  --dataset mbpp --model starcoder2_locate_ar_rewrite_t0.9
```

产物：`starcoder2_humaneval_locate_ar_rewrite_t0.9_summary.json`，`starcoder2_mbpp_locate_ar_rewrite_t0.9_summary.json`

---

## Part D：注册进 `gen_results_table.py`

文件：`src/coder/scripts/gen_results_table.py`

### D1：在 `section_table4_qwen_baselines()` 函数定义之后（约第 683 行，`> s/sample` 注释行之后），添加两个新 section。

**Step D1a** — 添加 Llama-3.1 baseline entries list 和 section 函数：

```python
# ---------------------------------------------------------------------------
# Section: Table 4c — Llama-3.1 8B Baselines
# ---------------------------------------------------------------------------

_LLAMA31_BASELINE_ENTRIES = [
    (
        "Llama-3.1 baseline",
        "llama31_humaneval_summary.json",
        "llama31_mbpp_summary.json",
        "llama31_humaneval_timed.jsonl.timing_summary.json",
        "llama31_mbpp_timed.jsonl.timing_summary.json",
    ),
    (
        "+ Self-Refine",
        "llama31_humaneval_selfrefine_r1_summary.json",
        "llama31_mbpp_selfrefine_r1_summary.json",
        "llama31_humaneval_selfrefine_r1.jsonl.timing_summary.json",
        "llama31_mbpp_selfrefine_r1.jsonl.timing_summary.json",
    ),
    (
        "+ Reflexion (w/ feedback)",
        "llama31_humaneval_reflexion_feedback_r1_summary.json",
        "llama31_mbpp_reflexion_feedback_r1_summary.json",
        "llama31_humaneval_reflexion_feedback_r1.jsonl.timing_summary.json",
        "llama31_mbpp_reflexion_feedback_r1.jsonl.timing_summary.json",
    ),
    (
        "+ Rerank logprob k=8",
        "llama31_humaneval_rerank_logprob_k8_summary.json",
        "llama31_mbpp_rerank_logprob_k8_summary.json",
        "llama31_humaneval_rerank_logprob_k8.jsonl.timing_summary.json",
        "llama31_mbpp_rerank_logprob_k8.jsonl.timing_summary.json",
    ),
    (
        "+ Locate-AR-Rewrite",
        "llama31_humaneval_locate_ar_rewrite_t0.9_summary.json",
        "llama31_mbpp_locate_ar_rewrite_t0.9_summary.json",
        "llama31_humaneval_locate_ar_rewrite_t0.9.jsonl.timing_summary.json",
        "llama31_mbpp_locate_ar_rewrite_t0.9.jsonl.timing_summary.json",
    ),
    (
        "+ LLaDA remask τ=0.9",
        "llama31_llada_remask_humaneval_t0.9_summary.json",
        "llama31_llada_remask_mbpp_t0.9_summary.json",
        "llama31_llada_remask_humaneval_t0.9.jsonl.timing_summary.json",
        "llama31_llada_remask_mbpp_t0.9.jsonl.timing_summary.json",
    ),
    (
        "+ Dream remask τ=0.9 (ours)",
        "llama31_dream_remask_humaneval_t0.9_summary.json",
        "llama31_dream_remask_mbpp_t0.9_summary.json",
        "llama31_dream_remask_humaneval_t0.9.jsonl.timing_summary.json",
        "llama31_dream_remask_mbpp_t0.9.jsonl.timing_summary.json",
    ),
]


def section_table4_llama31_baselines(out: list[str]) -> None:
    out.append("## Table 4c — Llama-3.1 8B Baselines（pass@1 plus%）\n")
    headers = [
        "方法",
        "HE+ plus%", "HE+ base%",
        "MBPP+ plus%", "MBPP+ base%",
        "s/sample (HE)", "s/sample (MBPP)",
    ]
    out.append(_fmt_row(*headers))
    out.append(_hr(len(headers)))

    for (label, he_f, mbpp_f, t_he_f, t_mb_f) in _LLAMA31_BASELINE_ENTRIES:
        he = _load_evalplus_summary(OUTPUTS / he_f if he_f else None)
        mb = _load_evalplus_summary(OUTPUTS / mbpp_f if mbpp_f else None)
        n_he = he["n_tasks"] if he else None
        n_mb = mb["n_tasks"] if mb else None

        t_he = _load_timing(OUTPUTS / t_he_f if t_he_f else None, n_he)
        t_mb = _load_timing(OUTPUTS / t_mb_f if t_mb_f else None, n_mb)

        out.append(_fmt_row(
            label,
            _pct(he["plus_pct"] if he else None),
            _pct(he["base_pct"] if he else None),
            _pct(mb["plus_pct"] if mb else None),
            _pct(mb["base_pct"] if mb else None),
            _sps(t_he), _sps(t_mb),
        ))
    out.append("")
    out.append("> s/sample = 方法总耗时 / 题目数。\n")
```

**Step D1b** — 添加 StarCoder2 baseline entries list 和 section 函数（紧接在 Llama-3.1 section 之后）：

```python
# ---------------------------------------------------------------------------
# Section: Table 4d — StarCoder2 7B Baselines
# ---------------------------------------------------------------------------

_STARCODER2_BASELINE_ENTRIES = [
    (
        "StarCoder2 baseline",
        "starcoder2_humaneval_summary.json",
        "starcoder2_mbpp_summary.json",
        "starcoder2_humaneval_timed.jsonl.timing_summary.json",
        "starcoder2_mbpp_timed.jsonl.timing_summary.json",
    ),
    (
        "+ Self-Refine",
        "starcoder2_humaneval_selfrefine_r1_summary.json",
        "starcoder2_mbpp_selfrefine_r1_summary.json",
        "starcoder2_humaneval_selfrefine_r1.jsonl.timing_summary.json",
        "starcoder2_mbpp_selfrefine_r1.jsonl.timing_summary.json",
    ),
    (
        "+ Reflexion (w/ feedback)",
        "starcoder2_humaneval_reflexion_feedback_r1_summary.json",
        "starcoder2_mbpp_reflexion_feedback_r1_summary.json",
        "starcoder2_humaneval_reflexion_feedback_r1.jsonl.timing_summary.json",
        "starcoder2_mbpp_reflexion_feedback_r1.jsonl.timing_summary.json",
    ),
    (
        "+ Rerank logprob k=8",
        "starcoder2_humaneval_rerank_logprob_k8_summary.json",
        "starcoder2_mbpp_rerank_logprob_k8_summary.json",
        "starcoder2_humaneval_rerank_logprob_k8.jsonl.timing_summary.json",
        "starcoder2_mbpp_rerank_logprob_k8.jsonl.timing_summary.json",
    ),
    (
        "+ Locate-AR-Rewrite",
        "starcoder2_humaneval_locate_ar_rewrite_t0.9_summary.json",
        "starcoder2_mbpp_locate_ar_rewrite_t0.9_summary.json",
        "starcoder2_humaneval_locate_ar_rewrite_t0.9.jsonl.timing_summary.json",
        "starcoder2_mbpp_locate_ar_rewrite_t0.9.jsonl.timing_summary.json",
    ),
    (
        "+ LLaDA remask τ=0.9",
        "starcoder2_llada_remask_humaneval_t0.9_summary.json",
        "starcoder2_llada_remask_mbpp_t0.9_summary.json",
        "starcoder2_llada_remask_humaneval_t0.9.jsonl.timing_summary.json",
        "starcoder2_llada_remask_mbpp_t0.9.jsonl.timing_summary.json",
    ),
    (
        "+ Dream remask τ=0.9 (ours)",
        "starcoder2_dream_remask_humaneval_t0.9_summary.json",
        "starcoder2_dream_remask_mbpp_t0.9_summary.json",
        "starcoder2_dream_remask_humaneval_t0.9.jsonl.timing_summary.json",
        "starcoder2_dream_remask_mbpp_t0.9.jsonl.timing_summary.json",
    ),
]


def section_table4_starcoder2_baselines(out: list[str]) -> None:
    out.append("## Table 4d — StarCoder2 7B Baselines（pass@1 plus%）\n")
    headers = [
        "方法",
        "HE+ plus%", "HE+ base%",
        "MBPP+ plus%", "MBPP+ base%",
        "s/sample (HE)", "s/sample (MBPP)",
    ]
    out.append(_fmt_row(*headers))
    out.append(_hr(len(headers)))

    for (label, he_f, mbpp_f, t_he_f, t_mb_f) in _STARCODER2_BASELINE_ENTRIES:
        he = _load_evalplus_summary(OUTPUTS / he_f if he_f else None)
        mb = _load_evalplus_summary(OUTPUTS / mbpp_f if mbpp_f else None)
        n_he = he["n_tasks"] if he else None
        n_mb = mb["n_tasks"] if mb else None

        t_he = _load_timing(OUTPUTS / t_he_f if t_he_f else None, n_he)
        t_mb = _load_timing(OUTPUTS / t_mb_f if t_mb_f else None, n_mb)

        out.append(_fmt_row(
            label,
            _pct(he["plus_pct"] if he else None),
            _pct(he["base_pct"] if he else None),
            _pct(mb["plus_pct"] if mb else None),
            _pct(mb["base_pct"] if mb else None),
            _sps(t_he), _sps(t_mb),
        ))
    out.append("")
    out.append("> s/sample = 方法总耗时 / 题目数。\n")
```

### D2：在 `main()` 函数中注册两个新 section（约第 900 行）

在 `section_table4_qwen_baselines(lines)` 这行之后插入：

```python
    section_table4_llama31_baselines(lines)
    section_table4_starcoder2_baselines(lines)
```

最终调用顺序应为：

```python
section_standalone(lines)
section_table3_model_pairs(lines)
section_table4_baselines(lines)
section_locator_ablation(lines)
section_table4_qwen_baselines(lines)
section_table4_llama31_baselines(lines)     # 新增
section_table4_starcoder2_baselines(lines)  # 新增
section_math(lines)
section_tau_sweep(lines)
section_table2_extended(lines)
```

---

## Part E：验收与最终生成

```bash
cd /model/tteng/CoCoder
PYTHONPATH=src python -m coder.scripts.gen_results_table
```

检查 `docs/results.md`：
- 出现 `## Table 4c — Llama-3.1 8B Baselines` 段落，7 行（含 baseline 行）
- 出现 `## Table 4d — StarCoder2 7B Baselines` 段落，7 行（含 baseline 行）
- 已完成的 baseline（Dream/LLaDA remask）显示实际数值，新跑的方法跑完后也填入

---

## 验收标准

| 检查项 | 期望 |
|---|---|
| `llama31_humaneval_selfrefine_r1_summary.json` n_tasks | 164 |
| `llama31_mbpp_selfrefine_r1_summary.json` n_tasks | 378 |
| `llama31_humaneval_reflexion_feedback_r1_summary.json` n_tasks | 164 |
| `llama31_mbpp_reflexion_feedback_r1_summary.json` n_tasks | 378 |
| `llama31_humaneval_rerank_logprob_k8_summary.json` n_tasks | 164 |
| `llama31_mbpp_rerank_logprob_k8_summary.json` n_tasks | 378 |
| `llama31_humaneval_locate_ar_rewrite_t0.9_summary.json` n_tasks | 164 |
| `llama31_mbpp_locate_ar_rewrite_t0.9_summary.json` n_tasks | 378 |
| `starcoder2_humaneval_selfrefine_r1_summary.json` n_tasks | 164 |
| `starcoder2_mbpp_selfrefine_r1_summary.json` n_tasks | 378 |
| `starcoder2_humaneval_reflexion_feedback_r1_summary.json` n_tasks | 164 |
| `starcoder2_mbpp_reflexion_feedback_r1_summary.json` n_tasks | 378 |
| `starcoder2_humaneval_rerank_logprob_k8_summary.json` n_tasks | 164 |
| `starcoder2_mbpp_rerank_logprob_k8_summary.json` n_tasks | 378 |
| `starcoder2_humaneval_locate_ar_rewrite_t0.9_summary.json` n_tasks | 164 |
| `starcoder2_mbpp_locate_ar_rewrite_t0.9_summary.json` n_tasks | 378 |
| Table 4c 行数 | 7（含 baseline 行） |
| Table 4d 行数 | 7（含 baseline 行） |

---

## 注意事项

1. **Locate-AR-Rewrite 双模型显存**：Dream locator（~15GB bfloat16）+ AR model（~14GB）可能超单卡 A100 80G。
   若 OOM，改为：`--locator_device cuda:0 --ar_device cuda:1`，并 `CUDA_VISIBLE_DEVICES=<GPU0>,<GPU1>`。

2. **Reflexion 输入用 sanitized JSONL**：`-sanitized.jsonl` 里包含 `eval.error` 字段（sanitizer 执行错误信息），
   用 `--feedback_key eval.error` 注入 Reflexion 反思 prompt。
   若 sanitized 文件不存在，先跑 `python -m coder.scripts.postprocess_evalplus --input ... --out ...`。

3. **StarCoder2 自我精炼（Self-Refine）效果预期很差**：StarCoder2 7B 是填充模型，instruct 跟随能力弱，
   self-refine 几乎不会提升，甚至可能下降。这是合理的负面结果，保留在表中。

4. **Dream remask timing**：llama31 和 starcoder2 的 Dream/LLaDA remask timing 文件已存在，
   直接复用 `*_t0.9.jsonl.timing_summary.json`，无需重跑。

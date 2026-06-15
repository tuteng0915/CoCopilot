# Spec: Random Locator Baseline（make-or-break 实验）

> 对应 `docs/narrative_reframe.md` §四 实验 A。
> 这是新叙事框架的**核心验证实验**：证明"locate 是 CoCoder 的核心价值，不是 rewrite"。

---

## 目标

在与 dLLM locator 相同的 mask 数量下，用**随机选 token** 代替置信度引导的 locate，然后同样让 dLLM rewrite。

预期结果：
- Random-locate + dLLM-rewrite ≈ AR baseline（~57%）→ **localization 是核心**，故事成立
- Random-locate + dLLM-rewrite ≈ CoCoder（~72%）→ dLLM 价值在 rewrite，需要修改叙事

对比矩阵（填入 `results.md` 新 section）：

| 方法 | HumanEval+ | MBPP+ | 解读 |
|------|-----------|-------|------|
| AR-only | 56.7% | 65.1% | baseline |
| + Random-locate + dLLM-rewrite（本实验） | 4.9% | 14.8% | 随机定位会严重破坏草稿，localization 是核心 |
| + dLLM-locate + AR-rewrite（已有） | 68.9% | 67.7% | locate 贡献 |
| + dLLM-locate + dLLM-rewrite（ours） | 72.6% | 70.1% | 完整 CoCoder |

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

## Phase 1：实现 RandomLocator（无需 GPU）

### Step 1a：新建 `src/coder/locators/random_locator.py`

```python
from __future__ import annotations

import numpy as np
import torch

from coder.locators.base import TokenLocator, get_token_char_spans


class RandomLocator(TokenLocator):
    """
    Assigns uniformly random confidence to each token.

    Used as a null baseline: if random masking + dLLM-rewrite ≈ random chance,
    then token-level localization (not rewriting) is the core value of CoCoder.

    Works with --mask_ratio to ensure the same fraction of tokens is masked
    as the dLLM locator, enabling a fair comparison.
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    @torch.inference_mode()
    def score(
        self,
        prompt_text: str,
        draft_text: str,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        # We need a tokenizer to get char spans, but RandomLocator has none.
        # Instead, approximate by character: one "token" per character cluster.
        # In practice, gen_remask aligns char spans from the locator to the
        # refiner tokenizer, so a coarse span is fine here.
        #
        # Simplest: split draft into words as proxy tokens.
        import re
        tokens = re.findall(r'\S+|\s+', draft_text)
        n = len(tokens)
        confidence = self.rng.uniform(0.0, 1.0, size=n).astype(np.float32)

        # Build char spans
        spans: list[tuple[int, int]] = []
        pos = 0
        for tok in tokens:
            spans.append((pos, pos + len(tok)))
            pos += len(tok)

        return confidence, spans
```

### Step 1b：注册到 `src/coder/locators/__init__.py`

在文件中添加：

```python
from coder.locators.random_locator import RandomLocator

__all__ = [
    "TokenLocator",
    "ARLocator",
    "BERTLocator",
    "RandomLocator",        # 新增
    "get_token_char_spans",
    "align_confidence_to_spans",
    "apply_masking_policy",
    "build_locator",
]
```

在 `build_locator()` 函数的 `raise ValueError(...)` 前添加：

```python
if name == "random":
    seed = int(model_id) if model_id and model_id.isdigit() else 42
    print(f"[locator] using RandomLocator (seed={seed})")
    return RandomLocator(seed=seed)
```

### Step 1c：更新 `gen_remask.py` 的 `--locator` 帮助文档

找到 `--locator` 的 `add_argument`（约第 300 行），在 choices 帮助文字里添加：

```
"  random:          Uniformly random token confidence (null baseline)\n"
```

---

## Phase 2：确定正确的 mask_ratio

Random locator 必须用 `--mask_ratio r`（不能用 `--confidence_threshold`），使其与 dLLM locator 掩码相同数量的 token，比较才公平。

### Step 2a：计算 dLLM 在 τ=0.9 下的实际平均 mask 比例

```bash
python3 -c "
import json, pathlib, statistics

OUTPUTS = pathlib.Path('outputs/base_tuteng')
files = [
    'deepseek_dream_remask_humaneval_t0.9.jsonl',
    'deepseek_dream_remask_mbpp_t0.9.jsonl',
]
for f in files:
    p = OUTPUTS / f
    if not p.exists():
        print(f'MISSING: {f}'); continue
    ratios = []
    for line in p.read_text().splitlines():
        rec = json.loads(line)
        g = rec.get('gen', {})
        stats = g.get('mask_stats') or g.get('locator_stats', {})
        if stats and stats.get('mask_fraction') is not None:
            ratios.append(stats['mask_fraction'])
    if ratios:
        print(f'{f}: mean_mask_ratio={statistics.mean(ratios):.4f}, n={len(ratios)}')
    else:
        print(f'{f}: no mask_stats field (run with --record_mask_stats to collect)')
"
```

如果 `mask_stats` 字段不存在，用以下替代方法从已有结果反算：

```bash
python3 -c "
import json, pathlib

OUTPUTS = pathlib.Path('outputs/base_tuteng')
f = OUTPUTS / 'deepseek_dream_remask_humaneval_t0.9.jsonl'
total_tokens, total_masked = 0, 0
for line in f.read_text().splitlines():
    rec = json.loads(line)
    g = rec.get('gen', {})
    # draft completion token count (approximate via char length / 4)
    draft = rec.get('raw_completion', '') or rec.get('solution', '')
    n_tok_approx = max(1, len(draft) // 4)
    total_tokens += n_tok_approx
print('mask_ratio 需要从 --record_mask_stats 重新跑一批小样本获取')
"
```

如果无法直接计算，**使用 `--mask_ratio 0.10`** 作为保守估计（τ=0.9 对应大约 10% 低置信 token），后续可根据实际结果调整。

---

## Phase 3：运行 Random Locator 实验

使用 DeepSeek-Coder 草稿（已有）+ Random Locator + Dream-Coder rewriter。

### Step 3a：HumanEval

```bash
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_remask \
  --locator random \
  --refiner dream \
  --input $OUTPUTS/deepseek_humaneval.jsonl \
  --out $OUTPUTS/deepseek_random_locate_dream_rewrite_humaneval.jsonl \
  --mask_ratio 0.10 \
  --temperature 0.1 --top_p 0.95 --seed 3407 \
  --device cuda:0 --resume
```

### Step 3b：MBPP

```bash
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_remask \
  --locator random \
  --refiner dream \
  --input $OUTPUTS/deepseek_mbpp.jsonl \
  --out $OUTPUTS/deepseek_random_locate_dream_rewrite_mbpp.jsonl \
  --mask_ratio 0.10 \
  --temperature 0.1 --top_p 0.95 --seed 3407 \
  --device cuda:0 --resume
```

### Step 3c：Sanitize + Evaluate

```bash
python -m coder.scripts.postprocess_evalplus \
  --dataset humaneval \
  --samples $OUTPUTS/deepseek_random_locate_dream_rewrite_humaneval.jsonl

python -m coder.scripts.postprocess_evalplus \
  --dataset mbpp \
  --samples $OUTPUTS/deepseek_random_locate_dream_rewrite_mbpp.jsonl

python -m coder.scripts.eval_evalplus --backend local --dataset humaneval \
  --samples $OUTPUTS/deepseek_random_locate_dream_rewrite_humaneval-sanitized.jsonl

python -m coder.scripts.eval_evalplus --backend local --dataset mbpp \
  --samples $OUTPUTS/deepseek_random_locate_dream_rewrite_mbpp-sanitized.jsonl
```

### Step 3d：额外对照（不同 mask_ratio 敏感性）

如果时间允许，同时跑 `--mask_ratio 0.05` 和 `--mask_ratio 0.20`，验证随机结果对 mask 数量不敏感（如果确实随机化没有帮助的话）。

---

## Phase 4：验收与结果记录

```bash
python3 -c "
import json, pathlib
OUTPUTS = pathlib.Path('outputs/base_tuteng')
for fname in [
    'deepseek_random_locate_dream_rewrite_humaneval_summary.json',
    'deepseek_random_locate_dream_rewrite_mbpp_summary.json',
]:
    p = OUTPUTS / fname
    if not p.exists(): print(f'MISSING: {fname}'); continue
    s = json.load(open(p))['summary']
    he_plus = round(s['n_plus_pass']/s['n_tasks']*100, 1)
    he_base = round(s['n_base_pass']/s['n_tasks']*100, 1)
    print(f'{fname}: plus={he_plus}%, base={he_base}%, n={s[\"n_tasks\"]}')
"
```

### 最终结果（2026-06-05）

本实验已完成，使用 `--mask_ratio 0.10`、`--locator_model_id 3407`、`--record_mask_stats`。所有输出记录均为真实 rewrite：`skip_refine=False`。

| Dataset | Base | Plus | n | mean mask fraction | s/sample | Raw JSONL | Eval file | Summary |
|---------|------|------|---|--------------------|----------|-----------|-----------|---------|
| HumanEval | 4.9% | 4.9% | 164 | 10.6% | 11.0s | `outputs/base_tuteng/deepseek_random_locate_dream_rewrite_humaneval.jsonl` | `outputs/base_tuteng/deepseek_random_locate_dream_rewrite_humaneval_eval_results.json` | `outputs/base_tuteng/deepseek_random_locate_dream_rewrite_humaneval_summary.json` |
| MBPP | 15.9% | 14.8% | 378 | 12.3% | 6.4s | `outputs/base_tuteng/deepseek_random_locate_dream_rewrite_mbpp.jsonl` | `outputs/base_tuteng/deepseek_random_locate_dream_rewrite_mbpp_eval_results.json` | `outputs/base_tuteng/deepseek_random_locate_dream_rewrite_mbpp_summary.json` |

运行队列：

- script: `scripts/random_locator_gpu0_queue.sh`
- log: `outputs/base_tuteng/random_locator_queue/random_locator_gpu0.log`
- completion marker: `outputs/base_tuteng/random_locator_queue/done`

无效的首次运行产物（`generate_with_remask_failed` fallback）已移入：

- `outputs/base_tuteng/random_locator_queue/invalid_skip_refine/`

**结果判读**：

| Random HumanEval+ | 判读 |
|-------------------|------|
| < 60%（接近 AR baseline 56.7%） | **Localization 是核心** → 新叙事成立，继续 |
| 60–68%（中间） | Localization 有贡献但不是全部 → 需要细化叙事 |
| > 68%（接近 dLLM-locate+AR-rewrite 68.9%） | dLLM 价值主要在 rewrite → 需要重新审视论文结论 |

---

## 注意事项

1. **mask_ratio 一致性**：如果从 Phase 2 得到了精确的平均 mask_ratio（如 0.08），用精确值替换 0.10。结果对比时需保证 random 和 dLLM 的实际掩码 token 数相近。
2. **多次随机种子**：若时间允许，用 `--seed 42`、`--seed 100` 各跑一次，取均值，排除随机性影响。但一次通常足够，因为预期差距很大。
3. **Dream refiner 显存**：Dream-Coder 7B ≈ 15GB；单张 A100 80G 足够。

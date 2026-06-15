# Spec: Math-to-Code Pipeline（扩展版 v2）

> v1 验证了 feasibility：math-to-code 在 GSM8K 上有小幅正收益（+0.5–1.3pp），MATH-500 DeepSeek 为 0pp。
> v2 目标：补全缺失 run、验证机制、做难度分层分析，使结果足够写进正文 §5 / §7。

---

## 当前进展（已完成）

| 模型 | 数据集 | AR (code) | CoCoder | Δ |
|------|--------|-----------|---------|---|
| DeepSeek-Coder 6.7B | GSM8K | 61.0% | 62.3% | **+1.3pp** |
| Qwen2.5-Coder 7B | GSM8K | 81.0% | 81.5% | **+0.5pp** |
| Llama-3.1 8B | GSM8K | 74.8% | 75.8% | **+1.0pp** |
| DeepSeek-Coder 6.7B | MATH-500 | 6.4% | 6.4% | 0pp |

**对比基准（text CoT）**：DeepSeek GSM8K text 19.0%，code 61.0%（code mode 本身就显著更好）；CoCoder 在 text 上 ≈0pp，code 上 +1.3pp。

**产物路径**：`outputs/math_code/`

---

## Phase 1：补全缺失 Run（高优先级）

### 1a. MATH-500（Qwen / Llama31）

```bash
# AR baseline
CUDA_VISIBLE_DEVICES=0 python -m coder.scripts.gen_math_code \
  --model qwen --dataset math500 \
  --out outputs/math_code/qwen_math500_code.jsonl --max_new_tokens 512

CUDA_VISIBLE_DEVICES=0 python -m coder.scripts.gen_math_code \
  --model llama31 --dataset math500 \
  --out outputs/math_code/llama31_math500_code.jsonl --max_new_tokens 512

# CoCoder
CUDA_VISIBLE_DEVICES=1 python -m coder.scripts.gen_remask \
  --ar_outputs outputs/math_code/qwen_math500_code.jsonl \
  --refiner dream --confidence_threshold 0.9 \
  --out outputs/math_code/qwen_math500_code_dream_t0.9.jsonl --dataset math

CUDA_VISIBLE_DEVICES=1 python -m coder.scripts.gen_remask \
  --ar_outputs outputs/math_code/llama31_math500_code.jsonl \
  --refiner dream --confidence_threshold 0.9 \
  --out outputs/math_code/llama31_math500_code_dream_t0.9.jsonl --dataset math

# Eval
python -m coder.scripts.eval_math_code \
  --input outputs/math_code/qwen_math500_code.jsonl \
  --out   outputs/math_code/qwen_math500_code_eval.json
python -m coder.scripts.eval_math_code \
  --input outputs/math_code/qwen_math500_code_dream_t0.9.jsonl \
  --out   outputs/math_code/qwen_math500_code_dream_t0.9_eval.json
python -m coder.scripts.eval_math_code \
  --input outputs/math_code/llama31_math500_code.jsonl \
  --out   outputs/math_code/llama31_math500_code_eval.json
python -m coder.scripts.eval_math_code \
  --input outputs/math_code/llama31_math500_code_dream_t0.9.jsonl \
  --out   outputs/math_code/llama31_math500_code_dream_t0.9_eval.json
```

### 1b. AIME 2022–2024（3 模型）

```bash
for MODEL in deepseek qwen llama31; do
  CUDA_VISIBLE_DEVICES=0 python -m coder.scripts.gen_math_code \
    --model $MODEL --dataset aime \
    --out outputs/math_code/${MODEL}_aime_code.jsonl --max_new_tokens 512

  CUDA_VISIBLE_DEVICES=1 python -m coder.scripts.gen_remask \
    --ar_outputs outputs/math_code/${MODEL}_aime_code.jsonl \
    --refiner dream --confidence_threshold 0.9 \
    --out outputs/math_code/${MODEL}_aime_code_dream_t0.9.jsonl --dataset math

  python -m coder.scripts.eval_math_code \
    --input outputs/math_code/${MODEL}_aime_code.jsonl \
    --out   outputs/math_code/${MODEL}_aime_code_eval.json
  python -m coder.scripts.eval_math_code \
    --input outputs/math_code/${MODEL}_aime_code_dream_t0.9.jsonl \
    --out   outputs/math_code/${MODEL}_aime_code_dream_t0.9_eval.json
done
```

### 1c. AIME 2025（3 模型）

```bash
for MODEL in deepseek qwen llama31; do
  CUDA_VISIBLE_DEVICES=0 python -m coder.scripts.gen_math_code \
    --model $MODEL --dataset aime2025 \
    --out outputs/math_code/${MODEL}_aime2025_code.jsonl --max_new_tokens 512

  CUDA_VISIBLE_DEVICES=1 python -m coder.scripts.gen_remask \
    --ar_outputs outputs/math_code/${MODEL}_aime2025_code.jsonl \
    --refiner dream --confidence_threshold 0.9 \
    --out outputs/math_code/${MODEL}_aime2025_code_dream_t0.9.jsonl --dataset math

  python -m coder.scripts.eval_math_code \
    --input outputs/math_code/${MODEL}_aime2025_code.jsonl \
    --out   outputs/math_code/${MODEL}_aime2025_code_eval.json
  python -m coder.scripts.eval_math_code \
    --input outputs/math_code/${MODEL}_aime2025_code_dream_t0.9.jsonl \
    --out   outputs/math_code/${MODEL}_aime2025_code_dream_t0.9_eval.json
done
```

**预期产物**：12 个新 eval.json（MATH-500×2 + AIME×3 + AIME2025×3，各含 AR/CoCoder）

---

## Phase 2：机制验证——Fault Detection Ratio on Math Code ✅ DONE

> **核心假设**：math-to-code 之所以比 text CoT 效果好，是因为 Python 代码的结构性错误比自然语言算术错误更容易被 dLLM confidence 检测到。
>
> **结果（2026-06-14）**：假设被**否定**。

| 模型 | pairs | fault conf | nonfault conf | ratio | 对比 |
|------|-------|-----------|--------------|-------|------|
| DeepSeek GSM8K (≤500ch) | 20 | 0.898 | 0.958 | **1.1×** | text CoT = 1.15× |
| Qwen GSM8K (≤60ch) | 10 | 0.807 | 0.927 | **1.15×** | |
| Llama-3.1 GSM8K (≤60ch) | 21 | 0.478 | 0.953 | **2.0×** | coding = 23–126× |

> **结论**：math code 的 fault detection ratio 与 text CoT 相同（1.1–2.0×），远低于纯代码任务（23–126×）。dLLM 无法区分 math code 中的算术/概念错误。
>
> **Note on deepseek pairs**: deepseek 的最小 diff 为 197 chars（math fix 需要整句改写），比 coding 任务的 token 级修改大得多。用 ≤500ch 才找到 20 对。
>
> **推论**：GSM8K 上 +0.5–1.3pp 的收益**不来自 locator**，应来自 rewriter 的 self-rewrite 效果（待 Phase 2b self-rewrite baseline 确认）。

### Step 2a：收集"改动样本"

从 DeepSeek GSM8K 结果中筛选 **AR 失败，CoCoder 成功** 的样本：

```python
import json, difflib, pathlib

ar_recs   = {r['id']: r for r in map(json.loads,
             pathlib.Path('outputs/math_code/deepseek_gsm8k_code.jsonl').read_text().splitlines()) if r.strip() if r}
col_recs  = {r['id']: r for r in map(json.loads,
             pathlib.Path('outputs/math_code/deepseek_gsm8k_code_dream_t0.9.jsonl').read_text().splitlines()) if r}
ar_eval   = {r['id']: r for r in json.load(open('outputs/math_code/deepseek_gsm8k_code_eval.json'))['results']}
col_eval  = {r['id']: r for r in json.load(open('outputs/math_code/deepseek_gsm8k_code_dream_t0.9_eval.json'))['results']}

corrected_pairs = []
for rid in ar_recs:
    if not ar_eval.get(rid, {}).get('correct') and col_eval.get(rid, {}).get('correct'):
        draft = ar_recs[rid].get('raw_completion', '')
        fixed = col_recs[rid].get('raw_completion', '')
        sm = difflib.SequenceMatcher(None, draft, fixed, autojunk=False)
        diff_len = sum(a1-a0 for op,a0,a1,b0,b1 in sm.get_opcodes() if op != 'equal')
        if 1 <= diff_len <= 30:
            corrected_pairs.append({'id': rid, 'draft': draft, 'fixed': fixed, 'diff_len': diff_len})

print(f'Surgical corrected pairs: {len(corrected_pairs)}')
```

### Step 2b：计算 Fault Detection Ratio（需要 GPU）

复用 `locator_scoring.py` 逻辑，或新建 `math_code_locator_ratio.py`：

```bash
CUDA_VISIBLE_DEVICES=0 python -m coder.scripts.math_code_locator_ratio \
  --ar_file    outputs/math_code/deepseek_gsm8k_code.jsonl \
  --collab_file outputs/math_code/deepseek_gsm8k_code_dream_t0.9.jsonl \
  --ar_eval    outputs/math_code/deepseek_gsm8k_code_eval.json \
  --col_eval   outputs/math_code/deepseek_gsm8k_code_dream_t0.9_eval.json \
  --locator dream \
  --out outputs/math_code/deepseek_gsm8k_code_locator_ratio.json
```

**期望输出**：

```json
{
  "n_pairs": 18,
  "fault_tokens": 22,
  "mean_fault_conf": 0.18,
  "mean_nonfault_conf": 0.91,
  "ratio": 5.1,
  "ar_ratio": 1.2
}
```

**解读阈值**：
- ratio < 3×：math code 与 text CoT 无本质区别，增益来自偶然
- ratio 3–10×：code structure 部分恢复了 dLLM 信号，增益可解释
- ratio > 10×：接近 code 任务，增益机制与代码任务相同

---

## Phase 3：MATH-500 难度分层分析

MATH-500 每道题带有 `level`（1–5）和 `type`（Algebra / Geometry 等）字段。

### Step 3a：分层统计 DeepSeek MATH-500 现有结果

```python
import json

records = json.load(open('outputs/math_code/deepseek_math500_code_eval.json'))['results']
# 加载原始数据集获取 level/type
from datasets import load_dataset
ds = load_dataset('HuggingFaceH4/MATH-500', split='test')
meta = {str(i): {'level': ds[i]['level'], 'type': ds[i]['type']} for i in range(len(ds))}

ar_results  = {r['id']: r['correct'] for r in records}
col_records = json.load(open('outputs/math_code/deepseek_math500_code_dream_t0.9_eval.json'))['results']
col_results = {r['id']: r['correct'] for r in col_records}

from collections import defaultdict
by_level = defaultdict(lambda: {'ar': [], 'col': []})
for rid, m in meta.items():
    lv = m['level']
    by_level[lv]['ar'].append(ar_results.get(rid, False))
    by_level[lv]['col'].append(col_results.get(rid, False))

for lv in sorted(by_level):
    ar_acc  = sum(by_level[lv]['ar'])  / len(by_level[lv]['ar'])
    col_acc = sum(by_level[lv]['col']) / len(by_level[lv]['col'])
    n = len(by_level[lv]['ar'])
    print(f'Level {lv}  n={n}  AR={ar_acc:.1%}  CoCoder={col_acc:.1%}  Δ={col_acc-ar_acc:+.1%}')
```

**假设**：Level 1–2（结构简单，代码逻辑错误多）→ 更高 Δ；Level 4–5（概念/推理错误）→ Δ ≈ 0

### Step 3b：对新增 Qwen / Llama31 MATH-500 结果做同样分层

---

## Phase 4：CoT vs Code 全对比表

补全后，生成完整对比表：

| 模型 | 数据集 | Text CoT AR | **Code AR** | Code CoCoder | Δ (code CoCoder vs CoT AR) |
|------|--------|------------|-------------|-------------|--------------------------|
| DeepSeek | GSM8K | 19.0% | 61.0% | 62.3% | +43.3pp |
| Qwen | GSM8K | 30.6% | 81.0% | 81.5% | +50.9pp |
| Llama-3.1 | GSM8K | 84.5% | 74.8% | 75.8% | −8.7pp |
| DeepSeek | MATH-500 | 4.6% | 6.4% | 6.4% | +1.8pp |
| Qwen | MATH-500 | 37.6% | — | — | — |
| Llama-3.1 | MATH-500 | 38.6% | — | — | — |
| DeepSeek | AIME | — | — | — | — |
| Qwen | AIME | — | — | — | — |
| Llama-3.1 | AIME | — | — | — | — |

> Llama-3.1 text CoT 84.5%（GSM8K）是因为 Llama 的 instruction-following 在 CoT 上已很强；code mode 反而弱，这与 DeepSeek/Qwen 模式不同，值得说明。

---

## Phase 5：Paper 写入计划

### §5 Experiments（新增 math-to-code 结果）

在现有 math 段落后补一段：

```
We also evaluate a \emph{code-mode} variant in which the AR model generates Python
code to solve math problems (rather than chain-of-thought text), and CoCoder refines
the code. Table~\ref{tab:math_code} shows that code-mode substantially improves the
baseline accuracy on GSM8K for DeepSeek (61.0\% vs.\ 19.0\%) and yields consistent
CoCoder gains of +0.5--1.3\,pp. Harder benchmarks (MATH-500, AIME) show near-zero
improvement, consistent with the fault-detection ratio analysis below.
```

### §7 Discussion（更新 boundary conditions 段）

在 SQL 段之后补：

```
A partial remedy exists for mathematical reasoning: translating problems into Python
code (``math-to-code'') restores some structural signal---the fault-detection ratio
rises from 1.15$\times$ (text CoT) to approximately X$\times$ (code), yielding
+0.5--1.3\,pp on GSM8K.
However, gains vanish on harder benchmarks (MATH-500 Level 4--5, AIME), where errors
are conceptual rather than syntactic: a wrong formula such as \texttt{area = l * l}
instead of \texttt{area = l * w} is locally plausible Python, even if the variable
names are visible in context.
The boundary therefore lies not at the code/text divide, but at the
structural/conceptual divide within code.
```

---

## 完成标准检查表

### Phase 1（补 run）✅ ALL DONE
- [x] MATH-500: Qwen AR + CoCoder
- [x] MATH-500: Llama31 AR + CoCoder
- [x] AIME: DeepSeek / Qwen / Llama31 AR + CoCoder
- [x] AIME-2025: DeepSeek / Qwen / Llama31 AR + CoCoder

### Phase 2a（fault detection ratio）✅ DONE
- [x] DeepSeek GSM8K ratio = 1.1× (≤500ch pairs)
- [x] Qwen GSM8K ratio = 1.15× (≤60ch)
- [x] Llama31 GSM8K ratio = 2.0× (≤60ch)
- [x] 结论：math code ratio ≈ text CoT，dLLM locator 无效于概念性错误

### Phase 2b（self-rewrite baseline）🔄 RUNNING
- [ ] deepseek self-rewrite: GSM8K / MATH-500 / AIME / AIME-2025（GPU 1）
- [ ] qwen self-rewrite: GSM8K / MATH-500 / AIME / AIME-2025（GPU 3）
- [ ] llama31 self-rewrite: GSM8K / MATH-500 / AIME / AIME-2025（GPU 4）
- [ ] 若 self-rewrite Δ ≈ CoCoder Δ → gain 来自 rewriter not locator
- [ ] 若 self-rewrite Δ ≈ 0 → gain 可能来自 dLLM rewriter 质量（非 locator 信号）

### Phase 3（难度分层）
- [ ] DeepSeek MATH-500 by level (1–5)
- [ ] Qwen / Llama31 MATH-500 by level（在 Phase 1 完成后）

### Phase 4（全表）
- [ ] 完整 CoT vs Code 对比表写入 results.md

### Phase 5（论文）
- [ ] §5 math-to-code 段落
- [ ] §7 boundary conditions 更新（填入 ratio 数值）
- [ ] tab:math_code LaTeX 表格

---

## 产物路径汇总

```
outputs/math_code/
  {model}_{dataset}_code.jsonl               # AR 草稿
  {model}_{dataset}_code_dream_t0.9.jsonl    # CoCoder 输出
  {model}_{dataset}_code_eval.json           # AR eval
  {model}_{dataset}_code_dream_t0.9_eval.json # CoCoder eval
  deepseek_gsm8k_code_locator_ratio.json     # Phase 2 fault detection ratio
```

model ∈ {deepseek, qwen, llama31}
dataset ∈ {gsm8k, math500, aime, aime2025}

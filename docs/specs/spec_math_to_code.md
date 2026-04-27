# Spec: Math-to-Code Pipeline（GSM8K / MATH-500 / AIME / AIME-2025）

## 目标

绕过 CoCoder 在直接数学推理上的局限（confidence signal 对算术错误不敏感），改为让 AR 模型生成**可执行 Python 代码**来解题，再对代码进行 remask 精炼，最后 exec 代码得到答案。

**背景**：现有 `gen_math.py` 让模型直接输出 chain-of-thought 文本，CoCoder 对此无效（GSM8K DeepSeek+Dream: -0.8pp，MATH500 -1.4pp）。根本原因是自然语言数学步骤中的算术错误在 token 分布上无法区分——但若换成 Python 代码，逻辑/算术错误会表现为结构性问题（缩进、运算符、变量名），dLLM 的 confidence signal 与代码错误的相关性更强。

## Contribution 定位

> **C2（分析贡献）**：系统研究 locator 信号在不同域的迁移性，发现 dLLM confidence 只适用于"syntactically verifiable"域（代码）。
> **C3（扩展贡献）**：通过 math→code 范式，将 CoCoder 扩展到数学推理，验证代码级 locator 可复用。

AIME 是高权威竞赛题（答案为整数 000-999），加入后强化 C3 的可信度。

## 数据集一览

| 数据集 | HuggingFace ID | 规模 | 答案类型 | 难度 |
|--------|---------------|------|---------|------|
| GSM8K | `openai/gsm8k` | 1319 | 数字 | 小学 |
| MATH-500 | `HuggingFaceH4/MATH-500` | 500 | 数学表达式 | 竞赛 |
| AIME 2022-2024 | `AI-MO/aimo-validation-aime` | 90 | 整数 0-999 | 顶级竞赛 |
| AIME 2025 | `MathArena/aime_2025` | 30 | 整数 0-999 | 顶级竞赛（最新）|

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

## 总体流程

```
问题文本
  → gen_math_code.py（AR 模型生成 Python 解题代码）
  → gen_remask.py（dLLM remask 精炼代码）
  → eval_math_code.py（exec 代码 → 提取答案 → 对比 ground truth）
```

输出格式与现有 math JSONL schema 兼容，新增 `code_solution` 字段存放 Python 代码。

---

## Phase 1：新建生成脚本 `gen_math_code.py`

**文件位置**：`src/coder/scripts/gen_math_code.py`

该脚本是 `gen_math.py` 的 code-mode 变体，主要差异在 prompt 模板和输出字段。

### Prompt 模板

**GSM8K**（答案为整数/小数，`print` 最终结果）：

```python
GSM8K_CODE_PROMPT = """\
Write a Python function `solution()` that solves the following math problem.
The function must return a single numeric value (int or float).
Do NOT use input(). Do NOT print inside the function.
Only output the function definition, no extra text.

Problem: {question}

def solution():
"""
```

**MATH-500**（答案可能是分数/表达式，用 `sympy` 或直接 return；return 值转 str 后与 ground truth 对比）：

```python
MATH500_CODE_PROMPT = """\
Write a Python function `solution()` that solves the following math problem.
The function must return the exact answer as a string or number.
You may use sympy. Do NOT use input(). Do NOT print.
Only output the function definition, no extra text.

Problem: {question}

def solution():
"""
```

### 输出 JSONL Schema

每条记录在现有 `gen_math.py` schema 基础上，`raw_completion` 存放模型生成的函数体（不含 `def solution():` 行，或含，均可，eval 脚本统一处理），新增：

```json
{
  "id": "gsm8k/0",
  "sample_id": 0,
  "question": "...",
  "prompt": "...",
  "answer_ref": "72",
  "raw_completion": "    result = 3 * 24\n    return result\n",
  "code_mode": true,
  "model": "deepseek-coder-6.7b-instruct",
  "dataset": "gsm8k",
  "gen": { ... }
}
```

### 脚本实现要点

- 直接复制 `gen_math.py` 的骨架（argparse、dataset loading、sharding、resume、timing）
- 仅替换 prompt builder 和记录的 `code_mode: true` 标记
- `--max_new_tokens` 默认改为 512（代码比 CoT 短）
- 模型列表与 `gen_math.py` 保持一致（deepseek/qwen/llama31/dream/llada 等）

### 运行命令

```bash
# ── GSM8K ──────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python -m coder.scripts.gen_math_code \
  --model deepseek --dataset gsm8k \
  --out outputs/math_code/deepseek_gsm8k_code.jsonl --max_new_tokens 512

CUDA_VISIBLE_DEVICES=0 python -m coder.scripts.gen_math_code \
  --model qwen --dataset gsm8k \
  --out outputs/math_code/qwen_gsm8k_code.jsonl

CUDA_VISIBLE_DEVICES=0 python -m coder.scripts.gen_math_code \
  --model llama31 --dataset gsm8k \
  --out outputs/math_code/llama31_gsm8k_code.jsonl

# ── MATH-500 ────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python -m coder.scripts.gen_math_code \
  --model deepseek --dataset math500 \
  --out outputs/math_code/deepseek_math500_code.jsonl --max_new_tokens 512

CUDA_VISIBLE_DEVICES=0 python -m coder.scripts.gen_math_code \
  --model qwen --dataset math500 \
  --out outputs/math_code/qwen_math500_code.jsonl

CUDA_VISIBLE_DEVICES=0 python -m coder.scripts.gen_math_code \
  --model llama31 --dataset math500 \
  --out outputs/math_code/llama31_math500_code.jsonl

# ── AIME 2022-2024（90 题，整数答案）────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python -m coder.scripts.gen_math_code \
  --model deepseek --dataset aime \
  --out outputs/math_code/deepseek_aime_code.jsonl --max_new_tokens 512

CUDA_VISIBLE_DEVICES=0 python -m coder.scripts.gen_math_code \
  --model qwen --dataset aime \
  --out outputs/math_code/qwen_aime_code.jsonl

CUDA_VISIBLE_DEVICES=0 python -m coder.scripts.gen_math_code \
  --model llama31 --dataset aime \
  --out outputs/math_code/llama31_aime_code.jsonl

# ── AIME 2025（30 题，最新竞赛）────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python -m coder.scripts.gen_math_code \
  --model deepseek --dataset aime2025 \
  --out outputs/math_code/deepseek_aime2025_code.jsonl --max_new_tokens 512

CUDA_VISIBLE_DEVICES=0 python -m coder.scripts.gen_math_code \
  --model qwen --dataset aime2025 \
  --out outputs/math_code/qwen_aime2025_code.jsonl

CUDA_VISIBLE_DEVICES=0 python -m coder.scripts.gen_math_code \
  --model llama31 --dataset aime2025 \
  --out outputs/math_code/llama31_aime2025_code.jsonl
```

产物：`outputs/math_code/<model>_<dataset>_code.jsonl`

---

## Phase 2：gen_remask.py 兼容 code_mode

**现有 `gen_remask.py` 已支持 math JSONL**（见文件头注释），但 code_mode 的差异在于：

- `raw_completion` 是 Python 代码，不含 `#### answer` 或 `\boxed{}`
- remask 应作用在函数体 token 上，不需要特殊改动——现有逻辑已按 completion token 全局 remask

**需要检查**：`gen_remask.py` 处理 math 记录时是否正确透传 `code_mode` 字段。如果没有，在 output record 中补上即可（评测脚本需要该字段区分 code/text 模式）。

### 运行命令

```bash
# 模板：<model> × <dataset>，dataset ∈ {gsm8k, math500, aime, aime2025}
CUDA_VISIBLE_DEVICES=1 python -m coder.scripts.gen_remask \
  --ar_outputs outputs/math_code/<model>_<dataset>_code.jsonl \
  --refiner dream \
  --confidence_threshold 0.9 \
  --out outputs/math_code/<model>_<dataset>_code_dream_t0.9.jsonl \
  --dataset math

# 具体示例：
CUDA_VISIBLE_DEVICES=1 python -m coder.scripts.gen_remask \
  --ar_outputs outputs/math_code/deepseek_aime_code.jsonl \
  --refiner dream --confidence_threshold 0.9 \
  --out outputs/math_code/deepseek_aime_code_dream_t0.9.jsonl \
  --dataset math

CUDA_VISIBLE_DEVICES=1 python -m coder.scripts.gen_remask \
  --ar_outputs outputs/math_code/deepseek_aime2025_code.jsonl \
  --refiner dream --confidence_threshold 0.9 \
  --out outputs/math_code/deepseek_aime2025_code_dream_t0.9.jsonl \
  --dataset math
```

---

## Phase 3：新建评测脚本 `eval_math_code.py`

**文件位置**：`src/coder/scripts/eval_math_code.py`

该脚本 exec 模型生成的 `solution()` 函数，提取返回值，与 `answer_ref` 对比。

### 核心逻辑

```python
import ast, math, signal, contextlib
from fractions import Fraction

TIMEOUT_S = 5

def exec_solution(code: str) -> str | None:
    """
    Reconstruct full function, exec in sandbox, call solution(), return str(result).
    Returns None on any exception or timeout.
    """
    # Normalize: prepend 'def solution():\n' if missing
    if not code.strip().startswith("def solution"):
        code = "def solution():\n" + code

    namespace = {}
    try:
        with time_limit(TIMEOUT_S):
            exec(compile(code, "<math_solution>", "exec"), namespace)
            result = namespace["solution"]()
            return str(result).strip()
    except Exception:
        return None


def normalize_answer(s: str) -> str:
    """Strip whitespace, remove trailing .0, normalize LaTeX fractions."""
    s = s.strip()
    # "72.0" → "72"
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return str(f)
    except ValueError:
        pass
    # LaTeX: \frac{1}{2} → "1/2" (rough)
    s = s.replace("\\frac{", "").replace("}{", "/").replace("}", "")
    return s


def answers_match(pred: str, ref: str) -> bool:
    pred_n = normalize_answer(pred)
    ref_n = normalize_answer(ref)
    if pred_n == ref_n:
        return True
    # Try numeric equality with tolerance
    try:
        return math.isclose(float(pred_n), float(ref_n), rel_tol=1e-6)
    except (ValueError, TypeError):
        return False
```

### 评测流程

```python
records = read_jsonl(args.input)
results = []
for rec in records:
    completion = rec.get("raw_completion", "")   # or "draft_completion" for remask output
    pred = exec_solution(completion)
    ref = rec["answer_ref"]
    correct = answers_match(pred, ref) if pred is not None else False
    results.append({"id": rec["id"], "correct": correct, "pred": pred, "ref": ref})

acc = sum(r["correct"] for r in results) / len(results)
print(f"accuracy: {acc:.1%}  ({sum(r['correct'] for r in results)}/{len(results)})")
```

### CLI

```bash
# 评测 AR baseline（通用模板，dataset 自动从 id 推断）
python -m coder.scripts.eval_math_code \
  --input outputs/math_code/<model>_<dataset>_code.jsonl \
  --out   outputs/math_code/<model>_<dataset>_code_eval.json

# AIME 示例
python -m coder.scripts.eval_math_code \
  --input outputs/math_code/deepseek_aime_code.jsonl \
  --out   outputs/math_code/deepseek_aime_code_eval.json

python -m coder.scripts.eval_math_code \
  --input outputs/math_code/deepseek_aime2025_code.jsonl \
  --out   outputs/math_code/deepseek_aime2025_code_eval.json

# 评测 remask 后的结果
python -m coder.scripts.eval_math_code \
  --input outputs/math_code/deepseek_aime_code_dream_t0.9.jsonl \
  --out   outputs/math_code/deepseek_aime_code_dream_t0.9_eval.json
```

`--completion_field` 默认 `raw_completion`；AIME 答案为整数，由 `infer_dataset()` 自动路由到 `answers_match_gsm8k()`（整数归一化）。

**注意**：`AI-MO/aimo-validation-aime` 的 `answer` 字段为整数字符串（如 `"365"`），加载时已规范化。如果 HuggingFace 字段名有变化（`solution` 而非 `answer`），需在 `load_aime()` 中调整提取逻辑。

---

## Phase 4：补全 results_table 集成 ✅

`gen_results_table.py` 已实现 `section_math_code()`，包含四列：

| 模型 | GSM8K acc% | MATH-500 acc% | AIME acc% | AIME-2025 acc% |
|---|---|---|---|---|
| DeepSeek-Coder 6.7B | — | — | — | — |
| Qwen2.5-Coder 7B | — | — | — | — |
| Llama-3.1 8B | — | — | — | — |
| *(CoCoder rows)* | — | — | — | — |

产物不存在时自动 fallback `—`，无需手动修改。

---

## 产物路径汇总

```
outputs/math_code/
  # AR 草稿（code mode） — dataset ∈ {gsm8k, math500, aime, aime2025}
  deepseek_{dataset}_code.jsonl
  qwen_{dataset}_code.jsonl
  llama31_{dataset}_code.jsonl

  # remask 精炼
  deepseek_{dataset}_code_dream_t0.9.jsonl
  qwen_{dataset}_code_dream_t0.9.jsonl
  llama31_{dataset}_code_dream_t0.9.jsonl

  # eval 结果
  deepseek_{dataset}_code_eval.json
  deepseek_{dataset}_code_dream_t0.9_eval.json
  ...
```

---

## 完成标准

- [x] `gen_math_code.py` 支持 gsm8k / math500 / aime / aime2025 四个数据集
- [x] `eval_math_code.py` 支持 exec 沙箱 + 四数据集类型推断 + AIME 整数答案匹配
- [x] `gen_remask.py` 对 code_mode JSONL 正确透传字段（`out_rec = dict(rec)`）
- [x] `gen_results_table.py` section_math_code() 含 GSM8K / MATH500 / AIME / AIME-2025 四列
- [ ] 实际跑 AR baseline（等待 GPU）：预期 DeepSeek GSM8K code-mode ~50-60%，AIME ~5-15%
- [ ] 跑 remask 并对比 AR baseline delta
- [ ] 验证 `AI-MO/aimo-validation-aime` 的 `answer` 字段名称（可能为 `solution`，需按实际调整 `load_aime()`）

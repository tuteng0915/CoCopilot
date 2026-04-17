# Spec: Rewriting Benchmarks（ASSET + CoEdIT）

## 概述与动机

CoCoder 的核心机制——给定草稿 → 找出低置信 token → 扩散重写——本质上就是一个 rewrite 操作。
Rewriting benchmark 是结构上最自然的泛化验证：目标只需改动局部 token，不需要长链推理，
且评测指标（SARI / BLEU）全自动，不依赖 LLM-as-judge。

| Benchmark | 规模 | 任务 | 平均长度 | 评测指标 |
|-----------|------|------|---------|---------|
| **ASSET** | 359 test × 10 refs | 句子简化 | ~20 词 | SARI（主指标）+ BLEU-4 |
| **CoEdIT** | val 1712 条 | GEC (485) / paraphrase (527) / neutralize (700) | 1~3 句 | SARI + BLEU-4 |

AR Drafter：**Llama-3.1-8B-Instruct**  
dLLM Refiner：**Dream-org/Dream-v0-Instruct-7B**（DreamGeneral）

代码已就位：
- `src/coder/scripts/gen_rewrite.py` — ASSET / CoEdIT 生成脚本
- `src/coder/models/dream_general.py` — DreamGeneral 模型类

**本 spec 中 Codex 需要完成的工作：**
1. 扩展 `gen_remask.py` 支持 `dream_general` refiner（若尚未完成，参见 `spec_research_writing_benchmarks.md` 任务 0）
2. 编写 `eval_rewrite.py`（SARI + BLEU-4，内置 SARI 实现，无需外部包）
3. 运行所有生成任务（AR baseline + DreamGeneral standalone + CoCoder pipeline）
4. 运行评测，汇报数字

---

## 环境准备

```bash
source /home/tteng/miniconda3/etc/profile.d/conda.sh
conda activate code
cd /model/tteng/CoCoder
export PYTHONPATH=src
mkdir -p outputs/rewrite
```

---

## 任务 0：确认 gen_remask.py 支持 dream_general

```bash
PYTHONPATH=src python -c "from coder.scripts.gen_remask import build_refiner; print('OK')"
```

如果输出 `OK`，跳过。否则按照 `spec_research_writing_benchmarks.md` 的任务 0 完成修改。

---

## 任务 1：编写 eval_rewrite.py

在 `src/coder/scripts/eval_rewrite.py` 中实现 SARI 和 BLEU-4 评测。

**核心要求：不依赖 easse / sacrebleu 等外部包，完全内置实现。**

### SARI 公式

SARI 衡量简化质量，对 add / keep / delete 三类操作分别打分：

```
SARI = (1/3) * (F1_add + F1_keep + P_delete)
```

对 n-gram（n=1,2,3,4）分别计算，最终取平均：
```
SARI = mean over n of SARI_n
```

其中对每个 n-gram 阶数 n：
- `add_n`：出现在输出中但不在输入中的 n-gram，与参考答案 n-gram 的交集
  - precision_add = |add_n ∩ refs_n| / |output_n \ input_n|
  - recall_add    = |add_n ∩ refs_n| / |refs_n \ input_n|
  - F1_add_n      = harmonic mean of precision_add and recall_add
- `keep_n`：出现在输入和输出中的 n-gram，与参考答案 n-gram 的交集
  - precision_keep = |keep_n ∩ refs_n| / |output_n ∩ input_n|
  - recall_keep    = |keep_n ∩ refs_n| / |refs_n ∩ input_n|
  - F1_keep_n      = harmonic mean
- `delete_n`：出现在输入中但不在输出中的 n-gram
  - precision_delete = |(input_n \ output_n) \ refs_n| / |input_n \ output_n|
  - P_delete_n       = precision_delete

参考：Xu et al. 2016 "Optimizing Statistical Machine Translation for Text Simplification".

### BLEU-4

使用标准 corpus-level BLEU-4，用 nltk.translate.bleu_score.corpus_bleu 实现即可（nltk 已安装）。

### 答案提取

从 `raw_completion` 中提取预测文本：
- 取第一个非空行（strip 后）
- 如果以 `"Simplified:"`, `"Rewritten:"` 等前缀开头，截掉前缀
- 截断到前 150 个词（防止模型复读 prompt）

### 命令行接口

```bash
python -m coder.scripts.eval_rewrite \
    --input outputs/rewrite/asset_llama31.jsonl \
    --out   outputs/rewrite/asset_llama31_eval.json
```

支持可选 `--task` 参数，按 task 分组报告（用于 CoEdIT）：
```bash
python -m coder.scripts.eval_rewrite \
    --input outputs/rewrite/coedit_llama31.jsonl \
    --out   outputs/rewrite/coedit_llama31_eval.json \
    --by_task
```

### 输出 JSON 格式

```json
{
  "dataset": "asset",
  "model": "...",
  "n_total": 359,
  "sari": 0.xx,
  "bleu4": 0.xx,
  "by_task": {
    "simplification": {"n": 359, "sari": 0.xx, "bleu4": 0.xx}
  },
  "per_item": [
    {"id": "asset/0", "sari": 0.xx, "bleu4": 0.xx, "prediction": "...", "task": "simplification"}
  ]
}
```

---

## 任务 2：运行生成任务

### 2a. ASSET — AR Baseline（Llama31）

```bash
python -m coder.scripts.gen_rewrite \
    --model llama31 \
    --dataset asset \
    --out outputs/rewrite/asset_llama31.jsonl \
    --max_new_tokens 128 \
    --temperature 0.0 \
    --seed 3407

# 验证（期望 359 行）
python3 -c "
import pathlib, json
lines = [l for l in pathlib.Path('outputs/rewrite/asset_llama31.jsonl').read_text().splitlines() if l.strip()]
print(f'records: {len(lines)}  (expected 359)')
rec = json.loads(lines[0])
print('original:', rec['original'][:80])
print('completion:', rec['raw_completion'][:80])
"
```

### 2b. ASSET — DreamGeneral Standalone

```bash
python -m coder.scripts.gen_rewrite \
    --model dream_general \
    --dataset asset \
    --out outputs/rewrite/asset_dream_general.jsonl \
    --max_new_tokens 128 \
    --temperature 0.1 \
    --seed 3407

# 验证（期望 359 行）
python3 -c "
import pathlib
lines = [l for l in pathlib.Path('outputs/rewrite/asset_dream_general.jsonl').read_text().splitlines() if l.strip()]
print(f'records: {len(lines)}  (expected 359)')
"
```

### 2c. ASSET — CoCoder pipeline（τ=0.9）

前置条件：2a 完成。

```bash
python -m coder.scripts.gen_remask \
    --input  outputs/rewrite/asset_llama31.jsonl \
    --out    outputs/rewrite/asset_llama31_dream_general_t0.9.jsonl \
    --refiner dream_general \
    --model_id Dream-org/Dream-v0-Instruct-7B \
    --confidence_threshold 0.9 \
    --temperature 0.1 --top_p 0.95 --seed 3407

# 验证（期望 359 行）
python3 -c "
import pathlib
lines = [l for l in pathlib.Path('outputs/rewrite/asset_llama31_dream_general_t0.9.jsonl').read_text().splitlines() if l.strip()]
print(f'records: {len(lines)}  (expected 359)')
"
```

### 2d. CoEdIT — AR Baseline（Llama31，GEC + paraphrase，共 1012 条）

```bash
python -m coder.scripts.gen_rewrite \
    --model llama31 \
    --dataset coedit \
    --tasks_filter gec,paraphrase \
    --out outputs/rewrite/coedit_llama31.jsonl \
    --max_new_tokens 128 \
    --temperature 0.0 \
    --seed 3407

# 验证（期望 1012 行：gec 485 + paraphrase 527）
python3 -c "
import pathlib, json
from collections import Counter
lines = [l for l in pathlib.Path('outputs/rewrite/coedit_llama31.jsonl').read_text().splitlines() if l.strip()]
print(f'records: {len(lines)}  (expected 1012)')
tasks = Counter(json.loads(l)['task'] for l in lines)
print('task breakdown:', dict(tasks))
"
```

### 2e. CoEdIT — DreamGeneral Standalone

```bash
python -m coder.scripts.gen_rewrite \
    --model dream_general \
    --dataset coedit \
    --tasks_filter gec,paraphrase \
    --out outputs/rewrite/coedit_dream_general.jsonl \
    --max_new_tokens 128 \
    --temperature 0.1 \
    --seed 3407

# 验证（期望 1012 行）
python3 -c "
import pathlib
lines = [l for l in pathlib.Path('outputs/rewrite/coedit_dream_general.jsonl').read_text().splitlines() if l.strip()]
print(f'records: {len(lines)}  (expected 1012)')
"
```

### 2f. CoEdIT — CoCoder pipeline（τ=0.9）

前置条件：2d 完成。

```bash
python -m coder.scripts.gen_remask \
    --input  outputs/rewrite/coedit_llama31.jsonl \
    --out    outputs/rewrite/coedit_llama31_dream_general_t0.9.jsonl \
    --refiner dream_general \
    --model_id Dream-org/Dream-v0-Instruct-7B \
    --confidence_threshold 0.9 \
    --temperature 0.1 --top_p 0.95 --seed 3407

# 验证（期望 1012 行）
python3 -c "
import pathlib
lines = [l for l in pathlib.Path('outputs/rewrite/coedit_llama31_dream_general_t0.9.jsonl').read_text().splitlines() if l.strip()]
print(f'records: {len(lines)}  (expected 1012)')
"
```

---

## 任务 3：评测

### 3a. ASSET 评测（3 个产物）

```bash
python -m coder.scripts.eval_rewrite \
    --input outputs/rewrite/asset_llama31.jsonl \
    --out   outputs/rewrite/asset_llama31_eval.json

python -m coder.scripts.eval_rewrite \
    --input outputs/rewrite/asset_dream_general.jsonl \
    --out   outputs/rewrite/asset_dream_general_eval.json

python -m coder.scripts.eval_rewrite \
    --input outputs/rewrite/asset_llama31_dream_general_t0.9.jsonl \
    --out   outputs/rewrite/asset_cocoder_eval.json
```

### 3b. CoEdIT 评测（3 个产物，按 task 分组）

```bash
python -m coder.scripts.eval_rewrite \
    --input outputs/rewrite/coedit_llama31.jsonl \
    --out   outputs/rewrite/coedit_llama31_eval.json \
    --by_task

python -m coder.scripts.eval_rewrite \
    --input outputs/rewrite/coedit_dream_general.jsonl \
    --out   outputs/rewrite/coedit_dream_general_eval.json \
    --by_task

python -m coder.scripts.eval_rewrite \
    --input outputs/rewrite/coedit_llama31_dream_general_t0.9.jsonl \
    --out   outputs/rewrite/coedit_cocoder_eval.json \
    --by_task
```

### 3c. 汇总结果

```bash
python3 -c "
import json, pathlib

print('=== ASSET (SARI / BLEU-4) ===')
for label, path in [
    ('AR (Llama31)  ', 'outputs/rewrite/asset_llama31_eval.json'),
    ('DreamGeneral  ', 'outputs/rewrite/asset_dream_general_eval.json'),
    ('CoCoder t=0.9 ', 'outputs/rewrite/asset_cocoder_eval.json'),
]:
    try:
        d = json.loads(pathlib.Path(path).read_text())
        print(f'  {label}: SARI={d[\"sari\"]:.2f}  BLEU={d[\"bleu4\"]:.2f}')
    except FileNotFoundError:
        print(f'  {label}: NOT FOUND')

print()
print('=== CoEdIT GEC (SARI / BLEU-4) ===')
for label, path in [
    ('AR (Llama31)  ', 'outputs/rewrite/coedit_llama31_eval.json'),
    ('DreamGeneral  ', 'outputs/rewrite/coedit_dream_general_eval.json'),
    ('CoCoder t=0.9 ', 'outputs/rewrite/coedit_cocoder_eval.json'),
]:
    try:
        d = json.loads(pathlib.Path(path).read_text())
        gec = d.get('by_task', {}).get('gec', {})
        para = d.get('by_task', {}).get('paraphrase', {})
        print(f'  {label}: GEC  SARI={gec.get(\"sari\", \"?\"):.2f}  BLEU={gec.get(\"bleu4\", \"?\"):.2f}')
        print(f'  {label}: Para SARI={para.get(\"sari\", \"?\"):.2f}  BLEU={para.get(\"bleu4\", \"?\"):.2f}')
    except FileNotFoundError:
        print(f'  {label}: NOT FOUND')
"
```

---

## 完成判定

**代码任务：**
- [ ] `gen_remask.py` 已支持 `--refiner dream_general`
- [ ] `eval_rewrite.py` 编写完毕，SARI 实现通过如下单元测试：

  ```bash
  PYTHONPATH=src python3 -c "
  from coder.scripts.eval_rewrite import compute_sari
  # Identity rewrite = max keep score; test that score > 0
  src = 'The cat sat on the mat .'
  sys = 'The cat sat on the mat .'
  refs = ['The cat sat on the mat .']
  s = compute_sari(src, sys, refs)
  assert s > 0, f'SARI identity failed: {s}'
  print(f'SARI identity: {s:.4f}  OK')
  # Deletion-only rewrite
  sys2 = 'The cat sat .'
  s2 = compute_sari(src, sys2, refs)
  print(f'SARI deletion: {s2:.4f}  OK')
  "
  ```

**生成任务（6 个产物）：**
- [ ] outputs/rewrite/asset_llama31.jsonl（359 行）
- [ ] outputs/rewrite/asset_dream_general.jsonl（359 行）
- [ ] outputs/rewrite/asset_llama31_dream_general_t0.9.jsonl（359 行）
- [ ] outputs/rewrite/coedit_llama31.jsonl（1012 行）
- [ ] outputs/rewrite/coedit_dream_general.jsonl（1012 行）
- [ ] outputs/rewrite/coedit_llama31_dream_general_t0.9.jsonl（1012 行）

**评测任务（6 个 eval JSON）：**
- [ ] outputs/rewrite/asset_llama31_eval.json
- [ ] outputs/rewrite/asset_dream_general_eval.json
- [ ] outputs/rewrite/asset_cocoder_eval.json
- [ ] outputs/rewrite/coedit_llama31_eval.json
- [ ] outputs/rewrite/coedit_dream_general_eval.json
- [ ] outputs/rewrite/coedit_cocoder_eval.json

---

## 注意事项

1. **gen_remask.py 字段兼容**：`gen_rewrite.py` 的输出用 `id` + `answer_ref` 字段（与 gen_math.py 一致），gen_remask 的 `is_math_record()` 分支会匹配它。输出的 `raw_completion` 字段即为 CoCoder 的精炼结果。

2. **ASSET 10 参考答案**：`eval_rewrite.py` 计算 SARI 时应使用 `references` 字段（10 个 ref），而不只是 `answer_ref`（第 1 个 ref）。对 BLEU-4 同样使用全部 10 个 ref。

3. **CoEdIT 只用 GEC + paraphrase**：neutralize（700 条）任务政治色彩较强，参考答案单一，SARI 评测噪声大，暂时跳过。如需加入，去掉 `--tasks_filter` 限制即可。

4. **显存**：Llama31 8B（~16GB）和 DreamGeneral 7B（~14GB）不需要同时在卡上。gen_rewrite 生成完后卸载，再跑 gen_remask。单卡 24GB 足够。

5. **CoEdIT instruction 前缀**：src 字段已包含任务指令（如 "Fix grammaticality: ..."），gen_rewrite.py 直接用它作为 prompt 的一部分。eval 时 `original` 字段存的是这个完整的 src（含指令前缀），计算 SARI 时 `src` 应使用去掉指令前缀后的文本（即冒号后部分）。在 eval_rewrite.py 的 `extract_src_text` 函数中处理这个剥离逻辑。

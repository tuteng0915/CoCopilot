# Spec: Research & Writing Benchmarks + DreamGeneral dLLM

## 概述

为验证 CoCoder 在代码之外的泛化能力，本组实验引入两类新 benchmark：

| 类别 | Benchmark | 规模 | 评测方式 |
|------|-----------|------|---------|
| Deep Research | FRAMES（multi-hop QA） | 824 条 | Exact match + token F1 |
| Deep Research | HotpotQA distractor（multi-hop QA） | 前 1000 条 | Exact match + token F1 |
| Creative Writing | WildBench v2 Creative Writing | 146 条 | LLM-as-judge（逐条 checklist） |

AR Drafter 使用 **Llama-3.1-8B-Instruct**（通用模型，非代码专用）。  
dLLM Refiner 使用 **Dream-org/Dream-v0-Instruct-7B**（通用 Dream，非代码专用）。

代码已就位：
- `src/coder/models/dream_general.py` — DreamGeneral 模型类
- `src/coder/scripts/gen_research.py` — FRAMES / HotpotQA 生成脚本
- `src/coder/scripts/gen_writing.py` — WildBench Creative Writing 生成脚本

**本 spec 中 Codex 需要完成的工作：**
1. 扩展 `gen_remask.py` 以支持 `dream_general` refiner
2. 编写 `eval_research.py`（exact match + token F1）
3. 编写 `eval_writing.py`（LLM-as-judge + checklist 打分）
4. 运行所有生成任务（AR baseline + DreamGeneral standalone + CoCoder pipeline）
5. 运行评测，汇报数字

---

## 环境准备

```bash
source /home/tteng/miniconda3/etc/profile.d/conda.sh
conda activate code
cd /model/tteng/CoCoder
export PYTHONPATH=src
mkdir -p outputs/research outputs/writing
```

---

## 任务 0：扩展 gen_remask.py 支持 dream_general

修改 `src/coder/scripts/gen_remask.py`，在 `build_refiner` 函数中加入 `dream_general` 支持。

需要改动的位置：

**1. 在文件顶部 import 中加入 DreamGeneral：**
```python
from coder.models import DreamCoder, LLaDACoder, DreamGeneral
```

**2. 在 `_DEFAULT_REFINER_MODELS` 字典中加入：**
```python
"dream_general": "Dream-org/Dream-v0-Instruct-7B",
```

**3. 在 `build_refiner` 函数中加入分支：**
```python
if name == "dream_general":
    return DreamGeneral(model_id=resolved_id, device=device)
```

**4. 在 `--refiner` 的 `choices` 列表中加入 `"dream_general"`。**

修改后用以下命令验证导入不报错：
```bash
PYTHONPATH=src python -c "from coder.scripts.gen_remask import build_refiner; print('OK')"
```

---

## 任务 1：编写 eval_research.py

在 `src/coder/scripts/eval_research.py` 中实现标准 QA 评测（参考 SQuAD evaluation 惯例）。

**评测逻辑：**
- 将 `raw_completion` 的第一行（或前 50 个词）视为预测答案
- 对预测和参考答案都做归一化：lowercase、去掉冠词（a/an/the）、去掉标点
- 计算 **Exact Match**（归一化后字符串完全相等为 1，否则 0）
- 计算 **Token F1**（按词 token 计算 precision/recall/F1，取每题 max-over-candidates）

**命令行接口：**
```bash
python -m coder.scripts.eval_research \
    --input outputs/research/frames_llama31.jsonl \
    --out    outputs/research/frames_llama31_eval.json
```

**输出 JSON 格式（写入 `--out`）：**
```json
{
  "dataset": "frames",
  "model": "...",
  "n_total": 824,
  "exact_match": 0.xx,
  "token_f1": 0.xx,
  "per_item": [{"id": "frames/0", "em": 0, "f1": 0.72}, ...]
}
```

**答案提取规则（`extract_answer` 函数）：**
- 如果 `raw_completion` 中含有 `"Answer:"` 后跟内容，取其后第一行
- 否则取 `raw_completion` 的前 50 个词
- strip 后返回

---

## 任务 2：编写 eval_writing.py

在 `src/coder/scripts/eval_writing.py` 中实现 LLM-as-judge 评测。
使用项目现有的 `ApiCoder` 调用 Claude API（`claude-sonnet-4-6` 或类似），对每条 WildBench checklist 逐项打分。

**评测逻辑：**
- 对每条记录，从 `checklist` 字段取出评判标准（字符串列表）
- 对每个 checklist item，构造 prompt 问 LLM：  
  *"Given this creative writing response, does it satisfy the following criterion? Answer only YES or NO.\nCriterion: {criterion}\nResponse: {raw_completion[:2000]}"*  
- 解析回答：YES → 1，NO → 0；解析失败 → 0
- 每条记录得分 = checklist 通过率（通过数/总数）
- 整体得分 = 所有记录平均

**命令行接口：**
```bash
python -m coder.scripts.eval_writing \
    --input outputs/writing/writing_llama31.jsonl \
    --out   outputs/writing/writing_llama31_eval.json \
    --judge_model claude-sonnet-4-6
```

**输出 JSON 格式：**
```json
{
  "dataset": "wildbench_writing",
  "model": "...",
  "judge_model": "claude-sonnet-4-6",
  "n_total": 146,
  "checklist_pass_rate": 0.xx,
  "per_item": [{"id": "...", "score": 0.75, "n_criteria": 4, "n_pass": 3}, ...]
}
```

**注意：** 使用 `--judge_model api` 时通过 `ApiCoder` 调用，参考 `src/coder/models/api_coder.py` 的实现方式。API key 从环境变量 `ANTHROPIC_API_KEY` 读取，若未设置则退出并提示。

---

## 任务 3：运行生成任务

### 3a. FRAMES — AR Baseline（Llama31）

```bash
python -m coder.scripts.gen_research \
    --model llama31 \
    --dataset frames \
    --out outputs/research/frames_llama31.jsonl \
    --max_new_tokens 256 \
    --temperature 0.0 \
    --seed 3407

# 验证（期望 824 行）
python3 -c "
import pathlib, json
lines = [l for l in pathlib.Path('outputs/research/frames_llama31.jsonl').read_text().splitlines() if l.strip()]
print(f'records: {len(lines)}  (expected 824)')
print('sample answer_ref:', json.loads(lines[0])['answer_ref'])
print('sample completion:', json.loads(lines[0])['raw_completion'][:100])
"
```

### 3b. FRAMES — DreamGeneral Standalone

```bash
python -m coder.scripts.gen_research \
    --model dream_general \
    --dataset frames \
    --out outputs/research/frames_dream_general.jsonl \
    --max_new_tokens 256 \
    --temperature 0.1 \
    --seed 3407

# 验证（期望 824 行）
python3 -c "
import pathlib, json
lines = [l for l in pathlib.Path('outputs/research/frames_dream_general.jsonl').read_text().splitlines() if l.strip()]
print(f'records: {len(lines)}  (expected 824)')
"
```

### 3c. HotpotQA — AR Baseline（Llama31，前 1000 条）

```bash
python -m coder.scripts.gen_research \
    --model llama31 \
    --dataset hotpotqa \
    --limit 1000 \
    --out outputs/research/hotpotqa_llama31.jsonl \
    --max_new_tokens 128 \
    --temperature 0.0 \
    --seed 3407

# 验证（期望 1000 行）
python3 -c "
import pathlib, json
lines = [l for l in pathlib.Path('outputs/research/hotpotqa_llama31.jsonl').read_text().splitlines() if l.strip()]
print(f'records: {len(lines)}  (expected 1000)')
"
```

### 3d. HotpotQA — DreamGeneral Standalone

```bash
python -m coder.scripts.gen_research \
    --model dream_general \
    --dataset hotpotqa \
    --limit 1000 \
    --out outputs/research/hotpotqa_dream_general.jsonl \
    --max_new_tokens 128 \
    --temperature 0.1 \
    --seed 3407

# 验证（期望 1000 行）
python3 -c "
import pathlib
lines = [l for l in pathlib.Path('outputs/research/hotpotqa_dream_general.jsonl').read_text().splitlines() if l.strip()]
print(f'records: {len(lines)}  (expected 1000)')
"
```

### 3e. FRAMES — CoCoder pipeline（Llama31 draft → DreamGeneral remask, τ=0.9）

前置条件：3a（`frames_llama31.jsonl`）已完成。

```bash
python -m coder.scripts.gen_remask \
    --input  outputs/research/frames_llama31.jsonl \
    --out    outputs/research/frames_llama31_dream_general_t0.9.jsonl \
    --refiner dream_general \
    --model_id Dream-org/Dream-v0-Instruct-7B \
    --confidence_threshold 0.9 \
    --temperature 0.1 --top_p 0.95 --seed 3407

# 验证（期望 824 行）
python3 -c "
import pathlib, json
lines = [l for l in pathlib.Path('outputs/research/frames_llama31_dream_general_t0.9.jsonl').read_text().splitlines() if l.strip()]
print(f'records: {len(lines)}  (expected 824)')
print('refiner:', json.loads(lines[0])['gen'].get('refiner', '?'))
"
```

### 3f. HotpotQA — CoCoder pipeline

前置条件：3c（`hotpotqa_llama31.jsonl`）已完成。

```bash
python -m coder.scripts.gen_remask \
    --input  outputs/research/hotpotqa_llama31.jsonl \
    --out    outputs/research/hotpotqa_llama31_dream_general_t0.9.jsonl \
    --refiner dream_general \
    --model_id Dream-org/Dream-v0-Instruct-7B \
    --confidence_threshold 0.9 \
    --temperature 0.1 --top_p 0.95 --seed 3407

# 验证（期望 1000 行）
python3 -c "
import pathlib
lines = [l for l in pathlib.Path('outputs/research/hotpotqa_llama31_dream_general_t0.9.jsonl').read_text().splitlines() if l.strip()]
print(f'records: {len(lines)}  (expected 1000)')
"
```

### 3g. Creative Writing — AR Baseline（Llama31）

```bash
python -m coder.scripts.gen_writing \
    --model llama31 \
    --out outputs/writing/writing_llama31.jsonl \
    --max_new_tokens 1024 \
    --temperature 0.7 --top_p 0.95 \
    --seed 3407

# 验证（期望 146 行）
python3 -c "
import pathlib
lines = [l for l in pathlib.Path('outputs/writing/writing_llama31.jsonl').read_text().splitlines() if l.strip()]
print(f'records: {len(lines)}  (expected 146)')
"
```

### 3h. Creative Writing — DreamGeneral Standalone

```bash
python -m coder.scripts.gen_writing \
    --model dream_general \
    --out outputs/writing/writing_dream_general.jsonl \
    --max_new_tokens 1024 \
    --temperature 0.7 --top_p 0.95 \
    --seed 3407

# 验证（期望 146 行）
python3 -c "
import pathlib
lines = [l for l in pathlib.Path('outputs/writing/writing_dream_general.jsonl').read_text().splitlines() if l.strip()]
print(f'records: {len(lines)}  (expected 146)')
"
```

### 3i. Creative Writing — CoCoder pipeline

前置条件：3g（`writing_llama31.jsonl`）已完成。

```bash
python -m coder.scripts.gen_remask \
    --input  outputs/writing/writing_llama31.jsonl \
    --out    outputs/writing/writing_llama31_dream_general_t0.9.jsonl \
    --refiner dream_general \
    --model_id Dream-org/Dream-v0-Instruct-7B \
    --confidence_threshold 0.9 \
    --temperature 0.1 --top_p 0.95 --seed 3407

# 验证（期望 146 行）
python3 -c "
import pathlib
lines = [l for l in pathlib.Path('outputs/writing/writing_llama31_dream_general_t0.9.jsonl').read_text().splitlines() if l.strip()]
print(f'records: {len(lines)}  (expected 146)')
"
```

---

## 任务 4：评测

### 4a. FRAMES 评测（3 个产物）

```bash
# AR baseline
python -m coder.scripts.eval_research \
    --input outputs/research/frames_llama31.jsonl \
    --out   outputs/research/frames_llama31_eval.json

# DreamGeneral standalone
python -m coder.scripts.eval_research \
    --input outputs/research/frames_dream_general.jsonl \
    --out   outputs/research/frames_dream_general_eval.json

# CoCoder pipeline
python -m coder.scripts.eval_research \
    --input outputs/research/frames_llama31_dream_general_t0.9.jsonl \
    --out   outputs/research/frames_cocoder_eval.json
```

### 4b. HotpotQA 评测（3 个产物）

```bash
python -m coder.scripts.eval_research \
    --input outputs/research/hotpotqa_llama31.jsonl \
    --out   outputs/research/hotpotqa_llama31_eval.json

python -m coder.scripts.eval_research \
    --input outputs/research/hotpotqa_dream_general.jsonl \
    --out   outputs/research/hotpotqa_dream_general_eval.json

python -m coder.scripts.eval_research \
    --input outputs/research/hotpotqa_llama31_dream_general_t0.9.jsonl \
    --out   outputs/research/hotpotqa_cocoder_eval.json
```

### 4c. Writing 评测（3 个产物）

需要 `ANTHROPIC_API_KEY` 环境变量已设置。

```bash
python -m coder.scripts.eval_writing \
    --input outputs/writing/writing_llama31.jsonl \
    --out   outputs/writing/writing_llama31_eval.json

python -m coder.scripts.eval_writing \
    --input outputs/writing/writing_dream_general.jsonl \
    --out   outputs/writing/writing_dream_general_eval.json

python -m coder.scripts.eval_writing \
    --input outputs/writing/writing_llama31_dream_general_t0.9.jsonl \
    --out   outputs/writing/writing_cocoder_eval.json
```

### 4d. 汇总结果

```bash
python3 -c "
import json, pathlib

print('=== FRAMES (Exact Match / Token F1) ===')
for label, path in [
    ('AR (Llama31)     ', 'outputs/research/frames_llama31_eval.json'),
    ('DreamGeneral     ', 'outputs/research/frames_dream_general_eval.json'),
    ('CoCoder t=0.9    ', 'outputs/research/frames_cocoder_eval.json'),
]:
    try:
        d = json.loads(pathlib.Path(path).read_text())
        print(f'  {label}: EM={d[\"exact_match\"]:.3f}  F1={d[\"token_f1\"]:.3f}')
    except FileNotFoundError:
        print(f'  {label}: NOT FOUND')

print()
print('=== HotpotQA (Exact Match / Token F1) ===')
for label, path in [
    ('AR (Llama31)     ', 'outputs/research/hotpotqa_llama31_eval.json'),
    ('DreamGeneral     ', 'outputs/research/hotpotqa_dream_general_eval.json'),
    ('CoCoder t=0.9    ', 'outputs/research/hotpotqa_cocoder_eval.json'),
]:
    try:
        d = json.loads(pathlib.Path(path).read_text())
        print(f'  {label}: EM={d[\"exact_match\"]:.3f}  F1={d[\"token_f1\"]:.3f}')
    except FileNotFoundError:
        print(f'  {label}: NOT FOUND')

print()
print('=== WildBench Creative Writing (Checklist Pass Rate) ===')
for label, path in [
    ('AR (Llama31)     ', 'outputs/writing/writing_llama31_eval.json'),
    ('DreamGeneral     ', 'outputs/writing/writing_dream_general_eval.json'),
    ('CoCoder t=0.9    ', 'outputs/writing/writing_cocoder_eval.json'),
]:
    try:
        d = json.loads(pathlib.Path(path).read_text())
        print(f'  {label}: checklist_pass_rate={d[\"checklist_pass_rate\"]:.3f}')
    except FileNotFoundError:
        print(f'  {label}: NOT FOUND')
"
```

---

## 完成判定

**代码任务：**
- [ ] `gen_remask.py` 已支持 `--refiner dream_general`，import 验证通过
- [ ] `eval_research.py` 编写完毕，在 frames_llama31.jsonl 上能正常运行并输出 JSON
- [ ] `eval_writing.py` 编写完毕，在 writing_llama31.jsonl 上能正常运行并输出 JSON

**生成任务（共 9 个产物）：**
- [ ] frames_llama31.jsonl（824 行）
- [ ] frames_dream_general.jsonl（824 行）
- [ ] frames_llama31_dream_general_t0.9.jsonl（824 行）
- [ ] hotpotqa_llama31.jsonl（1000 行）
- [ ] hotpotqa_dream_general.jsonl（1000 行）
- [ ] hotpotqa_llama31_dream_general_t0.9.jsonl（1000 行）
- [ ] writing_llama31.jsonl（146 行）
- [ ] writing_dream_general.jsonl（146 行）
- [ ] writing_llama31_dream_general_t0.9.jsonl（146 行）

**评测任务（共 9 个 eval JSON）：**
- [ ] frames_llama31_eval.json
- [ ] frames_dream_general_eval.json
- [ ] frames_cocoder_eval.json
- [ ] hotpotqa_llama31_eval.json
- [ ] hotpotqa_dream_general_eval.json
- [ ] hotpotqa_cocoder_eval.json
- [ ] writing_llama31_eval.json
- [ ] writing_dream_general_eval.json
- [ ] writing_cocoder_eval.json

---

## 注意事项

1. **Dream-v0-Instruct-7B 权重**：若服务器未缓存，首次运行时会从 HuggingFace 自动下载（约 14GB）。确认网络可访问或权重已在 `~/.cache/huggingface/`。

2. **gen_remask.py 兼容性**：gen_remask 的输入字段检测逻辑依赖 `task_id`（EvalPlus）或 `id` + `answer_ref`（math）。research/writing 的 jsonl 使用 `id` + `answer_ref` 格式，应与 math 路径相同，但要确认输出中 `raw_completion` 字段正确写入（而非 `solution`）。若有字段不匹配，调整 `is_math_record` 或在 gen_remask 中新增 `is_qa_record` 分支。

3. **显存估算**：  
   - Llama31 8B：约 16GB  
   - DreamGeneral 7B：约 14GB  
   - CoCoder pipeline（Llama31 生成完后卸载，再加载 DreamGeneral）：单卡 24GB 足够；两者同时在卡上需 A100 40GB

4. **Writing 评测 API 费用**：146 条 × 平均 4 个 checklist item × 1 次 API call = 约 584 次 API 调用。使用 `claude-haiku-4-5-20251001` 可显著降低成本（checklist 判断是简单 yes/no，不需要强模型）。

5. **HotpotQA --limit 1000**：items 按数据集原始顺序取前 1000 条（`shuffle` 未开启）。若后续想分析题型分布（bridge/comparison），eval_research.py 可按 `type` 字段分组报告。

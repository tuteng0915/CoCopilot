# Spec: Table 3 s/sample 补全

## 目标

Table 3（Model Pairs）中以下两行的 `s/sample` 列显示 `—`，需要补跑 gen_remask 并更新 `gen_results_table.py`：

| Slug | 缺失原因 |
|------|---------|
| `qwen_dream_humaneval_t0.9` | 已有 timing_summary.json 但仅 6 条（partial run） |
| `starcoder2_dream_humaneval_t0.9` | 已有 timing_summary.json 但仅 3 条（partial run） |

已完成：`deepseek_dream_mbpp_t0.9` 的 Group E timed run 已补完，Table 3 当前显示 `10.0s`。记录见 `docs/specs/done/spec_pending_experiments.md`。

---

## 前提条件

- conda env: `code`
- 项目根目录: `/model/tteng/CoCoder`
- PYTHONPATH 需设为 `src`
- 已有 AR 草稿 JSONL（作为 `--input`）：
  - `outputs/base_tuteng/qwen_humaneval.jsonl`
  - `outputs/base_tuteng/starcoder2_humaneval.jsonl`

---

## Step 1: 补跑 qwen + dream HumanEval timed

```bash
source /home/tteng/miniconda3/etc/profile.d/conda.sh
conda activate code
cd /model/tteng/CoCoder
export PYTHONPATH=src

CUDA_VISIBLE_DEVICES=<空闲GPU> python -m coder.scripts.gen_remask \
  --refiner dream \
  --input outputs/base_tuteng/qwen_humaneval.jsonl \
  --out outputs/base_tuteng/qwen_dream_remask_humaneval_t0.9_timed.jsonl \
  --confidence_threshold 0.9 \
  --device cuda:0
```

- 预期产物：
  - `outputs/base_tuteng/qwen_dream_remask_humaneval_t0.9_timed.jsonl`（164 条）
  - `outputs/base_tuteng/qwen_dream_remask_humaneval_t0.9_timed.jsonl.timing_summary.json`（`n_records_written=164`）
- 估计耗时：约 35 分钟（qwen 置信度高，多数 token 不被 mask，dream 生成量少）

---

## Step 2: 补跑 starcoder2 + dream HumanEval timed

```bash
CUDA_VISIBLE_DEVICES=<空闲GPU> python -m coder.scripts.gen_remask \
  --refiner dream \
  --input outputs/base_tuteng/starcoder2_humaneval.jsonl \
  --out outputs/base_tuteng/starcoder2_dream_remask_humaneval_t0.9_timed.jsonl \
  --confidence_threshold 0.9 \
  --device cuda:0
```

- 预期产物：
  - `outputs/base_tuteng/starcoder2_dream_remask_humaneval_t0.9_timed.jsonl`（164 条）
  - `outputs/base_tuteng/starcoder2_dream_remask_humaneval_t0.9_timed.jsonl.timing_summary.json`（`n_records_written=164`）
- 估计耗时：约 90–120 分钟（starcoder2 置信度低，dream 需去噪更多 token）

---

## Step 3: 更新 gen_results_table.py

文件：`src/coder/scripts/gen_results_table.py`

在 `_PAIR_TIMING` dict（约第 293 行）中，已有：

```python
"qwen_dream_humaneval_t0.9":
    OUTPUTS / "qwen_dream_remask_humaneval_t0.9.jsonl.timing_summary.json",
"starcoder2_dream_humaneval_t0.9":
    OUTPUTS / "starcoder2_dream_remask_humaneval_t0.9.jsonl.timing_summary.json",
```

将这两行改为指向新的 `_timed` 产物：

```python
"qwen_dream_humaneval_t0.9":
    OUTPUTS / "qwen_dream_remask_humaneval_t0.9_timed.jsonl.timing_summary.json",
"starcoder2_dream_humaneval_t0.9":
    OUTPUTS / "starcoder2_dream_remask_humaneval_t0.9_timed.jsonl.timing_summary.json",
```

---

## Step 4: deepseek + dream MBPP timing（已完成）

Group E 已生成：
`outputs/base_tuteng/deepseek_dream_remask_mbpp_t0.9_timed.jsonl.timing_summary.json`

当前验证：

```bash
python3 -c "
import json
d = json.load(open('outputs/base_tuteng/deepseek_dream_remask_mbpp_t0.9_timed.jsonl.timing_summary.json'))
print('n_records_written:', d['n_records_written'])
print('avg s/sample:', d['timing']['total_s'] / d['n_records_written'])
"
```

输出：

- `n_records_written: 378`
- `avg s/sample: 9.986256008642533`

`_PAIR_TIMING` 中已添加：

```python
"deepseek_dream_mbpp_t0.9":
    OUTPUTS / "deepseek_dream_remask_mbpp_t0.9_timed.jsonl.timing_summary.json",
```

---

## Step 5: 验证

```bash
cd /model/tteng/CoCoder
PYTHONPATH=src python -m coder.scripts.gen_results_table
```

确认 `docs/results.md` Table 3 中相应行 `s/sample` 列不再显示 `—`。

---

## 验收标准

- `n_records_written` 在 timing_summary.json 中 = 164（humaneval）
- Table 3 qwen+dream HE 和 starcoder2+dream HE 行均有数字 s/sample
- 不覆盖已有 accuracy 结果（新输出文件名带 `_timed` 后缀）

Group E 附加验收已满足：

- `deepseek_dream_remask_mbpp_t0.9_timed.jsonl` 为 378 条、378 个唯一 task
- `deepseek_dream_remask_mbpp_t0.9_timed.jsonl.timing_summary.json` 中 `n_records_written=378`
- Table 3 `deepseek_dream_mbpp_t0.9` 行显示 `10.0s`

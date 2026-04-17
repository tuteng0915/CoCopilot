# Spec: Group D/E + LLaDA MBPP + Qwen+LLaDA 收尾记录

## 概述（2026-04-13 更新）

以下工作已完成：
- ✅ `model_pairs_evalplus.py` 添加了 qwen_llada_humaneval/mbpp、deepseek_llada_mbpp、qwen_llada_mbpp 配置
- ✅ `model_pairs_all_t0.9.json` 重新生成，包含全部 12 个 pair
- ✅ `gen_results_table.py` 的 `_PAIR_TIMING` 正确指向已有 timing 文件
- ✅ DeepSeek + Dream MBPP timed（Group E）已补完并注册进 Table 3

本 spec 当前无剩余待处理事项。Table 3 timed rerun 记录见 `docs/specs/done/spec_table3_timing.md`。

| 任务 | 状态 | 目标产物 |
|------|------|---------|
| LLaDA pair accuracy 写入 model_pairs JSON | ✅ 完成 | `outputs/base_tuteng/model_pairs_all_t0.9.json` |
| DeepSeek + Dream MBPP timing | ✅ 完成，378/378，10.0s/sample | `deepseek_dream_remask_mbpp_t0.9_timed.jsonl.timing_summary.json` |

---

## A. 将 LLaDA 结果写入 model_pairs JSON

Group D 完成后，DeepSeek+LLaDA MBPP / Qwen+LLaDA HE / Qwen+LLaDA MBPP 的 accuracy summary 已写入 `model_pairs_all_t0.9.json`（该文件由 `model_pairs_evalplus.py` 生成）。

### 确认产物存在

```bash
ls outputs/base_tuteng/deepseek_llada_remask_mbpp_t0.9_summary.json
ls outputs/base_tuteng/qwen_llada_remask_humaneval_t0.9_summary.json
ls outputs/base_tuteng/qwen_llada_remask_mbpp_t0.9_summary.json
```

当前 summary.json 使用 EvalPlus summary schema，pass@1 从 `summary.n_plus_pass / summary.n_tasks` 和 `summary.n_base_pass / summary.n_tasks` 计算，不要求顶层存在 `plus_pct` / `base_pct` 字段。

### 重新生成 model_pairs JSON

```bash
source /home/tteng/miniconda3/etc/profile.d/conda.sh
conda activate code
cd /model/tteng/CoCoder
export PYTHONPATH=src

python -m coder.scripts.model_pairs_evalplus
```

然后检查输出：

```bash
python3 -c "
import json
data = json.load(open('outputs/base_tuteng/model_pairs_all_t0.9.json'))
for r in data['rows']:
    if 'llada' in r['slug']:
        print(r['slug'], '->', r.get('collab_pass_at_1_pct'))
"
```

当前结果：
- `deepseek_llada_humaneval_t0.9 -> 65.9`
- `deepseek_llada_mbpp_t0.9 -> 68.0`
- `qwen_llada_humaneval_t0.9 -> 77.4`
- `qwen_llada_mbpp_t0.9 -> 73.0`

---

## B. model_pairs_evalplus.py 中注册 LLaDA pairs

`src/coder/scripts/model_pairs_evalplus.py` 可能需要添加 LLaDA MBPP / Qwen+LLaDA 条目。

检查该文件中是否已定义相应的 pair 配置：

```bash
grep -n "llada\|qwen.*llada\|deepseek.*llada" src/coder/scripts/model_pairs_evalplus.py
```

如果缺少 `qwen+llada` 或 `deepseek_llada_mbpp` 条目，需要按已有 pair 的格式添加。参考已有条目（如 `deepseek+llada humaneval`）：

```python
{
    "ar_drafter": "Qwen2.5-Coder 7B",
    "dllm_refiner": "LLaDA 8B",
    "dataset": "humaneval",
    "slug": "qwen_llada_humaneval_t0.9",
    "collab_jsonl": "outputs/base_tuteng/qwen_llada_remask_humaneval_t0.9.jsonl",
    "ar_summary": "outputs/base_tuteng/qwen_humaneval_summary.json",
    "collab_summary": "outputs/base_tuteng/qwen_llada_remask_humaneval_t0.9_summary.json",
},
```

---

## C. Group E 完成后的 timing 注册

Group E 已完成。当前产物：

- `outputs/base_tuteng/deepseek_dream_remask_mbpp_t0.9_timed.jsonl`
- `outputs/base_tuteng/deepseek_dream_remask_mbpp_t0.9_timed.jsonl.timing_summary.json`
- `outputs/base_tuteng/deepseek_dream_remask_mbpp_t0.9_timed.jsonl.lock` 已清理

1. 验证完整性：
   ```bash
   python3 -c "
   import json
   d = json.load(open('outputs/base_tuteng/deepseek_dream_remask_mbpp_t0.9_timed.jsonl.timing_summary.json'))
   print('n_records_written:', d['n_records_written'])  # 期望 378
   print('avg s/sample:', d['timing']['total_s'] / d['n_records_written'])
   "
   ```
   当前输出：
   - `n_records_written: 378`
   - `avg s/sample: 9.986256008642533`（Table 3 显示为 `10.0s`）

2. `src/coder/scripts/gen_results_table.py` 的 `_PAIR_TIMING` 已注册：
   ```python
   "deepseek_dream_mbpp_t0.9":
       OUTPUTS / "deepseek_dream_remask_mbpp_t0.9_timed.jsonl.timing_summary.json",
   ```

### 运行记录

本次恢复时发现 `deepseek_dream_remask_mbpp_t0.9_timed.jsonl` 有 326 条有效 JSONL，最后一行是截断 JSON。处理步骤：

- 备份原文件到 `outputs/base_tuteng/deepseek_dream_remask_mbpp_t0.9_timed.jsonl.corrupt_tail_backup_20260413_2205`
- 截掉不完整尾行，保留 326 条有效记录
- 使用 `gen_remask --resume` 补完剩余 52 条
- 从最终 378 条记录中的 `gen.timing` 重新聚合 timing summary

相关代码修复：

- `gen_remask.py --resume` 现在会聚合整份输出 timing，而不是只统计本次新增记录
- `gen_remask.py` 正常完成后会释放并删除 `.lock`
- `gen_results_table.py` 会忽略 `remask_generate_s_total <= 0` 的无效 resume-only timing

---

## D. 最终验证

所有实验完成后，重新生成 results.md：

```bash
cd /model/tteng/CoCoder
PYTHONPATH=src python -m coder.scripts.gen_results_table
```

Table 3 期望状态：

| Slug | collab pass@1 | s/sample |
|------|--------------|---------|
| deepseek_dream_humaneval | 72.6% | 14.7s ✅ |
| qwen_dream_humaneval | 76.8% | 需跑 timed |
| llama31_dream_humaneval | 57.3% | 13.3s ✅ |
| starcoder2_dream_humaneval | 23.2% | 需跑 timed |
| deepseek_llada_humaneval | 65.9% | 15.3s ✅ |
| deepseek_dream_mbpp | 70.1% | 10.0s ✅ |
| qwen_dream_mbpp | 72.2% | 8.8s ✅ |
| llama31_dream_mbpp | 64.0% | 4.5s ✅ |
| starcoder2_dream_mbpp | 33.1% | 46.0s ✅ |
| deepseek_llada_mbpp | 68.0% | —（现有 timing 为 resume-only，无效） |
| qwen_llada_humaneval | 77.4% | —（现有 timing 为 resume-only，无效） |
| qwen_llada_mbpp | 73.0% | —（现有 timing 为 resume-only，无效） |

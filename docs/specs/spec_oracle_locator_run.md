# Spec: Oracle Locator — 运行实验 + 论文集成

> 完整设计见 `done/spec_oracle_locator.md`（gen_oracle_mask.py、oracle_locator.py 均已实现）。
> 本 spec 从"代码已有"出发，描述如何运行实验并将结果写入论文。

---

## 目标

构造 Oracle Locator：精确 mask AR draft 与 CoCoder 成功修复版之间**实际改变的 token**，
然后用 Dream-Coder 重写，量化"如果 localization 完美，CoCoder 能达到多少"。

结果将补充到 `tab:locator`（`tables/tab_locator.tex`）和 `06_analysis.tex` 的分析段。

---

## 前提条件

```bash
source /home/wjzhang/miniforge3/etc/profile.d/conda.sh
conda activate cocoder
cd /home/wjzhang/tt_workspace/model/CoCoder/CoCoder
export PYTHONPATH=src
export OUTPUTS=outputs/base_tuteng
```

已有文件（需要确认存在）：

```bash
ls $OUTPUTS/deepseek_humaneval.jsonl                                     # AR draft
ls $OUTPUTS/deepseek_dream_remask_humaneval_t0.9_timed.jsonl             # CoCoder result（注意 _timed 后缀）
ls $OUTPUTS/deepseek_humaneval-sanitized_eval_results.json               # AR eval
ls $OUTPUTS/deepseek_dream_remask_humaneval_t0.9_timed-sanitized_eval_results.json  # CoCoder eval

ls $OUTPUTS/deepseek_mbpp.jsonl
ls $OUTPUTS/deepseek_dream_remask_mbpp_t0.9_timed.jsonl
ls $OUTPUTS/deepseek_mbpp-sanitized_eval_results.json
ls $OUTPUTS/deepseek_dream_remask_mbpp_t0.9_timed-sanitized_eval_results.json
```

> ⚠️ collab_input 文件名可能无 `_timed`（取决于实际跑出的文件）。先 `ls $OUTPUTS/deepseek_dream_remask_humaneval_t0.9*.jsonl` 确认。

---

## Phase 1：构造 Oracle Mask JSONL（无需 GPU，~5 分钟）

```bash
# HumanEval
python -m coder.scripts.gen_oracle_mask \
  --ar_input     $OUTPUTS/deepseek_humaneval.jsonl \
  --collab_input $OUTPUTS/deepseek_dream_remask_humaneval_t0.9_timed.jsonl \
  --ar_eval      $OUTPUTS/deepseek_humaneval-sanitized_eval_results.json \
  --collab_eval  $OUTPUTS/deepseek_dream_remask_humaneval_t0.9_timed-sanitized_eval_results.json \
  --out          $OUTPUTS/deepseek_humaneval_oracle_mask.jsonl \
  --min_diff_chars 1 \
  --max_diff_chars 500

# MBPP
python -m coder.scripts.gen_oracle_mask \
  --ar_input     $OUTPUTS/deepseek_mbpp.jsonl \
  --collab_input $OUTPUTS/deepseek_dream_remask_mbpp_t0.9_timed.jsonl \
  --ar_eval      $OUTPUTS/deepseek_mbpp-sanitized_eval_results.json \
  --collab_eval  $OUTPUTS/deepseek_dream_remask_mbpp_t0.9_timed-sanitized_eval_results.json \
  --out          $OUTPUTS/deepseek_mbpp_oracle_mask.jsonl \
  --min_diff_chars 1 \
  --max_diff_chars 500
```

期望输出（脚本末尾日志）：
```
Eligible tasks (AR fail → Collab pass): <N>
Written: <M> tasks with oracle mask spans
```

检验：
```bash
python -c "
import json
he = [json.loads(l) for l in open('$OUTPUTS/deepseek_humaneval_oracle_mask.jsonl')]
n_oracle = sum(1 for r in he if r.get('oracle_mask_spans'))
print(f'HE oracle tasks: {n_oracle} / {len(he)}')
"
```

---

## Phase 2：运行 Oracle Locator 实验（需要 GPU，~1–2 小时）

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python -m coder.scripts.gen_remask \
  --locator oracle \
  --locator_model_id $OUTPUTS/deepseek_humaneval_oracle_mask.jsonl \
  --refiner dream \
  --input    $OUTPUTS/deepseek_humaneval.jsonl \
  --out      $OUTPUTS/deepseek_oracle_locate_dream_rewrite_humaneval.jsonl \
  --temperature 0.1 --top_p 0.95 --seed 3407 \
  --device cuda:0 --resume

CUDA_VISIBLE_DEVICES=<GPU_ID> python -m coder.scripts.gen_remask \
  --locator oracle \
  --locator_model_id $OUTPUTS/deepseek_mbpp_oracle_mask.jsonl \
  --refiner dream \
  --input    $OUTPUTS/deepseek_mbpp.jsonl \
  --out      $OUTPUTS/deepseek_oracle_locate_dream_rewrite_mbpp.jsonl \
  --temperature 0.1 --top_p 0.95 --seed 3407 \
  --device cuda:0 --resume
```

> 注意：gen_remask 是否已支持 `--locator oracle` 参数？先检查 `src/coder/scripts/gen_remask.py`
> 是否有 `if name == "oracle"` 分支。如果没有，参照 `done/spec_oracle_locator.md` §Step 2b/2c 补充。

---

## Phase 3：评估（无需 GPU）

```bash
for DATASET in humaneval mbpp; do
  python -m coder.scripts.postprocess_evalplus \
    --dataset $DATASET \
    --samples $OUTPUTS/deepseek_oracle_locate_dream_rewrite_${DATASET}.jsonl

  python -m coder.scripts.eval_evalplus \
    --backend local --dataset $DATASET \
    --samples $OUTPUTS/deepseek_oracle_locate_dream_rewrite_${DATASET}-sanitized.jsonl
done
```

读取结果：
```bash
python -c "
import json
for ds in ['humaneval','mbpp']:
    f = 'outputs/base_tuteng/deepseek_oracle_locate_dream_rewrite_' + ds + '-sanitized_eval_results.json'
    d = json.load(open(f))
    n = len(d); p = sum(1 for v in d.values() if v.get('eval',{}).get('plus_status')=='pass')
    print(f'{ds}: {p}/{n} = {p/n*100:.1f}% HE+/MBPP+')
"
```

---

## Phase 4：写入论文

### 4a. 更新 `tables/tab_locator.tex`

在 `dLLM (Dream-Coder, \textbf{ours})` 行下方、`dLLM (Dream-Coder) + AR-rewrite` 行上方，添加 Oracle 行：

```latex
Oracle (gold diff positions)   & Dream-Coder & \textit{XX.X} & \textit{XX.X} & — & — \\
```

（斜体标注表示上界/参考数字，不与其他行并排比较）

### 4b. 更新 `section/06_analysis.tex`

在 "Role decomposition" 段末尾添加：

```latex
To bound the contribution of localization quality, we also evaluate an oracle locator that masks
the exact character positions where the AR draft and the CoCoder-corrected solution differ.
This oracle achieves XX.X\% HumanEval+ and XX.X\% MBPP+ (italicized in Table~\ref{tab:locator}),
compared to 72.6\% / 70.1\% for the dLLM locator.
The X.X\,pp gap indicates that while dLLM confidence provides a strong localization signal,
[imprecise localization / rewriter quality] is still [the main / a secondary] bottleneck.
```

（具体措辞取决于实验结果：oracle >> dLLM ⟹ localization 是主瓶颈；oracle ≈ dLLM ⟹ rewriter 是瓶颈）

---

## 验收标准

| 检查项 | 期望 |
|--------|------|
| `deepseek_humaneval_oracle_mask.jsonl` 存在 | oracle_mask_spans non-null 的行 > 0 |
| `deepseek_oracle_locate_dream_rewrite_humaneval-sanitized_eval_results.json` 存在 | n_tasks = 164 |
| `deepseek_oracle_locate_dream_rewrite_mbpp-sanitized_eval_results.json` 存在 | n_tasks = 378 |
| `tab_locator.tex` 有 Oracle 行 | 有具体数字 |
| `06_analysis.tex` 有 oracle 的叙述段 | 根据结果选择措辞 |

---

## 时间估计

| 阶段 | 时间 | GPU |
|------|------|-----|
| Phase 1（oracle mask JSONL） | ~5 分钟 | 无 |
| Phase 2（gen_remask × 2） | ~1–2 小时 | 1 GPU |
| Phase 3（评估） | ~10 分钟 | 无 |
| Phase 4（写论文） | ~20 分钟 | 无 |

---

## 解读框架（结果出来前预判）

| Oracle 结果 | 与 dLLM 差距 | 解读 |
|------------|-------------|------|
| ≥ 78% HE+ | > 5pp | Localization 精度是主要剩余瓶颈；改进 locator 有大空间 |
| 73–78% HE+ | 1–5pp | Localization 有改进空间，但 rewriter 也是瓶颈之一 |
| ≈ 72.6% HE+ | < 1pp | 当前 dLLM locator 已接近最优；瓶颈在 rewriter |

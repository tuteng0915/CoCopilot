# Spec: Mask 粒度消融（token / span / line）

## 背景

CoCoder 的 Locator 打完置信度分后，mask 的"边界"有三种粒度：

| 粒度 | 行为 | 典型适用场景 |
|------|------|------------|
| `token` | 只 mask 单个低置信 token（默认）| 点状错误：变量名、运算符 |
| `span` | 把相邻低置信 token 合并成连续片段 | 局部逻辑错误，减少碎片化重写 |
| `line` | 只要某行有低置信 token，整行 mask | 块级/缩进错误，结构化编辑 |

**本组实验**：固定 Drafter（DeepSeek-Coder 6.7B）、Locator（dLLM，`--confidence_threshold 0.9`）、
Rewriter（Dream-Coder 7B），只改 `--mask_granularity`。

**`token` baseline** 已有现成产物（Table 3 τ=0.9 结果），**无需重跑**。
本 spec 只需新跑 `span` 和 `line` 两种粒度，共 4 组实验（2 粒度 × 2 datasets）。

---

## 当前进度（2026-04-22）

已完成并评测：

- `span + HumanEval`
  - raw: `outputs/ablation_granularity/deepseek_dream_humaneval_t0.9_gran_span.jsonl`
  - summary: `outputs/ablation_granularity/deepseek_dream_humaneval_t0.9_gran_span_summary.json`
  - 结果：HE+ plus = `65.2%`, HE+ base = `70.7%`
- `span + MBPP`
  - raw: `outputs/ablation_granularity/deepseek_dream_mbpp_t0.9_gran_span.jsonl`
  - summary: `outputs/ablation_granularity/deepseek_dream_mbpp_t0.9_gran_span_summary.json`
  - 结果：MBPP+ plus = `69.0%`, MBPP+ base = `80.2%`
- `line + HumanEval`
  - raw: `outputs/ablation_granularity/deepseek_dream_humaneval_t0.9_gran_line.jsonl`
  - summary: `outputs/ablation_granularity/deepseek_dream_humaneval_t0.9_gran_line_summary.json`
  - 结果：HE+ plus = `45.7%`, HE+ base = `51.2%`

尚未完成：

- `line + MBPP`
  - 还未生成：`outputs/ablation_granularity/deepseek_dream_mbpp_t0.9_gran_line.jsonl` 当前不存在
  - 原因：最近几次执行时没有空闲 GPU

注意：本 spec 里的 `postprocess_evalplus` / `eval_evalplus` CLI 已在代码中更新。
下面的命令块已经按当前脚本接口修正为 `--dataset --samples` / `--output_file`。

---

## 环境准备

```bash
source /home/tteng/miniconda3/etc/profile.d/conda.sh
conda activate code
cd /model/tteng/CoCoder
export PYTHONPATH=src
```

## 前置检查

```bash
# 确认输入文件存在
ls outputs/base_tuteng/deepseek_humaneval.jsonl   # HumanEval AR 草稿（164题）
ls outputs/base_tuteng/deepseek_mbpp.jsonl        # MBPP AR 草稿（378题）

# 确认 token baseline 已有（无需重跑）
ls outputs/base_tuteng/deepseek_dream_remask_humaneval_t0.9_timed.jsonl
ls outputs/base_tuteng/deepseek_dream_remask_mbpp_t0.9_timed.jsonl

# 创建输出目录
mkdir -p outputs/ablation_granularity
```

---

## 实验命令

### 2a. Span Granularity — HumanEval

`span_merge_gap=2`：合并间距 ≤ 2 个 token 的 mask 片段（推荐值，平衡连续性与过度扩展）。

> 状态：已完成，无需重跑。

```bash
python -m coder.scripts.gen_remask \
  --input  outputs/base_tuteng/deepseek_humaneval.jsonl \
  --out    outputs/ablation_granularity/deepseek_dream_humaneval_t0.9_gran_span.jsonl \
  --locator dream \
  --confidence_threshold 0.9 \
  --mask_granularity span \
  --span_merge_gap 2 \
  --temperature 0.1 --top_p 0.95 --seed 3407

# 验证完整性（期望 164 行）
python3 -c "
import pathlib
lines = [l for l in pathlib.Path('outputs/ablation_granularity/deepseek_dream_humaneval_t0.9_gran_span.jsonl').read_text().splitlines() if l.strip()]
print(f'records: {len(lines)}  (expected 164)')
"
```

### 2b. Span Granularity — MBPP

> 状态：已完成，无需重跑。

```bash
python -m coder.scripts.gen_remask \
  --input  outputs/base_tuteng/deepseek_mbpp.jsonl \
  --out    outputs/ablation_granularity/deepseek_dream_mbpp_t0.9_gran_span.jsonl \
  --locator dream \
  --confidence_threshold 0.9 \
  --mask_granularity span \
  --span_merge_gap 2 \
  --temperature 0.1 --top_p 0.95 --seed 3407

# 验证（期望 378 行）
python3 -c "
import pathlib
lines = [l for l in pathlib.Path('outputs/ablation_granularity/deepseek_dream_mbpp_t0.9_gran_span.jsonl').read_text().splitlines() if l.strip()]
print(f'records: {len(lines)}  (expected 378)')
"
```

### 2c. Line Granularity — HumanEval

> 状态：已完成，无需重跑。

```bash
python -m coder.scripts.gen_remask \
  --input  outputs/base_tuteng/deepseek_humaneval.jsonl \
  --out    outputs/ablation_granularity/deepseek_dream_humaneval_t0.9_gran_line.jsonl \
  --locator dream \
  --confidence_threshold 0.9 \
  --mask_granularity line \
  --temperature 0.1 --top_p 0.95 --seed 3407

# 验证（期望 164 行）
python3 -c "
import pathlib
lines = [l for l in pathlib.Path('outputs/ablation_granularity/deepseek_dream_humaneval_t0.9_gran_line.jsonl').read_text().splitlines() if l.strip()]
print(f'records: {len(lines)}  (expected 164)')
"
```

### 2d. Line Granularity — MBPP

> 状态：待执行。这是当前唯一剩余的生成任务。

```bash
python -m coder.scripts.gen_remask \
  --input  outputs/base_tuteng/deepseek_mbpp.jsonl \
  --out    outputs/ablation_granularity/deepseek_dream_mbpp_t0.9_gran_line.jsonl \
  --locator dream \
  --confidence_threshold 0.9 \
  --mask_granularity line \
  --temperature 0.1 --top_p 0.95 --seed 3407 \
  --resume

# 验证（期望 378 行）
python3 -c "
import pathlib
lines = [l for l in pathlib.Path('outputs/ablation_granularity/deepseek_dream_mbpp_t0.9_gran_line.jsonl').read_text().splitlines() if l.strip()]
print(f'records: {len(lines)}  (expected 378)')
"
```

---

## Sanitize + 评测（4 个新产物各跑一次）

```bash
# span — HumanEval
python -m coder.scripts.postprocess_evalplus \
  --dataset humaneval \
  --samples outputs/ablation_granularity/deepseek_dream_humaneval_t0.9_gran_span.jsonl

python -m coder.scripts.eval_evalplus \
  --dataset humaneval \
  --samples outputs/ablation_granularity/deepseek_dream_humaneval_t0.9_gran_span-sanitized.jsonl \
  --backend local \
  --parallel 16 \
  --output_file outputs/ablation_granularity/deepseek_dream_humaneval_t0.9_gran_span-sanitized_eval_results.json \
  --summary_out outputs/ablation_granularity/deepseek_dream_humaneval_t0.9_gran_span_summary.json \
  --summary_model deepseek_dream_humaneval_t0.9_gran_span

# span — MBPP
python -m coder.scripts.postprocess_evalplus \
  --dataset mbpp \
  --samples outputs/ablation_granularity/deepseek_dream_mbpp_t0.9_gran_span.jsonl

python -m coder.scripts.eval_evalplus \
  --dataset mbpp \
  --samples outputs/ablation_granularity/deepseek_dream_mbpp_t0.9_gran_span-sanitized.jsonl \
  --backend local \
  --parallel 16 \
  --output_file outputs/ablation_granularity/deepseek_dream_mbpp_t0.9_gran_span-sanitized_eval_results.json \
  --summary_out outputs/ablation_granularity/deepseek_dream_mbpp_t0.9_gran_span_summary.json \
  --summary_model deepseek_dream_mbpp_t0.9_gran_span

# line — HumanEval
python -m coder.scripts.postprocess_evalplus \
  --dataset humaneval \
  --samples outputs/ablation_granularity/deepseek_dream_humaneval_t0.9_gran_line.jsonl

python -m coder.scripts.eval_evalplus \
  --dataset humaneval \
  --samples outputs/ablation_granularity/deepseek_dream_humaneval_t0.9_gran_line-sanitized.jsonl \
  --backend local \
  --parallel 16 \
  --output_file outputs/ablation_granularity/deepseek_dream_humaneval_t0.9_gran_line-sanitized_eval_results.json \
  --summary_out outputs/ablation_granularity/deepseek_dream_humaneval_t0.9_gran_line_summary.json \
  --summary_model deepseek_dream_humaneval_t0.9_gran_line

# line — MBPP
python -m coder.scripts.postprocess_evalplus \
  --dataset mbpp \
  --samples outputs/ablation_granularity/deepseek_dream_mbpp_t0.9_gran_line.jsonl

python -m coder.scripts.eval_evalplus \
  --dataset mbpp \
  --samples outputs/ablation_granularity/deepseek_dream_mbpp_t0.9_gran_line-sanitized.jsonl \
  --backend local \
  --parallel 16 \
  --output_file outputs/ablation_granularity/deepseek_dream_mbpp_t0.9_gran_line-sanitized_eval_results.json \
  --summary_out outputs/ablation_granularity/deepseek_dream_mbpp_t0.9_gran_line_summary.json \
  --summary_model deepseek_dream_mbpp_t0.9_gran_line
```

---

## 结果读取

```bash
python3 -c "
import json

# 读取 wrapper summary（n_both_pass / n_base_pass）
def read_summary(path):
    try:
        d = json.load(open(path))
        s = d.get('summary', {})
        n = s.get('n_tasks', 0)
        if not n:
            return 'MISSING', 'MISSING'
        plus = f\"{s['n_both_pass'] / n * 100:.1f}%\"
        base = f\"{s['n_base_pass'] / n * 100:.1f}%\"
        return plus, base
    except FileNotFoundError:
        return 'MISSING', 'MISSING'

# token baseline：来自 results.md Table 3（已知，无需重读文件）
he_tok  = ('72.6%', '78.7%')
mb_tok  = ('70.1%', '80.4%')
he_span = read_summary('outputs/ablation_granularity/deepseek_dream_humaneval_t0.9_gran_span_summary.json')
mb_span = read_summary('outputs/ablation_granularity/deepseek_dream_mbpp_t0.9_gran_span_summary.json')
he_line = read_summary('outputs/ablation_granularity/deepseek_dream_humaneval_t0.9_gran_line_summary.json')
mb_line = read_summary('outputs/ablation_granularity/deepseek_dream_mbpp_t0.9_gran_line_summary.json')

print('Granularity  | HE+ plus  HE+ base  | MBPP+ plus  MBPP+ base')
print(f'token        | {he_tok[0]:>8}  {he_tok[1]:>8}  | {mb_tok[0]:>10}  {mb_tok[1]:>10}')
print(f'span (gap=2) | {he_span[0]:>8}  {he_span[1]:>8}  | {mb_span[0]:>10}  {mb_span[1]:>10}')
print(f'line         | {he_line[0]:>8}  {he_line[1]:>8}  | {mb_line[0]:>10}  {mb_line[1]:>10}')
"
```

---

## 完成判定

- [x] span HumanEval 生成完毕（164 行）
- [x] span MBPP 生成完毕（378 行）
- [x] line HumanEval 生成完毕（164 行）
- [ ] line MBPP 生成完毕（378 行）
- [ ] 4 个产物均通过 postprocess + eval_evalplus
- [ ] 结果填入 `docs/results.md` 的 Locator 消融 / 粒度消融表

## 恢复执行顺序

当有空闲 GPU 后，从这里继续：

1. 跑 `2d. Line Granularity — MBPP`
2. 跑 `line — MBPP` 的 postprocess + eval
3. 执行上面的“结果读取”脚本，确认四行齐全
4. 把结果补进 `docs/results.md` 的粒度消融表

## 显存估算

- Dream-Coder（7B, bf16）：约 14GB
- 本实验 Locator = Rewriter（共用 Dream），无额外模型
- 单卡 A100 40GB 充裕

## 参数说明

- `--span_merge_gap 2`：合并间距 ≤ 2 个 token 的 mask 区间。设 0 则与 token 模式等价（无合并）。
- `token` 模式の baseline 数字（72.6% HE+ plus, 78.7% HE+ base, 70.1% MBPP+ plus, 80.4% MBPP+ base）已硬编码在上方结果读取脚本中，无需额外文件。

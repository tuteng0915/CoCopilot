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

```bash
python -m coder.scripts.gen_remask \
  --input  outputs/base_tuteng/deepseek_mbpp.jsonl \
  --out    outputs/ablation_granularity/deepseek_dream_mbpp_t0.9_gran_line.jsonl \
  --locator dream \
  --confidence_threshold 0.9 \
  --mask_granularity line \
  --temperature 0.1 --top_p 0.95 --seed 3407

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
  --input outputs/ablation_granularity/deepseek_dream_humaneval_t0.9_gran_span.jsonl

python -m coder.scripts.eval_evalplus \
  --samples outputs/ablation_granularity/deepseek_dream_humaneval_t0.9_gran_span-sanitized.jsonl \
  --out_eval outputs/ablation_granularity/deepseek_dream_humaneval_t0.9_gran_span-sanitized_eval_results.json

# span — MBPP
python -m coder.scripts.postprocess_evalplus \
  --input outputs/ablation_granularity/deepseek_dream_mbpp_t0.9_gran_span.jsonl

python -m coder.scripts.eval_evalplus \
  --samples outputs/ablation_granularity/deepseek_dream_mbpp_t0.9_gran_span-sanitized.jsonl \
  --out_eval outputs/ablation_granularity/deepseek_dream_mbpp_t0.9_gran_span-sanitized_eval_results.json

# line — HumanEval
python -m coder.scripts.postprocess_evalplus \
  --input outputs/ablation_granularity/deepseek_dream_humaneval_t0.9_gran_line.jsonl

python -m coder.scripts.eval_evalplus \
  --samples outputs/ablation_granularity/deepseek_dream_humaneval_t0.9_gran_line-sanitized.jsonl \
  --out_eval outputs/ablation_granularity/deepseek_dream_humaneval_t0.9_gran_line-sanitized_eval_results.json

# line — MBPP
python -m coder.scripts.postprocess_evalplus \
  --input outputs/ablation_granularity/deepseek_dream_mbpp_t0.9_gran_line.jsonl

python -m coder.scripts.eval_evalplus \
  --samples outputs/ablation_granularity/deepseek_dream_mbpp_t0.9_gran_line-sanitized.jsonl \
  --out_eval outputs/ablation_granularity/deepseek_dream_mbpp_t0.9_gran_line-sanitized_eval_results.json
```

---

## 结果读取

```bash
python3 -c "
import json

# token baseline（已有产物）
def read_eval(path):
    try:
        d = json.load(open(path))
        s = d.get('summary', {})
        return s.get('plus', '?'), s.get('base', '?')
    except FileNotFoundError:
        return 'MISSING', 'MISSING'

# token baseline：来自 results.md Table 3（已知，无需重读文件）
he_tok  = ('72.6%', '78.7%')
mb_tok  = ('70.1%', '80.4%')
he_span = read_eval('outputs/ablation_granularity/deepseek_dream_humaneval_t0.9_gran_span-sanitized_eval_results.json')
mb_span = read_eval('outputs/ablation_granularity/deepseek_dream_mbpp_t0.9_gran_span-sanitized_eval_results.json')
he_line = read_eval('outputs/ablation_granularity/deepseek_dream_humaneval_t0.9_gran_line-sanitized_eval_results.json')
mb_line = read_eval('outputs/ablation_granularity/deepseek_dream_mbpp_t0.9_gran_line-sanitized_eval_results.json')

print('Granularity  | HE+ plus  HE+ base  | MBPP+ plus  MBPP+ base')
print(f'token        | {he_tok[0]:>8}  {he_tok[1]:>8}  | {mb_tok[0]:>10}  {mb_tok[1]:>10}')
print(f'span (gap=2) | {he_span[0]:>8}  {he_span[1]:>8}  | {mb_span[0]:>10}  {mb_span[1]:>10}')
print(f'line         | {he_line[0]:>8}  {he_line[1]:>8}  | {mb_line[0]:>10}  {mb_line[1]:>10}')
"
```

---

## 完成判定

- [ ] span HumanEval 生成完毕（164 行）
- [ ] span MBPP 生成完毕（378 行）
- [ ] line HumanEval 生成完毕（164 行）
- [ ] line MBPP 生成完毕（378 行）
- [ ] 4 个产物均通过 postprocess + eval_evalplus
- [ ] 结果填入 `docs/results.md` 的 Locator 消融 / 粒度消融表

## 显存估算

- Dream-Coder（7B, bf16）：约 14GB
- 本实验 Locator = Rewriter（共用 Dream），无额外模型
- 单卡 A100 40GB 充裕

## 参数说明

- `--span_merge_gap 2`：合并间距 ≤ 2 个 token 的 mask 区间。设 0 则与 token 模式等价（无合并）。
- `token` 模式の baseline 数字（72.6% HE+ plus, 78.7% HE+ base, 70.1% MBPP+ plus, 80.4% MBPP+ base）已硬编码在上方结果读取脚本中，无需额外文件。

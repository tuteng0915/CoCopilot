# Spec: Locator Ablation（AR logprob / BERT vs dLLM 定位）

## 背景

CoCoder 由三个可独立替换的角色组成：**Drafter → Locator → Rewriter**。
本组实验固定 Drafter（DeepSeek-Coder 6.7B）和 Rewriter（Dream-Coder 7B），
只替换 **Locator**，对比不同打分模型对 pass@1 的影响。

| Locator | 架构 | 参数量 | 感知方向 | 额外代价 |
|---------|------|--------|---------|---------|
| `dream`（默认） | 扩散 LM | 7B | 双向 | 0（共用 Rewriter） |
| `ar` | 自回归 LM | 7B | 单向（causal） | 1 次 AR forward |
| `bert` | MLM | 125M | 双向 | 极小 |

**核心消融问题**：
- AR locator < dLLM locator → 双向上下文感知有独立价值
- BERT locator ≈ dLLM locator → 125M 轻量模型足够定位，无需 7B

**masking 参数**：`--confidence_threshold 0.9`，与论文 Table 3 主结果一致。
dLLM baseline（已有产物）直接复用，无需重跑。

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

# 确认 dLLM baseline（无需重跑，直接用）
ls outputs/base_tuteng/deepseek_dream_remask_humaneval_t0.9_timed.jsonl  # HE baseline
ls outputs/base_tuteng/deepseek_dream_remask_mbpp_t0.9_timed.jsonl       # MBPP baseline

# 创建输出目录
mkdir -p outputs/ablation_locator
```

---

## 实验 1：Pass@1 对比

### 1a. AR Locator — HumanEval

```bash
python -m coder.scripts.gen_remask \
  --input  outputs/base_tuteng/deepseek_humaneval.jsonl \
  --out    outputs/ablation_locator/deepseek_dream_humaneval_t0.9_loc_ar.jsonl \
  --locator ar \
  --locator_model_id deepseek-ai/deepseek-coder-6.7b-instruct \
  --confidence_threshold 0.9 \
  --temperature 0.1 --top_p 0.95 --seed 3407

# 验证完整性（期望 164 行）
python3 -c "
import json, pathlib
lines = [l for l in pathlib.Path('outputs/ablation_locator/deepseek_dream_humaneval_t0.9_loc_ar.jsonl').read_text().splitlines() if l.strip()]
print(f'records: {len(lines)}  (expected 164)')
print('locator:', json.loads(lines[0])['gen']['locator'])
"
```

### 1b. AR Locator — MBPP

```bash
python -m coder.scripts.gen_remask \
  --input  outputs/base_tuteng/deepseek_mbpp.jsonl \
  --out    outputs/ablation_locator/deepseek_dream_mbpp_t0.9_loc_ar.jsonl \
  --locator ar \
  --locator_model_id deepseek-ai/deepseek-coder-6.7b-instruct \
  --confidence_threshold 0.9 \
  --temperature 0.1 --top_p 0.95 --seed 3407

# 验证（期望 378 行）
python3 -c "
import json, pathlib
lines = [l for l in pathlib.Path('outputs/ablation_locator/deepseek_dream_mbpp_t0.9_loc_ar.jsonl').read_text().splitlines() if l.strip()]
print(f'records: {len(lines)}  (expected 378)')
"
```

### 1c. BERT Locator — HumanEval

```bash
python -m coder.scripts.gen_remask \
  --input  outputs/base_tuteng/deepseek_humaneval.jsonl \
  --out    outputs/ablation_locator/deepseek_dream_humaneval_t0.9_loc_bert.jsonl \
  --locator bert \
  --locator_model_id microsoft/codebert-base-mlm \
  --confidence_threshold 0.9 \
  --temperature 0.1 --top_p 0.95 --seed 3407

# 验证（期望 164 行）
python3 -c "
import json, pathlib
lines = [l for l in pathlib.Path('outputs/ablation_locator/deepseek_dream_humaneval_t0.9_loc_bert.jsonl').read_text().splitlines() if l.strip()]
print(f'records: {len(lines)}  (expected 164)')
print('locator:', json.loads(lines[0])['gen']['locator'])
"
```

### 1d. BERT Locator — MBPP

```bash
python -m coder.scripts.gen_remask \
  --input  outputs/base_tuteng/deepseek_mbpp.jsonl \
  --out    outputs/ablation_locator/deepseek_dream_mbpp_t0.9_loc_bert.jsonl \
  --locator bert \
  --locator_model_id microsoft/codebert-base-mlm \
  --confidence_threshold 0.9 \
  --temperature 0.1 --top_p 0.95 --seed 3407

# 验证（期望 378 行）
python3 -c "
import json, pathlib
lines = [l for l in pathlib.Path('outputs/ablation_locator/deepseek_dream_mbpp_t0.9_loc_bert.jsonl').read_text().splitlines() if l.strip()]
print(f'records: {len(lines)}  (expected 378)')
"
```

---

## 实验 2：Sanitize + 评测（4 个产物各跑一次）

```bash
# AR locator — HumanEval
python -m coder.scripts.postprocess_evalplus \
  --dataset humaneval \
  --samples outputs/ablation_locator/deepseek_dream_humaneval_t0.9_loc_ar.jsonl

python -m coder.scripts.eval_evalplus \
  --backend local \
  --dataset humaneval \
  --samples outputs/ablation_locator/deepseek_dream_humaneval_t0.9_loc_ar-sanitized.jsonl \
  --output_file outputs/ablation_locator/deepseek_dream_humaneval_t0.9_loc_ar-sanitized_eval_results.json \
  --summary_out outputs/ablation_locator/deepseek_dream_humaneval_t0.9_loc_ar_summary.json \
  --summary_model deepseek_dream_humaneval_t0.9_loc_ar

# AR locator — MBPP
python -m coder.scripts.postprocess_evalplus \
  --dataset mbpp \
  --samples outputs/ablation_locator/deepseek_dream_mbpp_t0.9_loc_ar.jsonl

python -m coder.scripts.eval_evalplus \
  --backend local \
  --dataset mbpp \
  --samples outputs/ablation_locator/deepseek_dream_mbpp_t0.9_loc_ar-sanitized.jsonl \
  --output_file outputs/ablation_locator/deepseek_dream_mbpp_t0.9_loc_ar-sanitized_eval_results.json \
  --summary_out outputs/ablation_locator/deepseek_dream_mbpp_t0.9_loc_ar_summary.json \
  --summary_model deepseek_dream_mbpp_t0.9_loc_ar

# BERT locator — HumanEval
python -m coder.scripts.postprocess_evalplus \
  --dataset humaneval \
  --samples outputs/ablation_locator/deepseek_dream_humaneval_t0.9_loc_bert.jsonl

python -m coder.scripts.eval_evalplus \
  --backend local \
  --dataset humaneval \
  --samples outputs/ablation_locator/deepseek_dream_humaneval_t0.9_loc_bert-sanitized.jsonl \
  --output_file outputs/ablation_locator/deepseek_dream_humaneval_t0.9_loc_bert-sanitized_eval_results.json \
  --summary_out outputs/ablation_locator/deepseek_dream_humaneval_t0.9_loc_bert_summary.json \
  --summary_model deepseek_dream_humaneval_t0.9_loc_bert

# BERT locator — MBPP
python -m coder.scripts.postprocess_evalplus \
  --dataset mbpp \
  --samples outputs/ablation_locator/deepseek_dream_mbpp_t0.9_loc_bert.jsonl

python -m coder.scripts.eval_evalplus \
  --backend local \
  --dataset mbpp \
  --samples outputs/ablation_locator/deepseek_dream_mbpp_t0.9_loc_bert-sanitized.jsonl \
  --output_file outputs/ablation_locator/deepseek_dream_mbpp_t0.9_loc_bert-sanitized_eval_results.json \
  --summary_out outputs/ablation_locator/deepseek_dream_mbpp_t0.9_loc_bert_summary.json \
  --summary_model deepseek_dream_mbpp_t0.9_loc_bert
```

实际执行时使用 tmux wrapper：

```bash
tmux new-session -d -s eval_spec_loc_ar_he "bash outputs/ablation_locator/eval_spec_loc_ar_humaneval.sh"
tmux new-session -d -s eval_spec_loc_ar_mbpp "bash outputs/ablation_locator/eval_spec_loc_ar_mbpp.sh"
tmux new-session -d -s eval_spec_loc_bert_he "bash outputs/ablation_locator/eval_spec_loc_bert_humaneval.sh"
tmux new-session -d -s eval_spec_loc_bert_mbpp "bash outputs/ablation_locator/eval_spec_loc_bert_mbpp.sh"
```

---

## 实验 3：Fault Detection Ratio 分析

比较各 Locator 对"真实故障 token"的识别能力，无需重跑生成（用已有 t0.9 产物）。

```bash
# 同时加载 DLLM + BERT + AR，在已有 fault pair 上打分
# 注意：DLLM 和 AR 共 ~27GB，需单卡 A100 40GB 或分别运行
python -m coder.analysis.locator_scoring \
  --remask_dir outputs/base_tuteng \
  --dataset humaneval \
  --device cuda \
  --batch_size 16

python -m coder.analysis.locator_scoring \
  --remask_dir outputs/base_tuteng \
  --dataset mbpp \
  --device cuda \
  --batch_size 16
```

如内存不足，可分开加载：
```bash
# 只跑 AR（跳过 DLLM 和 BERT）
python -m coder.analysis.locator_scoring \
  --remask_dir outputs/base_tuteng --dataset humaneval \
  --no_dream --no_bert --device cuda
```

实际执行时为避免一次性加载 DLLM + BERT + AR 的 I/O/显存风险，使用单模型顺序 runner：

```bash
tmux new-session -d -s locator_scoring_split \
  "bash outputs/ablation_locator/run_locator_scoring_split.sh"
```

日志：`outputs/ablation_locator/locator_scoring_split.log`。

---

## 结果读取

```bash
# 读取 wrapper summary，打印 plus% 和 base%
python3 -c "
import json
files = {
  'AR  HE ': 'outputs/ablation_locator/deepseek_dream_humaneval_t0.9_loc_ar_summary.json',
  'AR  MBPP': 'outputs/ablation_locator/deepseek_dream_mbpp_t0.9_loc_ar_summary.json',
  'BERT HE ': 'outputs/ablation_locator/deepseek_dream_humaneval_t0.9_loc_bert_summary.json',
  'BERT MBPP': 'outputs/ablation_locator/deepseek_dream_mbpp_t0.9_loc_bert_summary.json',
}
for label, path in files.items():
    try:
        d = json.load(open(path))
        s = d['summary']
        n = s['n_tasks']
        plus = 100 * s['n_plus_pass'] / n
        base = 100 * s['n_base_pass'] / n
        print(f'{label}: plus={plus:.1f}%  base={base:.1f}%')
    except FileNotFoundError:
        print(f'{label}: NOT FOUND')
"
```

---

## 实际结果（2026-04-16）

### Pass@1

| Locator | HE+ plus% | HE+ base% | MBPP+ plus% | MBPP+ base% | s/sample (HE) | s/sample (MBPP) |
|---|---:|---:|---:|---:|---:|---:|
| dLLM locator (ours) | 72.6% | 78.7% | 70.1% | 80.4% | 14.7s | 10.0s |
| AR logprob locator | 71.3% | 78.7% | 68.5% | 78.8% | 7.8s | 5.7s |
| CodeBERT locator | 69.5% | 76.2% | 68.5% | 78.3% | 7.7s | 5.5s |

正式结果已写入 `docs/results.md` 的 Locator Ablation 表。AR HumanEval sanitize 后有 1/164 样本不可编译，EvalPlus 已完整评测并计入失败/timeout 统计。

### Fault Detection Ratio

ratio = `P(non-fault) / P(fault)`，越高表示模型越能把真实故障 token 打低分。

| Dataset | Locator scorer | pairs | P(fault) | P(non-fault) | ratio | n_fault | n_nonfault |
|---|---|---:|---:|---:|---:|---:|---:|
| HumanEval | dLLM / Dream | 15 | 0.053971 | 0.989499 | 18.33x | 15 | 1590 |
| HumanEval | AR / DeepSeek | 15 | 0.758518 | 0.962481 | 1.27x | 15 | 1762 |
| HumanEval | MLM / CodeBERT | 15 | 0.562128 | 0.813536 | 1.45x | 15 | 2335 |
| MBPP | dLLM / Dream | 34 | 0.064006 | 0.980816 | 15.32x | 37 | 1328 |
| MBPP | AR / DeepSeek | 34 | 0.774335 | 0.953220 | 1.23x | 37 | 1605 |
| MBPP | MLM / CodeBERT | 34 | 0.797646 | 0.837671 | 1.05x | 36 | 1692 |

### Clean Fault Detection Ratio（t0.9 + DeepSeek + dedupe task）

该补充实验只保留 `threshold=0.9`、`ar_tag=deepseek`，并对重复 task 去重。样本数更小，但排除了跨 threshold 重复和非 DeepSeek draft 对 AR scorer 的干扰。

日志：`outputs/ablation_locator/locator_scoring_clean_t09_deepseek.log`。

| Dataset | Locator scorer | pairs | P(fault) | P(non-fault) | ratio | n_fault | n_nonfault |
|---|---|---:|---:|---:|---:|---:|---:|
| HumanEval | dLLM / Dream | 3 | 0.042428 | 0.984647 | 23.21x | 3 | 358 |
| HumanEval | AR / DeepSeek | 3 | 0.718521 | 0.956690 | 1.33x | 3 | 406 |
| HumanEval | MLM / CodeBERT | 3 | 0.686024 | 0.812815 | 1.18x | 3 | 509 |
| MBPP | dLLM / Dream | 4 | 0.007755 | 0.980563 | 126.44x | 4 | 173 |
| MBPP | AR / DeepSeek | 4 | 0.939379 | 0.959398 | 1.02x | 4 | 209 |
| MBPP | MLM / CodeBERT | 4 | 0.842150 | 0.845851 | 1.00x | 4 | 219 |

---

## 完成判定

- [x] AR locator HumanEval 生成完毕（164 行）
- [x] AR locator MBPP 生成完毕（378 行）
- [x] BERT locator HumanEval 生成完毕（164 行）
- [x] BERT locator MBPP 生成完毕（378 行）
- [x] 4 个产物均完成 postprocess + eval_evalplus
- [x] fault-detection ratio 分析跑完（humaneval + mbpp）
- [x] 将数字填入 `docs/results.md` 的 Locator 消融表

## 注意事项

- AR locator（6.7B）+ Dream refiner（7B）同时在卡上：约 27GB，A100 40GB 可行
- BERT locator（125M）极轻，几乎不增加显存；可与 Dream refiner 完全共存
- `--locator_model_id` 若不填，AR 默认 `deepseek-ai/deepseek-coder-6.7b-instruct`，BERT 默认 `microsoft/codebert-base-mlm`
- `locator_scoring.py` 需要同时加载 DLLM + BERT + AR 做 LOO 分析，约 42GB；如不够，用 `--no_dream` / `--no_bert` / `--no_ar` 分批跑

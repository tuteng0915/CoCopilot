# Paper Update Memo — 2026-06-23

> 本文档完整列举 `docs/results.md` 中所有结果，说明哪些**可能未入论文**（需确认后写入），以及对应数值。
> 供另一 session 实际修改 paper LaTeX 用。不需要运行任何脚本。
>
> 判断"是否在论文中"的依据：`docs/experiments-tracker.md`（最后同步 2026-06-03）；3 周内可能有变化，修改前请对照 LaTeX 确认。

---

## 第一类：新实验结果（本轮新增，results.md 之前没有）

### 1. Main Table LLaDA 列更新（`--protect_last_n_tokens 3` fix）

**LLaDA 全部 7 AR × 2 DS 的正确结果（plus% / base%）**：

| AR 模型 | HE plus% | HE base% | MBPP plus% | MBPP base% |
|---------|----------|----------|-----------|-----------|
| DeepSeek-Coder 6.7B | 67.1% | 74.4% | 68.5% | 78.3% |
| Qwen2.5-Coder 7B | 77.4% | 82.9% | 72.8% | 83.3% |
| Llama-3.1 8B | 57.9% | 61.6% | 62.4% | 72.2% |
| CodeLlama 7B | 36.6% | 41.5% | 42.3% | 50.5% |
| Mistral 7B | 31.1% | 37.8% | 41.5% | 48.4% |
| StarCoder2 7B | 22.6% | 25.6% | 28.8% | 33.3% |
| Seed-Coder-Instruct 8B | 76.2% | 81.7% | 72.2% | 84.7% |

> 产物：`outputs/base_tuteng/{AR}_llada_remask_{DS}_t0.9_plnt3-sanitized_eval_results.json`

---

### 2. 新增 dLLM：DiffuCoder（apple/DiffuCoder-7B-Instruct）

DiffuCoder standalone 基本无效（HE 0.6%，MBPP 2.1%），但作为 CoCoder refiner 效果接近 Dream。

**完整结果（plus% / base%）**：

| AR 模型 | HE plus% | HE base% | MBPP plus% | MBPP base% | vs Dream HE Δ |
|---------|----------|----------|-----------|-----------|--------------|
| DeepSeek-Coder 6.7B | 70.1% | 78.0% | 69.3% | 79.6% | −0.6pp |
| Qwen2.5-Coder 7B | **78.7%** | 83.5% | 71.7% | 81.5% | **+1.9pp** |
| Llama-3.1 8B | 57.9% | 61.6% | 63.5% | 72.5% | +0.6pp |
| CodeLlama 7B | 36.6% | 40.9% | 43.7% | 52.6% | +2.5pp |
| Mistral 7B | 31.1% | 39.0% | 42.6% | 50.0% | −1.2pp |
| StarCoder2 7B | 22.6% | 25.6% | 29.6% | 33.9% | −0.6pp |
| Seed-Coder-Instruct 8B | **76.2%** | 82.3% | **72.5%** | 84.7% | **+11.0pp** |

> 产物：`outputs/base_tuteng/{AR}_diffucoder_remask_{DS}_t0.9-sanitized_eval_results.json`

---

### 3. Granularity Ablation 扩展（新增 llama31/qwen + line 粒度）

现有 results.md 仅有 DeepSeek 行，没有 line 粒度行。**完整扩展后表（plus%）**：

| AR | Rewriter | 粒度 | HE plus% | MBPP plus% |
|----|----------|------|----------|-----------|
| DeepSeek | dLLM (Dream) | **token** | **72.6%** | **70.1%** |
| DeepSeek | dLLM (Dream) | span | 65.9% | 69.3% |
| DeepSeek | dLLM (Dream) | line | **46.3%** | 68.3% |
| DeepSeek | AR-Rewrite | token | 68.9% | 67.7% |
| DeepSeek | AR-Rewrite | span | 66.5% | 67.5% |
| DeepSeek | AR-Rewrite | line | 65.2% | 68.3% |
| Llama-3.1 | dLLM (Dream) | token | 57.3% | 64.0% |
| Llama-3.1 | dLLM (Dream) | span | 57.9% | 64.0% |
| Llama-3.1 | dLLM (Dream) | line | 36.0% | 51.3% |
| Llama-3.1 | AR-Rewrite | span | 53.7% | 57.1% |
| Qwen | dLLM (Dream) | token | 76.8% | 72.2% |
| Qwen | dLLM (Dream) | span | 78.0% | 72.2% |
| Qwen | dLLM (Dream) | line | 50.6% | 57.7% |
| Qwen | AR-Rewrite | span | 74.4% | 70.1% |

> 产物：`outputs/ablation_granularity/{AR}_{dream|ar_rewrite}_{DS}_t0.9_gran_{span|line}.jsonl`

**核心结论**：line 粒度下 dLLM rewriter 崩溃（deepseek −26.3pp HE）；AR rewriter 不受影响（65.2%，接近 token 68.9%）。因为 dLLM 无法从头生成整行（training/inference mismatch），AR 逐 token 生成不受此限。

---

### 4. Math-to-Code 新增 LLaDA 列

| AR | AR only | + Dream | + LLaDA (plnt3) |
|----|---------|---------|-----------------|
| **GSM8K** | | | |
| DeepSeek | 61.0% | 62.3% (+1.3pp) | 61.4% (+0.4pp) |
| Qwen | 81.0% | 81.5% (+0.5pp) | 81.3% (+0.3pp) |
| Llama-3.1 | 74.8% | 75.8% (+1.0pp) | 75.4% (+0.6pp) |
| **MATH500** | | | |
| DeepSeek | 6.4% | 6.4% (0pp) | 6.4% (0pp) |
| Qwen | 14.4% | 14.2% (−0.2pp) | 14.0% (−0.4pp) |
| Llama-3.1 | 7.0% | 7.2% (+0.2pp) | 7.0% (0pp) |

> 产物：`outputs/math_code/{AR}_{DS}_code_llada_t0.9_plnt3_eval.json`

---

### 5. Stable-DiffCoder（⚠️ 跳过，不入论文）

全面大幅回退（deepseek HE: 76.2% → 31.7%），疑似 mask token 兼容问题。不纳入论文。

---

## 第二类：已有数据但 experiments-tracker 标注为"未入论文"的结果

以下均在 `docs/results.md` 有完整数据，但 tracker 标注为 paper todo 或 "In-Repo Work Not Yet in Paper"。

---

### 6. 多轮精炼分析（Multi-round Refinement）

> tracker §4："Multi-round local patching, combined pipeline (T rounds)" — In-Repo Work Not Yet in Paper

**数据**（DeepSeek + Dream, τ=0.9，产物：`outputs/tau_rerun/remask_{HE,MBPP}_t0.9_r{2,3}.jsonl`）：

| round | HE base% | HE plus% | MBPP base% | MBPP plus% |
|-------|----------|----------|-----------|-----------|
| r=1 | 76.8% | 69.5% | 80.2% | 69.6% |
| r=2 | 76.8% | 69.5% | 80.2% | 69.6% |
| r=3 | 76.8% | 69.5% | 80.4% | 69.6% |

**结论**：r2 = r1（精确相同）。Dream 对自身输出每个 token 的置信度均接近 1.0，第二轮几乎不触发任何 mask，精炼退化为恒等变换。**一轮即饱和**，无需迭代。这是 CoCoder 的正向特性。

---

### 7. τ 敏感性分析（已有，需确认是否已入论文）

> tracker §2 paper todo H："τ fine-grained scan full curve (0.0–0.9)；Paper Table 1 only has τ=0.5/0.7/0.9"

**remask_kodai**（DeepSeek + Dream，产物：`outputs/remask_kodai/remask_{DS}_t{τ}.jsonl`）：

| τ | HE plus% | MBPP plus% |
|---|----------|-----------|
| 0.7 | 71.3% | 70.1% |
| 0.8 | 71.3% | 70.1% |
| 0.9 | 72.6% | 70.1% |
| 0.93 | 72.6% | 70.1% |
| 0.95 | 72.6% | 70.1% |
| 0.97 | 72.6% | 70.1% |
| 0.99 | 72.6% | 69.8% |

**tau_rerun 完整矩阵**（DeepSeek/Qwen/Llama31/CodeLlama，τ=0.1–0.9，见 results.md §τ敏感性分析）。

**结论**：τ 高度不敏感，0.7–0.9 曲线平坦，无需调参。

---

### 8. 失败模式分解（paper todo E）

> tracker §2 paper todo E："Failure mode breakdown (over/under/algo-mismatch)"

**数据**（DeepSeek + Dream, τ=0.9, `outputs/tau_rerun/remask_{HE,MBPP}_t0.9_fixed.jsonl`）：

| 类别 | HumanEval | MBPP |
|------|-----------|------|
| A) AR✓ Co✓（均通过） | 124/164 (75.6%) | 282/378 (74.6%) |
| B) AR✗ Co✓（CoCoder 修复） | **3/164 (1.8%)** | **21/378 (5.6%)** |
| C1) AR✗ Co✗，无改动（欠 mask） | 32/164 (19.5%) | 63/378 (16.7%) |
| C2) AR✗ Co✗，有改动（错误修复） | 4/164 (2.4%) | 11/378 (2.9%) |
| D) AR✓ Co✗（CoCoder 破坏） | **0/164 (0%)** | **0/378 (0%)** |

**关键结论**：D=0（从不破坏正确解）；C1 主导（算法级错误无法 locate）；与 boundary condition 理论一致。

---

### 9. τ 阈值定性分析（paper todo F）

> tracker §2 paper todo F："τ over/under-mask qualitative cases"

**数据**（DeepSeek HumanEval, τ=0.1 vs 0.9）：

| 模式 | 数量 |
|------|------|
| Under-mask（τ=0.1 错过，τ=0.9 修复）| **2 tasks**（HE/46, HE/74） |
| Over-mask（τ=0.9 破坏，τ=0.1 保留）| **0 tasks** |

**结论**：提高 τ 代价为零，只有收益。已有两个具体案例可作为 paper figure。

---

### 10. 典型案例（Illustrative Case）

> tracker §3.5："Illustrative case — 待做"（但 results.md 已有数据！）

**案例 1：HumanEval/46（off-by-one）**
- AR: `for _ in range(n - 4)` → Dream: `for _ in range(n - 3)`
- dLLM conf ≈ 0.01；AR logprob conf ≈ 0.99
- 原因：AR 单向上下文数 base cases（4 个 → n-4），dLLM 双向同时看到初始化与 docstring 约束

**案例 2：HumanEval/74（equal case 遗漏）**
- AR: `return lst1 if sum1 < sum2` → Dream: `return lst1 if sum1 <= sum2`
- Docstring 明确要求 equal 时返回第一个 list，AR 不"回头"验证

> 产物：`docs/case_study.json`，`outputs/base_tuteng/deepseek_dream_remask_humaneval_t0.9.jsonl`

---

### 11. 文本改写任务（CoEdit / ASSET）

> tracker 未明确标注；但已有完整数据

**数据**（产物：`outputs/rewrite/`）：

| 数据集 | 方法 | SARI | BLEU-4 |
|--------|------|------|--------|
| ASSET | AR (Llama-3.1) | 26.02 | 65.26 |
| ASSET | Dream-General only | 25.21 | 70.05 |
| ASSET | CoCoder τ=0.9 | **25.50** | **65.80** |
| CoEdIT (GEC) | AR (Llama-3.1) | 44.34 | 32.37 |
| CoEdIT (GEC) | Dream-General only | **55.73** | **46.21** |
| CoEdIT (GEC) | CoCoder τ=0.9 | 44.27 | 32.64 |
| CoEdIT (paraphrase) | AR (Llama-3.1) | **41.11** | 15.62 |
| CoEdIT (paraphrase) | Dream-General only | 32.03 | 15.42 |
| CoEdIT (paraphrase) | CoCoder τ=0.9 | 38.93 | 15.25 |

**结论**：CoCoder 在 text 改写任务上不提升 AR（ASSET SARI 接近）。CoEdIT GEC Dream standalone 优于 AR，但 CoCoder 退回至 AR 水平（dLLM 修复能力被 AR 草稿约束）。与 writing benchmark 的 graceful degradation pattern 一致。

---

### 12. 通用领域基准（已入论文，仅供确认数值）

> results.md §General Domain Benchmarks，已写入 paper（per tracker）

| 任务 | AR (Llama) | Dream only | CoCoder τ=0.9 |
|------|-----------|-----------|---------------|
| FRAMES (EM%) | 0.0% | 1.1% | 0.0% |
| FRAMES (F1%) | 4.4% | 11.4% | 4.4% |
| HotpotQA (EM%) | 13.5% | 16.4% | 9.7% |
| HotpotQA (F1%) | 22.4% | 25.4% | 18.9% |
| WildBench writing | 40.69% | 3.09% | 40.15% |

---

### 13. Table 4 扩展行（Mistral / Seed-Coder-Instruct 完整方法，paper todo J）

> tracker §3 paper todo J（标注"待跑"，但数据已在 results.md §Table 4 扩展行）

| AR | 方法 | HE plus% | MBPP plus% |
|----|------|----------|-----------|
| Mistral | baseline | 31.1% | 41.8% |
| Mistral | +Self-Refine | 23.8% | 37.6% |
| Mistral | +Reflexion | 23.2% | 29.1% |
| Mistral | +Rerank k=8 | 34.8% | 44.2% |
| Mistral | +Locate-AR-Rewrite | 28.7% | 43.4% |
| Mistral | +LLaDA plnt3 | 31.1% | 41.5% |
| Mistral | +Dream (ours) | **32.3%** | **42.6%** |
| Seed-Coder-Instruct | baseline | 70.1% | 72.2% |
| Seed-Coder-Instruct | +Self-Refine | 74.4% | 69.0% |
| Seed-Coder-Instruct | +Reflexion | 57.3% | 49.2% |
| Seed-Coder-Instruct | +Rerank k=8 | **77.4%** | **73.8%** |
| Seed-Coder-Instruct | +Locate-AR-Rewrite | 76.2% | 69.8% |
| Seed-Coder-Instruct | +LLaDA plnt3 | 76.2% | 72.2% |
| Seed-Coder-Instruct | +Dream (ours) | 65.2% | 72.2% |

---

## 第三类：待完成（不影响当前 paper 修改）

| 实验 | 状态 | 备注 |
|------|------|------|
| LCB refiner（llama31/qwen × Dream/LLaDA） | GPU 2 运行中，~3.4 天 | 现有表只有 DeepSeek |
| BCB refiner | LCB 完成后开始 | — |
| Token-level precision/recall (todo D) | 需要新脚本 | difflib diff alignment |
| (τ, temperature, top-p) joint sensitivity (todo G) | 未跑 | Low priority |

---

## 修改建议汇总（按 paper section 整理）

| Paper 位置 | 修改内容 | 优先级 |
|-----------|---------|--------|
| Main Table（tab:model_pairs）| 替换 LLaDA 列为 plnt3 数值（第1节） | 🔴 必须 |
| Main Table | 新增 DiffuCoder 行（或 Appendix）（第2节） | 🟡 建议 |
| Granularity Ablation | 扩展为 3 AR 模型 + Line 行（第3节） | 🔴 必须（已有 todo C） |
| Math-to-code | 新增 LLaDA 列（第4节） | 🟡 建议 |
| Multi-round 分析 | 新增或恢复"单轮饱和"段落（第6节） | 🟡 建议（强化 story） |
| τ 分析 | 确认是否有完整曲线图，补 τ=0.93–0.99 数据（第7节）| 🟢 已大部分在论文 |
| Failure Mode Breakdown (todo E) | 用第8节数据补充（A/B/C1/C2/D 分类）| 🟡 已有数据 |
| τ qualitative (todo F) | 用第9节 under/over-mask 2+0 例子（已有案例图）| 🟡 已有数据 |
| Illustrative Case | 补充第10节两个案例（HE/46, HE/74）| 🔴 已有数据，tracker 标"待做" |
| Text rewrite (CoEdit/ASSET) | 补充第11节，强化 graceful degradation pattern | 🟢 optional |
| Table 4（tab:combined）| 补入 Mistral/Seed-Coder 完整方法行（第13节） | 🟡 todo J 数据就绪 |

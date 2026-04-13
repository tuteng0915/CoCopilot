# 实验追踪（NeurIPS 2026 投稿准备）

本文档汇总投稿前需要补做的实验，以及仓库里**已实现但结果尚未汇总进论文表**的工作，便于与 `docs/ablation_ideas.md`、`docs/completion-checklist.md` 对齐。

---

## 0. Rebuttal-CoCoder 分支比对（2026-04-13 更新）

**比对结论**：合作者的 Rebuttal-CoCoder 分支与 NeurIPS26-CoCoder 在实验数字上**完全一致**，无额外新数据。以下为各 rebuttal 承诺项目的最新状态：

| 承诺（rebuttal 位置） | 内容 | 当前状态 |
|----------------------|------|---------|
| Group 2 / Table 5 | Self-Refine/Reflexion/Reranking/dLLM-locate+AR-rewrite pass@1 | ✅ 已写入论文 |
| Group 4 / Table 2 | LiveCodeBench 12.0%，BigCodeBench 23.0% | ✅ 已写入论文 |
| Group 9 / Table 4 | DeepSeek + LLaDA-8B（HumanEval+ 65.9%）| ✅ 已写入论文 |
| Group 4 / Table 4 | Qwen2.5-Coder/Llama-3.1/StarCoder2 + Dream-Coder（HumanEval+）| ❌ 仅 MBPP 完成，HumanEval+ 缺（见 §3）|
| Group 2 / Table 5 | Wall-clock latency 列 | ❌ 未填入表中 |
| Group 7 | Mask 粒度消融（token/span/line）| ❌ 未跑 |
| Group 11 | Token 级 precision/recall | ❌ 无脚本 |
| Group 7 JwDe Q4 | 失败模式分解（over/under/algo-mismatch）| ❌ 未做 |
| Group 5 XBny Q1 | τ 过/欠 mask 定性案例（`qualitative_analysis_2.png` 可能已有图）| ❌ 未集成 |
| Group 12 YFoi Q2 | (τ, temperature, top-p) 联合敏感性 | ❌ 未跑 |

**注**：论文中已修复的"幽灵内容"：已移除 `tab:granularity` 悬空引用、AR+Multi-round 和 MLM-style baseline 描述（见 NeurIPS26-CoCoder/docs/changelog.md）。

---

## 1. 主文表格待填数字（管线已实现，结果待跑）

| 论文位置 | 内容 | 状态 | 代码/产物线索 |
|----------|------|------|----------------|
| Table 5（`tab:baselines`） | Self-Refine、Reflexion、Reranking ($k{=}8$)、dLLM-locate + AR-rewrite；pass@1 | ✅ 已填 | — |
| Table 5（`tab:baselines`） | **Overhead 列**（s/sample, HE）：Self-Refine 4.6s、Reflexion 9.5s、Reranking 61.7s、Locate-AR-Rewrite 7.2s、Collab ≈5s | ✅ 已填（2026-04-13）| results.md Table 4 |
| Table 2（`tab:extended`） | LiveCodeBench 全量（LCB 1055题：DeepSeek 11.4%，Dream 2.9%，Collab 11.6%）；BigCodeBench 全量（BCB 1140题：DeepSeek 24.7%，Dream 22.5%，Collab 24.6%）| ✅ 已填（2026-04-13）| results.md Extended Table Shards |
| Table 4（`tab:model_pairs`） | DeepSeek+Dream-Coder（+15.9pp）、DeepSeek+LLaDA（+9.2pp）、Qwen+Dream（−0.6pp）、Llama+Dream（−0.6pp）、StarCoder2+Dream（±0.0pp on HE+）| ✅ 已填（2026-04-13）| results.md Table 3 |
| ~~Table 3（tab:math）~~ | GSM8K/MATH — **已注释掉，数字存疑** | ⚠️ 占位符 | results.md 显示 DeepSeek GSM8K=19.0%，与论文 48.7% 不符；Dream 无数学结果 |

---

## 2. 附录分析（非单一 pass@1 数字）

| 内容 | 优先级 | 说明 | 相关实现或文档 |
|------|--------|------|----------------|
| **Mask 粒度消融** | 🔴 高 | token / span / line（及 `span_merge_gap`）；结果写入附录 tab:granularity | `gen_remask --mask_granularity {token,span,line}`；设计见 `ablation_ideas.md` §A |
| **τ 阈值定性案例** | 🟡 中 | 保守 τ=0.7 漏修 vs 激进 τ=0.9 过 mask 的对比样例 | 手动筛选；`images/qualitative_analysis_2.png` 可能已有图，需确认内容 |
| **失败模式分解** | 🟡 中 | (i) over-masking、(ii) under-masking、(iii) algorithm mismatch + 具体样例 | 从已评测结果中筛"refinement 后变差"案例，规则/人工分类 |
| **Token 级 precision/recall** | 🟡 中 | 低置信 token 与真错误位置对齐；计算 precision/recall | 需新建 token 对齐脚本（difflib diff + confidence mask 比对） |
| **(τ, temperature, top-p) 联合敏感性** | 🟢 低 | 附录 sweep 分析 | `gen_remask --temperature --top_p`；设计见 `ablation_ideas.md` |

---

## 3. 待跑数据

> 实际进度见 `docs/results.md`（自动生成）。

- [x] HumanEval / MBPP：`*_reflexion_feedback*.jsonl` — **已有结果**
- [x] HumanEval / MBPP：`*_rerank_logprob_k8*.jsonl` — **已有结果**
- [x] Table 3 HumanEval+：qwen/llama31/starcoder2 + dream remask — **全部完成**（见 results.md）
- [x] Table 3 MBPP：qwen/llama31/starcoder2 + dream remask — **全部完成**
- [x] Extended table shards（dream + collab on LCB/BCB）— **全部完成**（LCB: 1055✅，BCB: 1140✅）
- [x] DeepSeek + LLaDA on HumanEval+ — ✅（65.9%）
- [ ] DeepSeek + LLaDA on MBPP — ❌ 尚未跑（results.md Table 3 中该行 MBPP 无数据）
- [ ] LiveCodeBench 其他模型（qwen / llama31 / dream）：`n_scored=0`，评测未跑通（original_json 字段缺失，见 pitfalls.md）
- [ ] BigCodeBench raw pass@1=0.0 核查（deepseek_pass1_clean 正常，raw 版本异常）
- [ ] GSM8K / MATH Collaborative Coding 实验 — ❌ **尚未跑**（论文中数字为占位符，已注释掉）

---

## 4. 代码库已有、尚未进论文的后续实验

| 主题 | 文档 |
|------|------|
| Reflexion + EvalPlus **真实失败反馈** pipeline | `ablation_ideas.md` §C |
| Reranking 用 **AR logprob** 替代启发式打分 | `ablation_ideas.md` §B |
| 多轮局部修补、组合管线（T 轮） | `ablation_ideas.md` §C1 |

---

## 5. 完成判定与产物路径

- EvalPlus / LiveCodeBench / BigCodeBench 的「跑完且可信」检查：见 **`completion-checklist.md`**。
- 已知交互/评测坑：见 **`pitfalls.md`**。

---

## 6. 数学任务（泛化性验证）

为论证方案不局限于代码生成，我们在数学推理任务上补充实验。数据集选择依据：答案验证清晰（数字/表达式）、社区引用广泛、与代码实验形成互补。

| Benchmark | 规模 | 说明 | 脚本 |
|-----------|------|------|------|
| GSM8K | 1319（test） | 小学数学，链式推理，整数/小数答案 | `gen_math --dataset gsm8k` |
| MATH-500 | 500（test） | 竞赛数学代表子集（AMC/AIME 难度），LaTeX 答案，含 5 个难度级别与 7 个学科 | `gen_math --dataset math500` |

待跑实验：

- [x] GSM8K：DeepSeek/Qwen/Llama-3.1 baseline — **已完成**（DeepSeek 19.0%，Qwen 30.6%，Llama-3.1 84.5%）
- [x] MATH-500：DeepSeek/Qwen/Llama-3.1 baseline + subject breakdown — **已完成**（DeepSeek 4.6%，Qwen 37.6%，Llama-3.1 38.6%）
- [ ] 在上述数据集上跑 CoCoder 管线，与 baseline 对比 accuracy + latency（Dream 过慢~217s/题，需另选策略）

评测产物命名约定同代码任务，见 `runbook.md` § 数学任务。

---

## 7. Scope（不在本轮实验清单内）

- **多文件仓库级 / SWE-bench 级**：当前管线为单段代码 + 固定窗口；完整仓库评测需 agent/检索/基础设施，不在本轮范围内（见 `main.tex` Limitations）。

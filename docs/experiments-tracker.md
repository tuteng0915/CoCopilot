# 实验追踪（NeurIPS 2026 投稿准备）

本文档汇总投稿前需要补做的实验，以及仓库里**已实现但结果尚未汇总进论文表**的工作，便于与 `docs/ablation_ideas.md`、`docs/completion-checklist.md` 对齐。

---

## 1. 主文表格待填数字（管线已实现，结果待跑）

| 论文位置 | 内容 | 代码/产物线索 |
|----------|------|----------------|
| Table 4（`tab:baselines`） | Self-Refine、Reflexion、Reranking ($k{=}8$)、dLLM-locate + AR-rewrite vs. Collaborative Coding；pass@1 + 延迟 | `gen_self_refine`、`gen_reflexion`、`gen_rerank`、`gen_locate_ar_rewrite`、`gen_remask`；EvalPlus 评测链见 `completion-checklist.md` |
| Table 2（`tab:extended`） | LiveCodeBench、BigCodeBench | `gen_livebench`、`eval_livebench`、`gen_bigcodebench`、`eval_bigcodebench` |
| Table 3（`tab:model_pairs`） | 多 AR 草稿模型 + 备选 dLLM（如 LLaDA-8B）在 HumanEval+ 等上 | `gen_evalplus` 与各 `*_coder` 模型封装 |

---

## 2. 附录分析（非单一 pass@1 数字）

| 内容 | 说明 | 相关实现或文档 |
|------|------|----------------|
| **τ 阈值定性案例** | 保守 τ 漏修 vs 激进 τ 过 mask 的样例 | 跑完后手动筛选写附录；与 `main` 中 threshold 讨论对应 |
| **(τ, temperature, top-p) 联合敏感性** | 附录分析 | 需在统一协议下 sweep；`dream_coder` / `gen_remask` 超参 |
| **Mask 粒度消融** | token / span / line（及 `span_merge_gap`） | `dream_coder.generate_with_remask`、`gen_remask --mask_granularity`；设计见 `ablation_ideas.md` §A |
| **失败模式分解** | 过 mask、欠 mask、算法整体错误等 + 示例 | 需离线标注或规则分类 + 个案 |
| **Token 级 precision/recall** | 低置信 token vs 真错误位置对齐 | 需 draft 与参考解 token 对齐脚本 |

---

## 3. 待跑数据

> 实际进度见 `docs/results.md`（自动生成）。

- [x] HumanEval / MBPP：`*_reflexion_feedback*.jsonl` — **已有结果**
- [x] HumanEval / MBPP：`*_rerank_logprob_k8*.jsonl` — **已有结果**
- [x] Table 3 MBPP：qwen/llama31/starcoder2 + dream remask on MBPP — **全部完成**（9 个 pair 全 ✅，见 results.md Table 3）
- [ ] DeepSeek + LLaDA on MBPP（baseline 表中该行显示 —，尚未跑）
- [ ] LiveCodeBench 其他模型（qwen / llama31 / dream）：`n_scored=0`，评测未跑通（original_json 字段缺失）
- [ ] BigCodeBench raw pass@1=0.0 核查（deepseek_pass1_clean 正常，raw 版本异常）
- [ ] Extended table shards（dream + collab on LCB/BCB）：仍在进行，`run_extended_table.py`

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

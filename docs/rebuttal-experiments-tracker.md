> **此文档已迁移**：请使用 [`docs/experiments-tracker.md`](experiments-tracker.md)。
> 已完成的实现进展见 [`docs/done/impl_progress_2026-03.md`](done/impl_progress_2026-03.md)。

# [已归档] Rebuttal 实验与承诺追踪

本文档已整合进 `docs/experiments-tracker.md`，保留此文件仅作历史参考。

---

## 1. 主文表格待填数字（已实现管线，结果待汇总）

| 承诺位置（rebuttal / 论文） | 内容 | 代码/产物线索 |
|----------------------------|------|----------------|
| Table 4（`tab:baselines`） | Self-Refine、Reflexion、Reranking ($k{=}8$)、dLLM-locate + AR-rewrite vs. Collaborative Coding；pass@1 + 延迟 | `coder.scripts.gen_self_refine`、`gen_reflexion`、`gen_rerank`、`gen_locate_ar_rewrite`、`gen_remask`；EvalPlus 评测链见 `completion-checklist.md` |
| Table 2（`tab:extended`） | LiveCodeBench、BigCodeBench | `gen_livebench`、`eval_livebench`、`gen_bigcodebench`、`eval_bigcodebench` |
| Table 3（`tab:model_pairs`） | 多 AR 草稿模型 + 备选 dLLM（如 LLaDA-8B）在 HumanEval+ 等上 | `gen_evalplus` 与各 `*_coder` 模型封装 |

---

## 2. 附录 / camera-ready 承诺（分析类，非单一 pass@1 数字）

| 承诺 | 说明 | 相关实现或文档 |
|------|------|----------------|
| **τ 阈值定性案例**（Group 5 Q1） | 保守 τ 漏修 vs 激进 τ 过 mask 的样例 | 跑完筛选后写附录；与 `main` 中 threshold 讨论呼应 |
| **(τ, temperature, top-p) 联合敏感性**（Group 12 YFoi Q2） | 附录分析 | 需在统一协议下 sweep；`dream_coder` / `gen_remask` 超参 |
| **Mask 粒度消融**（Group 7） | token / span / line（及 `span_merge_gap`） | `dream_coder.generate_with_remask`、`gen_remask --mask_granularity`；设计见 `ablation_ideas.md` §A |
| **失败模式分解**（Group 7 JwDe Q4） | 过 mask、欠 mask、算法整体错误等 + 示例 | 需离线标注或规则分类 + 个案 |
| **Token 级 precision/recall**（Group 11） | 低置信 token vs 真错误位置对齐 | 需 draft 与参考解 token 对齐脚本（rebuttal 中描述） |

---

## 3. Group 1（τ=0.5 与阈值敏感性）— 当前按作者计划可暂缓

- 更细 τ 扫描、过渡区刻画、与审稿人 GhgG 承诺的深入解释等：**未列入本文档的强制交付**，待专项实验后再更新 rebuttal/正文。

---

## 4. 代码库已有、rebuttal 未单独承诺但相关的后续实验

| 主题 | 文档 |
|------|------|
| Reflexion + EvalPlus **真实失败反馈** pipeline | `TODO_reflexion_evalplus_feedback.md` |
| Reranking 用 **AR logprob** 替代启发式打分 | `ablation_ideas.md` §B |
| 多轮局部修补、组合管线（T 轮） | `ablation_ideas.md` §C |
| **MLM 风格** refinement baseline | rebuttal 已列为 **future work**（公平对比需额外设计，见 `rebuttal.tex` JwDe W2） |

### 4.1 本 session 进展（2026-03-30）

- [x] `coder.analysis.evalplus_feedback`：支持更稳健的失败摘要抽取（含 `base_status_counts`、多字段详情兼容、`raw_details` 可选）
- [x] `coder.scripts.gen_reflexion`：支持 `--feedback_field`，并在输出中记录 `gen.feedback` / `eval_feedback`
- [x] `docs/runbook.md`：补充 EvalPlus 真实失败反馈版 Reflexion 命令模板与命名约定
- [x] `docs/ablation_ideas.md`：补充 logprob rerank 与反馈驱动 Reflexion 的最新用法

### 4.2 仍待跑数（脚本已就绪）

- [ ] 在 HumanEval/MBPP 上各跑一组 `*_reflexion_feedback*.jsonl`，并汇总 pass@1 + latency
- [ ] 在 HumanEval/MBPP 上各跑一组 `*_rerank_logprob_k8*.jsonl`，并汇总与 `self_judge/heuristic` 对比

---

## 5. 完成判定与产物路径

- EvalPlus / LiveCodeBench / BigCodeBench 的「跑完且可信」检查：见 **`completion-checklist.md`**。
- 已知交互/评测坑：见 **`pitfalls.md`**。

---

## 6. Scope（与 rebuttal 一致，非实验待办）

- **多文件仓库级 / SWE-bench 级**：当前管线为单段代码 + 固定窗口；完整仓库评测需 agent/检索/基础设施，**不在本轮实验清单内**（见 `rebuttal` Group 4、`main.tex` Limitations）。

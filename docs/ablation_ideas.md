# Ablation Experiments and Mini-Study Ideas (CoCoder)

已完成并入论文的实验：§A（粒度消融）、§B（Locator 模型替换）、§C（启发式 reranking）、§D（Reflexion baseline）、E1（多轮精炼）、E2（解耦组合全覆盖）。以下仅保留**未完成**条目。

---

## 框架速查

```
AR drafter ──► Locator ──► dLLM rewriter
```

---

## 剩余待完成实验

### A. `step` 粒度（代码有，结果未跑）

Locator 粒度的第四档：找最低置信 step，**截断续写**（rewriter 接管后续生成），而非 fill-in-the-blank。

与 token/span/line 在同一个消融维度，但语义上从"填坑"变为"续写"，更适合 CoT 场景。

```bash
gen_remask --mask_granularity step
```

优先级：低（代码任务意义有限，主要用于 CoT/math 扩展研究）。

---

### B. Reranking 进阶评分模式（代码有，未入论文）

当前论文中 AR+Reranking 使用 `heuristic` 模式。以下两种模式已实现但未跑：

```bash
# logprob 模式：sum/avg log p(completion|prompt) 选最优
gen_rerank --score_mode logprob --logprob_norm avg

# self_judge 模式：AR 模型 listwise 自判断
gen_rerank --score_mode self_judge
```

优先级：中低（论文已有 heuristic reranking 作为 baseline，增量价值有限；但 logprob 模式是更"公平"的 reranking baseline）。

---

### C. Oracle Locator 性能上界（可选）

Mask 掉 AR draft 与 reference solution 实际有差异的 token，然后用 dLLM rewrite。

**目的**：量化 localization 质量瓶颈——如果 oracle locator 也只有 +X pp，说明 rewrite 才是瓶颈；如果有大幅提升，说明 localize 准确率是主要上升空间。

需要 token-level diff 对齐脚本（未实现）。优先级：中（narrative_reframe §4.1.B）。

---

### D. Calibration Plot（数据有，图未画）

将 dLLM 置信度按分位数分桶，计算每个桶内实际 fault token 的比例。

论文已有 ROC/AUC 曲线（app:calibration），但 calibration plot 可视化 **置信度校准性**，与 ROC 互补。从现有 `locator_scoring.py` 输出直接计算，无需新实验。优先级：中高（narrative_reframe §4.2.A）。

---

### E. (τ, temperature, top-p) 联合敏感性（未跑）

`gen_remask --temperature --top_p` 网格搜索，量化三个解码参数的交互效果。

优先级：低（τ 单维扫描已在论文，联合实验信息增量小）。

---

### F. 编辑量 vs. 成功率分析图（部分完成）

对每个问题统计 mask 数、diff 距离、pass/fail，找最优编辑量区间（与 Locator 类型交叉分析）。

当前论文有失败模式分解（A/B/C1/C2/D，app:failure），但缺少连续变量（编辑量）vs. 成功率的分布图。

优先级：中（对"surgical fix"叙事有补充）。

---

### G. Prompt / 输出格式鲁棒性（未实现）

Code fence 敏感性、system prompt 强度、max\_new\_tokens 截断敏感性。优先级：低。

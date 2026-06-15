# CoCoder 论文叙事重构

> 写于 2026-06-04。本文档记录将论文核心论证从"互补组合"重构为"可检测性"框架的思路，包括需要补充的实验设计。

---

## 一、现有叙事的问题

**旧逻辑**：AR 模型擅长起草代码，dLLM 擅长双向精炼，两者结合效果更好。

这个叙述的缺陷：
- "擅长 X"是直觉说法，不是机制性解释
- 没有回答：为什么 dLLM 精炼比 AR 自我精炼（Self-Refine）更好？
- 没有回答：为什么这个组合只在 25–60% pass@1 区间有效？
- 将 math 任务上的失败仅作为负面结果汇报，而非理论预测的边界条件

---

## 二、新叙事的起点：一个真实悖论

| 模型 | HumanEval+ |
|------|-----------|
| Dream-Coder 7B（dLLM standalone） | 43.3% |
| DeepSeek-Coder 6.7B（AR standalone） | 56.7% |
| DeepSeek + Dream-Coder（CoCoder） | **72.6%（+15.9pp）** |

> **悖论**：dLLM 单独生成代码比 AR **更差**，但将它加入 AR 的工作流之后，性能反而**大幅超越** AR。
>
> → dLLM 的价值**不在于生成代码**，而在于**感知 AR 草稿中的错误**。

这个悖论本身就是论文最好的开篇——它迫使读者思考"为什么"，而不是接受一个显而易见的结论。

---

## 三、新论证链

```
① dLLM standalone 很差
   └─ 为什么？parallel denoising 不保证 sequential 一致性
   └─ 证据：Dream-Coder 43.3%、LLaDA 12.8%（远低于同量级 AR）

② 但 CoCoder 有大幅提升（悖论成立）
   └─ 端到端数字：+15.9pp（DeepSeek+Dream）

③ 端到端消融：提升的来源是 locate，不是 rewrite
   └─ dLLM-locate + AR-rewrite：已有 68.9%（locate 贡献 +12.2pp）
   └─ dLLM-locate + dLLM-rewrite（ours）：72.6%（rewrite 额外贡献 +3.7pp）
   └─ Random-locate + dLLM-rewrite：【需要补充】——直接验证 locate 是否是核心
   └─ 结论：locate 是主体，rewrite 是锦上添花

④ 机制分析：为什么 dLLM 能 locate 而 AR 不能？
   └─ dLLM fault detection ratio：23x（HE）、126x（MBPP）
   └─ AR logprob ratio：1.33x（几乎没有区分能力）
   └─ 根因：AR 生成时只有左侧上下文；dLLM 评估时有完整双向上下文
   └─ 【需要补充】Calibration plot、ROC/AUC、典型案例

⑤ 边界条件：什么错误类型可以被 locate
   └─ Code：结构性错误（syntax/type），token 层面可检测 → CoCoder 有效
   └─ Math：算术/语义错误（28+15=42 语言上 fluent），token 层面不可检测 → CoCoder 无效
   └─ 证据：math token 置信度 Cohen's d ≈ 0.05（vs code 的 23x ratio）
   └─ 意义：边界条件是理论预测的，不是 empirical 观察
```

---

## 四、需要补充的实验

### 4.1 端到端消融（最关键）

#### 实验 A：Random Locator Baseline（**最高优先级**）

**设计**：用完全随机的 mask（与 dLLM 相同的 mask ratio，但随机选 token），然后用 dLLM rewrite。

**目的**：直接回答"dLLM 的价值是在 locate 还是在 rewrite？"

| 结果预期 | 解读 |
|---------|------|
| Random-locate ≈ dLLM-locate（~72%） | Locate 不重要，dLLM 的价值仅在 rewrite → 故事不成立 |
| Random-locate << dLLM-locate（~58%）| **Locate 是核心**，dLLM 的 discriminative 置信度是关键 → 故事成立 |

**实现**：`gen_remask --locator random --mask_ratio <τ对应的平均mask比例>`

已有 `ablation_ideas.md` §E2 提到过 random locator，但一直没有实际跑。这是整个新叙事的 make-or-break 实验。

---

#### 实验 B：Oracle Locator Upper Bound（可选，增强说服力）

**设计**：mask 掉 AR draft 与 reference solution 实际有差异的 token，然后用 dLLM rewrite。

**目的**：如果 locator 完美，CoCoder 能到多少？量化 localization 质量和最终性能之间的关系。

**实现**：需要先对齐 AR draft 和 gold solution（token-level diff），然后把 diff 位置作为 mask。

---

#### 实验 C：分解 locate vs rewrite 的贡献（已有，需要重新呈现）

| 方法 | HumanEval+ | Δ vs AR-only |
|------|-----------|-------------|
| AR-only | 56.7% | — |
| AR + Random-locate + dLLM-rewrite | **【待跑】** | — |
| AR + dLLM-locate + **AR**-rewrite | 68.9% | +12.2pp |
| AR + dLLM-locate + **dLLM**-rewrite（ours） | 72.6% | +15.9pp |
| AR + **Oracle**-locate + dLLM-rewrite | **【可选跑】** | — |

> locate 的贡献（+12.2pp）远大于 rewrite 的额外贡献（+3.7pp），支持"locate 是核心"的论点。

---

### 4.2 机制分析（mechanistic）

#### 分析 A：Calibration Plot（半天，从现有数据直接算）

**设计**：将 dLLM 置信度按分位数分桶（如 10 个桶），计算每个桶内实际是 fault token 的比例。

**目的**：可视化 dLLM 置信度的 discriminativeness。如果低置信分桶里 fault 比例显著更高，图形呈单调递减曲线，直观且有说服力。

**实现**：从现有 `locator_scoring.py` 的输出直接计算，无需新实验。

---

#### 分析 B：ROC / AUC（半天，从现有数据直接算）

**设计**：把 locator 当 binary classifier（低置信 = predict fault），计算三种 locator 的 ROC 曲线和 AUC。

| Locator | 预期 AUC |
|---------|---------|
| dLLM (Dream-Coder) | 高（23x ratio 对应） |
| AR logprob | 低（1.33x ratio 对应） |
| CodeBERT | 低（1.18x ratio 对应） |
| Random | 0.5（baseline） |

**意义**：把 23x ratio 这个数字转化为 AUC，更标准化、更易于读者比较。

---

#### 分析 C：典型案例——AR 盲区，dLLM 可见（1 小时，手工）

**设计**：在已有 case study jsonl 里，找满足以下条件的样本：
1. AR draft 有错误（failed）
2. CoCoder 修复了（passed）
3. 被 mask 的 token 数量 ≤ 5（"surgical" fix）
4. **具体找**：AR logprob 对那个错误 token 高置信，dLLM 对其低置信

**目的**：一个具体案例比任何定量描述都有说服力。

**呈现方式**：
```
AR draft:    for i in range(n+1):  ← AR 对 "+1" 高置信（左侧循环模式支持），dLLM 低置信（后续逻辑需要 range(n)）
CoCoder:     for i in range(n):    ← dLLM 感知到全局不一致，mask 并修正
```

**来源**：`outputs/base_tuteng/` 里的 `codellama` 或 `deepseek` + dream remask 的 case study 产物。

---

#### 分析 D：为什么 dLLM Standalone 差（理论分析，无需新实验）

**需要在 paper 中解释清楚**：

- **dLLM 从 scratch 生成**：从完全 mask 状态开始，每步 unmask 若干 token。每个 token 虽然有双向注意力，但因为其他 token 也在同步 unmask，整体缺乏 sequential commitment。代码要求严格的 left-to-right 语法/语义依赖，parallel denoising 不保证这一点。
- **dLLM 评估已有序列**：给定完整的 AR draft，对每个 token 计算 `P(t_i | t_1...t_{i-1}, t_{i+1}...t_n)`。这正是 dLLM 的训练目标（masked token prediction），因此 calibrated。
- **关键对称性**：dLLM 的训练弱点（从 scratch 生成代码）和训练优势（评估 token 在完整上下文中的合理性）恰好形成互补——前者是 AR 的强项，后者是 CoCoder 利用的价值。

---

### 4.3 实验优先级总结

| 实验 | 优先级 | GPU 需求 | 实现难度 | 支撑的论点 |
|------|--------|---------|---------|-----------|
| Random locator baseline | 🔴 最高 | 1 GPU，~2h | 低（加一行） | locate 是核心价值（make-or-break） |
| Calibration plot | 🔴 高 | 无 | 低（现有数据） | dLLM 置信度 discriminative |
| ROC / AUC | 🔴 高 | 无 | 低（现有数据） | dLLM 作为 locator 的定量质量 |
| Illustrative case | 🔴 高 | 无 | 低（手工） | 机制直觉，读者记忆点 |
| Oracle locator | 🟡 中 | 1 GPU，~2h | 中 | 性能上界，理解 localization 质量瓶颈 |
| 错误类型分布分析 | 🟡 中 | 无 | 高 | 细化边界条件 |

---

## 五、现有实验的重新解读

新框架下，Table 3（tab:combined）的 baseline 对比可以重新叙述：

| 方法 | 结果 | 新框架下的解读 |
|------|------|--------------|
| Self-Refine | 小幅提升或退化 | AR 自己找不到自己的错误（AR logprob ratio 1.3x），反思是盲目的 |
| Reflexion | 普遍退化 | 同上，且 verbal reflection 进一步放大了 AR 的"自我一致性"偏见 |
| Reranking (k=8) | 有效但代价高 | 通过采样多样性间接绕过 locate 问题，但不解决根本；61.7s/sample |
| dLLM-locate + AR-rewrite | 有效（68.9%） | **直接验证**：locate 是主体价值，rewrite 可以是 AR 本身 |
| CoCoder（ours） | 最好（72.6%） | locate + dLLM-rewrite 叠加，dLLM 在两个角色都有优势 |

> **原来的叙述**：我们比所有 baselines 好。
>
> **新的叙述**：所有 AR-based 精炼方法（Self-Refine/Reflexion/Reranking）失败的共同原因是无法精确 locate 错误；dLLM-locate + AR-rewrite 证明 locate 是关键；我们的方法在 locate 上最好，在 rewrite 上也略优。

---

## 六、对 math 实验结论的重新定位

**原来的定位**：Failure case——CoCoder 在数学上不好使。

**新的定位**：**理论预测的边界条件验证**。

新框架预测：CoCoder 有效当且仅当 AR 的错误具有"结构性可检测信号"（dLLM 置信度能区分 fault/non-fault）。

| 领域 | 错误类型 | 可检测性 | dLLM 信号 | CoCoder 效果 |
|------|---------|---------|----------|------------|
| Code | 结构性（syntax/type/边界） | 高 | Cohen's d >> 0（23x ratio） | 有效 |
| Math | 算术/语义（28+15=42 语言上 fluent） | 低 | Cohen's d ≈ 0.05 | 无效（理论预测） |

这把一个"我们没做好"的结果转化为"我们的理论准确预测了边界条件"，显著强化了论文的 intellectual contribution。

---

## 七、论文修改影响范围（最小化改动）

| Section | 改动内容 | 工作量 |
|---------|---------|-------|
| Introduction | 以悖论开篇；用"可检测性"替换"互补" | 中 |
| §4 Method | 添加"为什么 dLLM 适合 locate"的机制解释 | 小 |
| §5 Experiments | 重新呈现 Table 3 的消融解读；加入 Random locator 结果 | 中 |
| §6 Analysis | 升格为核心机制 section：Calibration plot、AUC、典型案例 | 大 |
| §7 Discussion | math 从"负面结果"升格为"理论预测的边界条件" | 小 |

---

## 八、一句话版本的新论文贡献

> We show that diffusion LMs, despite being weaker code generators than AR models, possess a unique ability to detect token-level errors in AR drafts through bidirectional confidence estimation — an ability that AR models systematically lack. CoCoder exploits this asymmetry: let the AR model do what it is good at (generating coherent code), and let the dLLM do what it is good at (finding where the AR model went wrong).

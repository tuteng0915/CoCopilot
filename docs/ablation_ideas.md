# 消融实验与小实验想法（CoCoder）

这份文档记录“容易做、能出结论”的 ablation / model study 点子，避免只停留在口头讨论。

## A. Mask 粒度（已实现）

目标：验证“定位/重写”的粒度对最终通过率与稳定性的影响。

### A1. Diffusion remask（DreamCoder）

脚本：`python -m coder.scripts.gen_remask`

参数：

- `--mask_granularity {token,span,line}`
- `--span_merge_gap K`（仅 span）

解释：

- **token**：只 mask 低置信 token（最细粒度，默认旧行为）
- **span**：把相邻/近邻的 mask 合并成较连续的片段（减少碎片化重写）
- **line**：只要某行命中低置信 token，就 mask 整行（更偏“结构化编辑”）

建议 sweep：

- `mask_granularity ∈ {token, span, line}`
- `span_merge_gap ∈ {0, 1, 2, 4}`
- 在对齐 “masked token 数量/比例” 的前提下比较（否则不公平）

### A2. Locator→AR targeted rewrite（已实现）

脚本：`python -m coder.scripts.gen_locate_ar_rewrite`

参数：

- `--mask_granularity {token,span,line}`（默认 `span`）
- `--span_merge_gap K`
- `--confidence_threshold` 或 `--mask_ratio`（二选一）

注意：这个 baseline 不是硬约束编辑（AR 可能会改动未 mask 部分），但通过 prompt 强约束“尽量保持不变”可以当作合理 ablation。

## B. AR token logprob（建议做的 model study）

动机：当前 `gen_rerank.py` 用的是启发式 `score_candidate()`，而更合理的是使用 **AR 自己对候选的对数似然** 作为可解释的打分。

两种常见做法：

1. **Self-scoring（同模型打分）**
   - 采样得到多个候选 `c_i`
   - 计算 `log p_AR(c_i | prompt)`（或按 token 平均）
   - 选择最大者作为 best-of-n

2. **Cross-scoring（异模型/辅助模型打分）**
   - 采样来自模型 A（更会“发散”）
   - 打分来自模型 B（更“稳/守规矩”）

实现提示（待做）：

- 对 HF `AutoModelForCausalLM`：
  - 拼接 `prompt_ids + completion_ids`
  - 用 teacher forcing 取每个 completion token 的 `log_softmax`
  - 汇总为 sum / mean（建议 report 两者）
- 对 API 模型：
  - 若提供 logprobs 接口（因服务商而异），可直接取；否则只能用本地模型做 proxy

建议报告：

- `sum_logprob`、`avg_logprob_per_token`
- length normalization 是否引入偏好（短输出更占优）

## C. Reflexion baseline（Shinn et al., 2023）（已实现简化版）

脚本：`python -m coder.scripts.gen_reflexion`

简化流程（默认 1 轮）：

- 输入：problem + previous attempt（draft）
- 产出：`reflection`（不输出代码）→ `revised`（只输出代码）
- 输出 JSONL：保留 `draft_completion`，写 `raw_completion/solution` 为修订版，同时保存 `reflexion_trace`（逐轮记录）

可选项：

- `--rounds T`：多轮 reflexion（默认 1）
- `--feedback_key KEY`：如果输入 JSONL 里包含某种“失败反馈”，可把该字段拼进 reflection prompt（支持 dotted key，如 `eval.error`）

## C. Pending：后续可做但先不实现（记录）

### C1. 多轮局部修补（T 轮）

定位→rewrite→再定位→再rewrite，观察 T=1/2/3 的收益与退化。

### C2. 组合方向性

- AR→diffusion(remask)
- diffusion(locator)→AR(rewrite)
- diffusion→AR(self-refine)

### C3. Prompt / 输出格式鲁棒性

- code fence、system prompt 强度、max_new_tokens 截断敏感性

### C4. “编辑幅度 vs 成功率”分析图

统计每题 mask 数、diff 距离、是否通过，寻找最佳编辑幅度区间。


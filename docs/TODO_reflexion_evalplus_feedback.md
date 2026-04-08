> **实现已完成**：详见 [`docs/done/reflexion_evalplus_feedback.md`](done/reflexion_evalplus_feedback.md)。
> 待跑实验追踪见 [`docs/experiments-tracker.md`](experiments-tracker.md) §待跑数据。

# [已归档] TODO：EvalPlus「真实失败反馈」Reflexion Baseline

目标：实现更贴近 Reflexion（Shinn et al., 2023）的 baseline —— **使用真实执行/评测失败反馈**驱动 reflection→revise（先只在 EvalPlus 上实现）。

## 总体流程（建议）

1. 生成 samples（AR / 其他 baseline）
2. `postprocess_evalplus.py` sanitize
3. `eval_evalplus.py` 跑评测，并产出 **可解析的 per-task 失败信息**
4. 抽取失败反馈 → `feedback.jsonl`（按 `task_id` 对齐）
5. `gen_reflexion.py` 读取 `feedback.jsonl`，将反馈拼进 reflection prompt，产出修订版 samples

---

## 待办清单（按依赖顺序）

### 1) EvalPlus 失败反馈抽取器（核心）✅

- **任务**：从 EvalPlus 评测产物中抽取每个 `task_id` 的失败摘要，生成 `feedback.jsonl`（一行一个 task）。
- **输入**：
  - `eval_results.json`（`evalplus.evaluate` 的输出；`src/coder/scripts/eval_evalplus.py` 会帮你定位/搬运）
  - 可选：`--test-details` 生成的更详细失败信息（如果你启用）
- **输出**：`outputs/<name>.evalplus_feedback.jsonl`（建议命名）
- **建议输出 schema**：
  - `task_id`
  - `passed_base`（bool）
  - `base_status`（string）
  - `failure_summary`（string，尽量短但包含关键信号：异常类型/断言信息/超时/语法错误等）
  - `raw_details`（可选，保留原始字段便于 debug）

> 备注：这一块需要兼容不同版本 evalplus 的结果文件命名/字段差异。

### 2) 扩展 `gen_reflexion.py`：自动 join 失败反馈（推荐）✅

- **任务**：在 `src/coder/scripts/gen_reflexion.py` 新增参数：
  - `--feedback_file outputs/...feedback.jsonl`
  - `--feedback_field failure_summary`（或固定用 `failure_summary`）
- **行为**：
  - 按 `task_id` join
  - 将反馈插入 reflection prompt 的 `[Feedback]` 段
  - 仍保留 `--feedback_key` 作为“手工模式/兼容模式”（可选）
- **输出**：
  - 在每条样本里新增（或补充）`gen.feedback` / `eval_feedback` 字段，便于后续分析
  - 保持现有 timing/trace 输出（`reflexion_trace` + `<out>.timing_summary.json`）

### 3) 扩展 `eval_evalplus.py`：稳定产出可解析失败细节（必要时）

- **任务**：检查 `src/coder/scripts/eval_evalplus.py` 在开启 `--test_details` 时的产物是否稳定、路径是否可预测。
- **输出**：
  - 若 evalplus 版本差异导致输出不稳定，wrapper 需要做“探测/兼容”
  - 最终保证抽取器（TODO 1）能稳定拿到失败信息

### 4) 文档与命令模板 ✅

- **任务**：将最终可复现命令写入 `docs/runbook.md`（或另起一节），并在 `docs/ablation_ideas.md` 里更新 Reflexion 部分：
  - 产物命名约定
  - 推荐 pipeline（含 feedback 抽取与 Reflexion 修订）

---

## 备注：为什么先做 EvalPlus

- EvalPlus 有明确可执行的测试反馈（pass/fail + 可选 test details）
- 更符合 Reflexion 的“基于环境反馈迭代”设定
- LiveBench/LCB 的 scorer 接入更复杂，后续再扩展


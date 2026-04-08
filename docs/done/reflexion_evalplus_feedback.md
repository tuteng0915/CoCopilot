# [已完成] EvalPlus 真实失败反馈 Reflexion 实现

> 实现完成于 2026-03-30。命令模板见 `docs/runbook.md` §EvalPlus 失败反馈 -> Reflexion。

## 完成的实现项

### 1) EvalPlus 失败反馈抽取器 ✅

脚本：`python -m coder.analysis.evalplus_feedback`

- 输入：`*_eval_results.json`（`eval_evalplus.py` 产物）
- 输出：`*.evalplus_feedback.jsonl`，字段：`task_id`、`passed_base`、`base_status`、`base_status_counts`、`failure_summary`（可选 `raw_details`）
- 支持不同版本 evalplus 输出字段的兼容

### 2) gen_reflexion.py 扩展：自动 join 失败反馈 ✅

新增参数：
- `--feedback_file`：外部反馈文件路径
- `--feedback_field`：反馈字段名（默认 `failure_summary`）

行为：按 `task_id` join，将反馈插入 reflection prompt 的 `[Feedback]` 段；输出保留 `gen.feedback` / `eval_feedback` 字段及 `reflexion_trace`。

### 3) eval_evalplus.py 稳定性检查 ✅（按需）

evalplus 输出路径探测（`_eval_results.json` 与 `-eval_results.json` 两种命名）已在 wrapper 中处理。

### 4) 文档与命令模板 ✅

- `docs/runbook.md`：已补充完整 pipeline 命令
- `docs/ablation_ideas.md`：已更新 Reflexion 部分

## 仍待跑的实验（非实现问题）

- [ ] HumanEval / MBPP 各跑一组 `*_reflexion_feedback*.jsonl`，汇总 pass@1 + latency
  - 追踪见 `docs/experiments-tracker.md` §待跑数据

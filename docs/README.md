# CoCoder 运行手册（精简版）

这份 `docs/` 用来把”容易忘/容易踩坑”的运行上下文显式化，避免只靠聊天记忆导致重复试错。

## 你最常用的入口

### 代码任务

- EvalPlus（HumanEval/MBPP）：
  - 生成：`python -m coder.scripts.gen_evalplus`
  - sanitize：`python -m coder.scripts.postprocess_evalplus`
  - 评测：`python -m coder.scripts.eval_evalplus`
- LiveCodeBench / LiveBench-Coding（注意脚本名历史遗留）：
  - 生成：`python -m coder.scripts.gen_livebench --benchmark livecodebench|livebench-coding`
  - 评测：`python -m coder.scripts.eval_livebench --benchmark livecodebench|livebench-coding`
- BigCodeBench：
  - 生成：`python -m coder.scripts.gen_bigcodebench`
  - 评测包装：`python -m coder.scripts.eval_bigcodebench`

### 数学任务（泛化性验证）

为考察方案在代码之外的泛化能力，我们同时在数学推理任务上进行测试：

- GSM8K / MATH-500：
  - 生成：`python -m coder.scripts.gen_math --dataset gsm8k|math500`
  - 评测：`python -m coder.scripts.eval_math --samples <out.jsonl>`

## 活跃文档（需要关注）

- `docs/runbook.md`：环境、命令模板、产物命名约定
- `docs/tmux.md`：如何稳定挂长任务（以及为什么”迁移”只能重启）
- `docs/pitfalls.md`：已知坑位与对策（覆盖提示、字段缺失等）
- `docs/ablation_ideas.md`：消融实验与小实验想法（含 mask 粒度与 AR logprob 方向）
- `docs/completion-checklist.md`：判定某组实验是否跑完且可信
- `docs/experiments-tracker.md`：NeurIPS 2026 投稿前待跑实验汇总（待填论文表 + 附录分析）
- `docs/results.md`：**所有已跑实验的结果汇总表**（自动生成，更新命令：`python -m coder.scripts.gen_results_table`）

## 已归档文档（实现已完成，不再需要主动维护）

存放在 `docs/done/`：

- `docs/done/reflexion_evalplus_feedback.md`：EvalPlus 真实失败反馈 Reflexion 的实现说明
- `docs/done/impl_progress_2026-03.md`：2026-03-30 session 完成的实现进展

以下文件保留为历史参考，已有重定向说明：

- `docs/TODO_reflexion_evalplus_feedback.md` → 见 `done/reflexion_evalplus_feedback.md`
- `docs/rebuttal-experiments-tracker.md` → 见 `experiments-tracker.md`


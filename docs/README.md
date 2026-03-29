# CoCoder 运行手册（精简版）

这份 `docs/` 用来把“容易忘/容易踩坑”的运行上下文显式化，避免只靠聊天记忆导致重复试错。

## 你最常用的入口

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

## 推荐先读

- `docs/runbook.md`：环境、命令模板、产物命名约定
- `docs/tmux.md`：如何稳定挂长任务（以及为什么“迁移”只能重启）
- `docs/pitfalls.md`：已知坑位与对策（覆盖提示、字段缺失等）
- `docs/ablation_ideas.md`：消融实验与小实验想法（含 mask 粒度与 AR logprob 方向）
- `docs/TODO_reflexion_evalplus_feedback.md`：EvalPlus「真实失败反馈」Reflexion baseline 的待办与实现路线


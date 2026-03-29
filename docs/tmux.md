## 为什么需要 tmux

长任务（大模型生成/评测）非常容易因为断开连接、终端关闭、IDE 重启而中断。`tmux` 可以让任务在后台持续运行。

## 基本操作

- 列出会话：`tmux ls`
- 进入会话：`tmux attach -t <session>`
- 退出但不终止：`Ctrl+b` 然后 `d`
- 关闭会话（会杀掉其中进程）：`tmux kill-session -t <session>`

## 迁移注意事项（关键）

“把正在跑的非 tmux 任务迁移进 tmux”通常**做不到无损迁移**。

- **能做的**：停止旧进程，然后在 tmux 里用相同命令重启（进度会丢）
- **不能做的**：把已有进程直接“搬进” tmux（除非用很底层的 reparent/ptrace 技巧，本项目不做）

## 推荐的挂法

### 方案 A：直接在 tmux 里跑

```bash
tmux new-session -d -s myjob "bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate code && cd /model/tteng/CoCoder && export PYTHONPATH=\"/model/tteng/CoCoder/src:${PYTHONPATH:-}\" && <YOUR_CMD>'"
```

### 方案 B：tmux 里跑脚本 + 日志

把复杂命令封装进脚本（`outputs/base_tuteng/*.sh`），tmux 只负责启动脚本：

```bash
tmux new-session -d -s myjob "bash /model/tteng/CoCoder/outputs/base_tuteng/run_one_model_missing.sh deepseek 0"
```

这种方式的好处：
- 命令更短，不易因为引号/变量展开出错
- 日志路径更可控


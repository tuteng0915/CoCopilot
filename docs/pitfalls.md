## 常见坑位与对策

这份文件记录我们已经踩过的坑，后续遇到相似症状优先来这里对照。

## LiveCodeBench 评测：`n_scored=0` / `accuracy=None`

### 症状

- `outputs/base_tuteng/*_livecodebench_judgments.jsonl` 有 1055 行
- 但 `*_livecodebench_summary.json` 里：
  - `n_scored: 0`
  - `accuracy: null`
  - `by_task: {}`

### 根因

LiveCodeBench 官方 scorer 期望题目 dict 里存在 `original_json` 字段；缺失会触发：

- `KeyError: 'original_json'`

于是每题 `score=None`，最终 summary 里 `n_scored=0`。

### 修复

已在 `src/coder/scripts/eval_livebench.py` 中修复：加载题目时补充 `row["original_json"]=original`。

修复后可直接重跑：

```bash
python -m coder.scripts.eval_livebench --benchmark livecodebench \
  --samples outputs/base_tuteng/<model>_livecodebench.jsonl \
  --out_judgments outputs/base_tuteng/<model>_livecodebench_judgments.jsonl \
  --out_summary outputs/base_tuteng/<model>_livecodebench_summary.json
```

## BigCodeBench 评测卡住：提示覆盖 `*_eval_results.json`

### 症状

在 tmux/日志里看到类似：

`<path>_eval_results.json already exists. Press [Y/N] to overwrite or exit...`

这会让任务在无人值守时**永久卡住**。

### 对策

- **在启动前**清理旧的 `*_eval_results.json`
- 或者避免调用会弹交互提示的评测命令（优先用我们包装脚本/评测脚本写 summary）

示例（清理）：

```bash
rm -f outputs/base_tuteng/*_bigcodebench_instruct_*_eval_results.json
```

## tmux 会话“秒退”

### 常见原因

- 启动命令里引号层级太复杂、变量在外层被提前展开，导致最终命令语法错误
- 命令一开始就报错退出，但没有写日志，所以看不到原因

### 对策

- 优先写成脚本再 tmux 启动脚本
- 或在 tmux 命令中强制 `2>&1 | tee -a <log>`，并在末尾 `sleep` 保持会话存在以便排错

## DiffuLLaMA（diffusionfamily/diffullama）生成参数对不齐

### 现象

- 输出大量重复/截断/不按 prompt（HumanEval 甚至 0 pass@1）

### 官方 quick-start 参数（HKUNLP/DiffuLLaMA）

`inf_diffullama.py` 默认：
- `diffusion_steps=64`
- `logits_temp=0.9`
- `topp_temp=0.9`
- `shift=True`（官方标注“不要改”）
- 条件生成需要 `src_mask` 固定 prefix

### 对策

适配器应尽量：
- 传入 `src_mask`（若模型接口支持）
- 使用接近官方的 steps/temp/top_p（steps 是去噪步数，不是 max_new_tokens）



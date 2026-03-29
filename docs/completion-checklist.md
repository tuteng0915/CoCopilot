## 完成判定清单（outputs/base_tuteng）

这份清单用于“快速判断某个模型/benchmark 是否跑完并且结果可信”。

## EvalPlus：HumanEval / MBPP

以 `${MODEL}` 为例（例如 `qwen`、`deepseek`）。

### 生成完成

- **必须存在**：`outputs/base_tuteng/${MODEL}_${DATASET}.jsonl`

### sanitize 完成

- **必须存在**：`outputs/base_tuteng/${MODEL}_${DATASET}-sanitized.jsonl`

### 评测完成（本地）

- **必须存在**：`outputs/base_tuteng/${MODEL}_${DATASET}-sanitized_eval_results.json`
- **必须存在**：`outputs/base_tuteng/${MODEL}_${DATASET}_summary.json`

### 结果可信的最小检查

打开 `*_summary.json` 看：
- `summary.n_tasks` 与数据集题目数一致（HumanEval=164，MBPP=378）
- `summary.n_samples_total` 与 `n_tasks` 一致（单样本）

## LiveCodeBench（benchmark=livecodebench）

### 生成完成

- **必须存在**：`outputs/base_tuteng/${MODEL}_livecodebench.jsonl`

### 评测完成

- **必须存在**：`outputs/base_tuteng/${MODEL}_livecodebench_judgments.jsonl`
- **必须存在**：`outputs/base_tuteng/${MODEL}_livecodebench_summary.json`

### 结果可信的最小检查

- `*_livecodebench_judgments.jsonl` 行数约等于题目数（当前 test=1055）
- `*_livecodebench_summary.json`：
  - `n_scored > 0`
  - `accuracy` 非空

如果 `judgments` 有 1055 行但 `n_scored=0`，优先看 `docs/pitfalls.md` 的 `original_json` 条目。

## BigCodeBench（split=instruct，subset=full/hard）

### 生成完成

- **必须存在**：`outputs/base_tuteng/${MODEL}_bigcodebench_instruct_${SUBSET}.jsonl`

### 评测完成（本地执行）

- **建议存在**：`outputs/base_tuteng/${MODEL}_bigcodebench_instruct_${SUBSET}_summary.json`

### 常见卡点

若看到提示 `Press [Y/N] to overwrite`，说明评测在等待交互输入，会卡死；见 `docs/pitfalls.md`。


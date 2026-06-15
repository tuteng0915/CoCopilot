# Spec: Oracle Locator 上界实验

> 对应 `docs/narrative_reframe.md` §四 实验 B。
> 量化"如果 locator 完美，CoCoder 能达到多少"。

---

## 目标

构造一个 Oracle Locator：精确 mask 掉 AR draft 与 CoCoder 成功修复版本之间**实际改变的 token**，然后让 dLLM rewrite，观察能否重现甚至超越 CoCoder 的结果。

这个实验回答：**localization 质量是当前 CoCoder 的主要瓶颈吗？**

| 结果 | 解读 |
|------|------|
| Oracle >> CoCoder（如 +5pp） | Locator 质量是主要瓶颈；改进 locator 是未来方向 |
| Oracle ≈ CoCoder | Locator 已接近最优；dLLM rewriter 本身是瓶颈 |

---

## 前提条件

```bash
source /home/tteng/miniconda3/etc/profile.d/conda.sh
conda activate code
cd /model/tteng/CoCoder
export PYTHONPATH=src
export OUTPUTS=outputs/base_tuteng
```

已有产物（Oracle 的"gold mask"来源）：
- `$OUTPUTS/deepseek_humaneval.jsonl`（AR draft，164 条）
- `$OUTPUTS/deepseek_dream_remask_humaneval_t0.9.jsonl`（CoCoder 成功修复的版本）
- 对应 `_eval_results.json`（知道哪些 task 是"draft failed, collab passed"）

---

## Phase 1：构造 Oracle Mask JSONL（无需 GPU）

### Step 1a：新建 `src/coder/scripts/gen_oracle_mask.py`

脚本逻辑：
1. 加载 AR draft JSONL（`deepseek_humaneval.jsonl`）
2. 加载 CoCoder 结果 JSONL（`deepseek_dream_remask_humaneval_t0.9.jsonl`）
3. 加载两者的 eval_results.json，筛选"draft failed & collab passed"的 task_id
4. 对每个这样的 task，对 draft completion 和 collab completion 做 token-level diff（用 difflib）
5. 输出一个"oracle mask JSONL"，格式与普通 draft JSONL 相同，但额外添加字段：
   - `oracle_mask_spans: list[tuple[int,int]]`（char-level，draft 中需要被 mask 的位置）

```python
#!/usr/bin/env python3
"""
gen_oracle_mask.py — Build an oracle-masked JSONL for the Oracle Locator experiment.

For each task where AR draft failed but CoCoder succeeded, compute the
character-level diff between the AR draft and the CoCoder solution.
The differing characters become the oracle mask.

Output JSONL has all fields from the AR draft JSONL, plus:
  oracle_mask_spans: list of [start, end] char-spans in raw_completion
"""
from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path


def char_diff_spans(a: str, b: str) -> list[tuple[int, int]]:
    """
    Return (start, end) char spans in `a` that differ from `b`.
    Uses SequenceMatcher to find replacement/deletion blocks in `a`.
    """
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    spans = []
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        if op != "equal":
            if a0 < a1:
                spans.append((a0, a1))
    return spans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar_input",     required=True, help="AR draft JSONL")
    ap.add_argument("--collab_input", required=True, help="CoCoder result JSONL")
    ap.add_argument("--ar_eval",      required=True, help="AR eval_results.json")
    ap.add_argument("--collab_eval",  required=True, help="CoCoder sanitized eval_results.json")
    ap.add_argument("--out",          required=True, help="Output oracle-masked JSONL")
    ap.add_argument("--min_diff_chars", type=int, default=1,
                    help="Skip pairs where diff < N chars (trivial)")
    ap.add_argument("--max_diff_chars", type=int, default=500,
                    help="Skip pairs where diff > N chars (too noisy)")
    args = ap.parse_args()

    # Load eval results: which tasks passed?
    def load_eval(path: str) -> dict[str, bool]:
        data = json.loads(Path(path).read_text())
        result = {}
        for task_id, info in data.items():
            if isinstance(info, dict) and "eval" in info:
                result[task_id] = info["eval"].get("base_status") == "pass"
        return result

    ar_pass    = load_eval(args.ar_eval)
    collab_pass = load_eval(args.collab_eval)

    # Load JSONL records
    ar_recs     = {json.loads(l)["task_id"]: json.loads(l)
                   for l in Path(args.ar_input).read_text().splitlines() if l.strip()}
    collab_recs = {json.loads(l)["task_id"]: json.loads(l)
                   for l in Path(args.collab_input).read_text().splitlines() if l.strip()}

    # Eligible: AR failed, CoCoder passed
    eligible = {tid for tid in ar_recs
                if not ar_pass.get(tid, True) and collab_pass.get(tid, False)}
    print(f"Eligible tasks (AR fail → Collab pass): {len(eligible)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w") as fout:
        for tid in sorted(ar_recs.keys()):
            rec = dict(ar_recs[tid])
            if tid in eligible:
                ar_comp     = rec.get("raw_completion", "") or rec.get("solution", "")
                collab_comp = collab_recs[tid].get("raw_completion", "") or collab_recs[tid].get("solution", "")
                spans = char_diff_spans(ar_comp, collab_comp)
                diff_len = sum(e - s for s, e in spans)
                if diff_len < args.min_diff_chars or diff_len > args.max_diff_chars:
                    rec["oracle_mask_spans"] = None
                    rec["oracle_diff_chars"]  = diff_len
                else:
                    rec["oracle_mask_spans"] = spans
                    rec["oracle_diff_chars"]  = diff_len
                    written += 1
            else:
                rec["oracle_mask_spans"] = None
                rec["oracle_diff_chars"]  = 0
            fout.write(json.dumps(rec) + "\n")

    print(f"Written: {written} tasks with oracle mask spans")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
```

### Step 1b：运行脚本

```bash
python -m coder.scripts.gen_oracle_mask \
  --ar_input     $OUTPUTS/deepseek_humaneval.jsonl \
  --collab_input $OUTPUTS/deepseek_dream_remask_humaneval_t0.9.jsonl \
  --ar_eval      $OUTPUTS/deepseek_humaneval-sanitized_eval_results.json \
  --collab_eval  $OUTPUTS/deepseek_dream_remask_humaneval_t0.9-sanitized_eval_results.json \
  --out          $OUTPUTS/deepseek_humaneval_oracle_mask.jsonl \
  --min_diff_chars 1 \
  --max_diff_chars 500

python -m coder.scripts.gen_oracle_mask \
  --ar_input     $OUTPUTS/deepseek_mbpp.jsonl \
  --collab_input $OUTPUTS/deepseek_dream_remask_mbpp_t0.9.jsonl \
  --ar_eval      $OUTPUTS/deepseek_mbpp-sanitized_eval_results.json \
  --collab_eval  $OUTPUTS/deepseek_dream_remask_mbpp_t0.9-sanitized_eval_results.json \
  --out          $OUTPUTS/deepseek_mbpp_oracle_mask.jsonl \
  --min_diff_chars 1 \
  --max_diff_chars 500
```

产物：
- `deepseek_humaneval_oracle_mask.jsonl`（`oracle_mask_spans` 字段 non-null 的任务 = oracle-eligible tasks）
- `deepseek_mbpp_oracle_mask.jsonl`

---

## Phase 2：实现 Oracle Locator（无需 GPU）

Oracle Locator 从 JSONL 中读取预计算的 `oracle_mask_spans`，直接返回：mask 位置 confidence=0，其余 confidence=1。

### Step 2a：新建 `src/coder/locators/oracle_locator.py`

```python
from __future__ import annotations

import numpy as np
import torch

from coder.locators.base import TokenLocator


class OracleLocator(TokenLocator):
    """
    Pre-computed oracle locator: reads oracle_mask_spans from a side JSONL.

    For each task_id, masks exactly the characters that differ between the
    AR draft and the CoCoder-corrected solution.

    Usage: pass this locator to gen_remask via --locator oracle, with the
    oracle JSONL provided as --locator_model_id <path_to_oracle_jsonl>.
    """

    def __init__(self, oracle_jsonl: str) -> None:
        import json
        self._spans: dict[str, list[tuple[int, int]] | None] = {}
        for line in open(oracle_jsonl):
            rec = json.loads(line)
            tid = rec["task_id"]
            spans = rec.get("oracle_mask_spans")
            self._spans[tid] = [(s, e) for s, e in spans] if spans else None
        print(f"[oracle_locator] loaded {len(self._spans)} tasks, "
              f"{sum(v is not None for v in self._spans.values())} with oracle spans")

    def score_for_task(
        self,
        task_id: str,
        draft_text: str,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Extended API: caller passes task_id to look up oracle spans."""
        spans = self._spans.get(task_id)
        n = len(draft_text)
        if not spans:
            # No oracle for this task: return all-confident (no masking)
            confidence = np.ones(n, dtype=np.float32)
        else:
            # confidence=1 everywhere, then set oracle positions to 0
            confidence = np.ones(n, dtype=np.float32)
            for s, e in spans:
                if s < n and e <= n:
                    confidence[s:e] = 0.0

        # Character-level spans (one per character)
        char_spans = [(i, i + 1) for i in range(n)]
        return confidence, char_spans

    @torch.inference_mode()
    def score(
        self,
        prompt_text: str,
        draft_text: str,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        # Without task_id, fall back to all-confident (no masking)
        # gen_remask should call score_for_task() when using Oracle
        n = len(draft_text)
        return np.ones(n, dtype=np.float32), [(i, i+1) for i in range(n)]
```

### Step 2b：注册到 `locators/__init__.py` 和 `build_locator()`

```python
from coder.locators.oracle_locator import OracleLocator

# 在 build_locator() 中添加：
if name == "oracle":
    if not model_id:
        raise ValueError("--locator oracle requires --locator_model_id <path_to_oracle_jsonl>")
    return OracleLocator(oracle_jsonl=model_id)
```

### Step 2c：更新 `gen_remask.py` 支持 task_id 传递给 Oracle

在 `gen_remask.py` 的主循环中，如果 `locator` 是 `OracleLocator`，用 `locator.score_for_task(task_id, draft_text)` 替代 `locator.score(prompt, draft_text)`。在调用 `build_locator()` 之后，检查：

```python
is_oracle = hasattr(external_locator, "score_for_task")

# 在每个 task 的处理代码中：
if is_oracle:
    ext_conf_np, ext_spans = external_locator.score_for_task(task_id, raw_completion)
else:
    ext_conf_np, ext_spans = external_locator.score(prompt_text, raw_completion)
```

---

## Phase 3：运行 Oracle Locator 实验（需要 GPU）

```bash
# HumanEval
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_remask \
  --locator oracle \
  --locator_model_id $OUTPUTS/deepseek_humaneval_oracle_mask.jsonl \
  --refiner dream \
  --input $OUTPUTS/deepseek_humaneval.jsonl \
  --out $OUTPUTS/deepseek_oracle_locate_dream_rewrite_humaneval.jsonl \
  --temperature 0.1 --top_p 0.95 --seed 3407 \
  --device cuda:0 --resume

# MBPP
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_remask \
  --locator oracle \
  --locator_model_id $OUTPUTS/deepseek_mbpp_oracle_mask.jsonl \
  --refiner dream \
  --input $OUTPUTS/deepseek_mbpp.jsonl \
  --out $OUTPUTS/deepseek_oracle_locate_dream_rewrite_mbpp.jsonl \
  --temperature 0.1 --top_p 0.95 --seed 3407 \
  --device cuda:0 --resume
```

Sanitize + Evaluate：

```bash
for DATASET in humaneval mbpp; do
  python -m coder.scripts.postprocess_evalplus \
    --dataset $DATASET \
    --samples $OUTPUTS/deepseek_oracle_locate_dream_rewrite_${DATASET}.jsonl

  python -m coder.scripts.eval_evalplus --backend local --dataset $DATASET \
    --samples $OUTPUTS/deepseek_oracle_locate_dream_rewrite_${DATASET}-sanitized.jsonl
done
```

---

## 验收标准

| 文件 | 期望 |
|------|------|
| `deepseek_humaneval_oracle_mask.jsonl` | 164 行，部分有 `oracle_mask_spans` |
| `deepseek_oracle_locate_dream_rewrite_humaneval_summary.json` | n_tasks=164 |
| `deepseek_oracle_locate_dream_rewrite_mbpp_summary.json` | n_tasks=378 |

记录到 `results.md` 对比表（见 `spec_random_locator.md`）。

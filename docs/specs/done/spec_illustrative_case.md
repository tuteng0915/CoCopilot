# Spec: 典型案例——"AR 盲区，dLLM 可见"

> 对应 `docs/narrative_reframe.md` §四 分析 C。
> 无需 GPU，从现有产物手工/脚本筛选。论文最有说服力的定性证据。

---

## 目标

找到满足以下条件的**一个典型代码样本**，在论文 §6 Analysis section 中用代码框展示：

1. **AR draft 失败**（test case 不通过）
2. **CoCoder 成功**（修复后通过）
3. **改动 ≤ 5 tokens**（"surgical" fix，便于展示）
4. **AR logprob 对那个错误 token 高置信**（>0.8）
5. **dLLM 对那个 token 低置信**（<0.3）
6. 错误类型直觉上可解释：右侧代码已经"暗示"了正确答案，只有看全局才能发现错误

这样的案例视觉上是：
```python
# AR draft（错误，AR 自己看不出）：
for i in range(n + 1):   ← AR logprob: 0.92；dLLM conf: 0.04

# CoCoder 修复（正确）：
for i in range(n):
```

---

## Phase 1：脚本筛选候选案例（无需 GPU）

### Step 1a：新建筛选脚本

```bash
python3 - <<'EOF'
import json, pathlib, difflib

OUTPUTS = pathlib.Path("outputs/base_tuteng")

# 加载 AR 草稿 + eval 结果
ar_recs = {}
for line in (OUTPUTS / "deepseek_humaneval.jsonl").read_text().splitlines():
    if line.strip():
        r = json.loads(line)
        ar_recs[r["task_id"]] = r

ar_eval = json.loads((OUTPUTS / "deepseek_humaneval-sanitized_eval_results.json").read_text())

# 加载 CoCoder 结果 + eval 结果
collab_recs = {}
for line in (OUTPUTS / "deepseek_dream_remask_humaneval_t0.9.jsonl").read_text().splitlines():
    if line.strip():
        r = json.loads(line)
        collab_recs[r["task_id"]] = r

collab_eval = json.loads((OUTPUTS / "deepseek_dream_remask_humaneval_t0.9-sanitized_eval_results.json").read_text())

def passed(ev, tid):
    return ev.get(tid, {}).get("eval", {}).get("base_status") == "pass"

# 筛选：AR 失败，CoCoder 成功，diff ≤ 20 chars
candidates = []
for tid in ar_recs:
    if not passed(ar_eval, tid) and passed(collab_eval, tid):
        draft    = ar_recs[tid].get("raw_completion", "") or ar_recs[tid].get("solution", "")
        corrected = collab_recs[tid].get("raw_completion", "") or collab_recs[tid].get("solution", "")
        sm = difflib.SequenceMatcher(None, draft, corrected, autojunk=False)
        diff_len = sum(a1-a0 for op, a0, a1, b0, b1 in sm.get_opcodes() if op != "equal")
        if 1 <= diff_len <= 20:
            candidates.append((diff_len, tid))

candidates.sort()
print(f"Found {len(candidates)} surgical candidates:")
for diff_len, tid in candidates[:20]:
    draft    = ar_recs[tid].get("raw_completion", "")
    corrected = collab_recs[tid].get("raw_completion", "")
    # Show diff
    sm = difflib.SequenceMatcher(None, draft, corrected, autojunk=False)
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        if op != "equal":
            print(f"  [{tid}] diff={diff_len}c  AR: {repr(draft[a0:a1])}  →  CoCoder: {repr(corrected[b0:b1])}")
EOF
```

### Step 1b：记录候选 task_id

从输出中选出：
- diff 很小（1–5 chars）
- 改动直觉上可解释（off-by-one, 边界条件, 错误变量名等）
- 非空白/注释改动

---

## Phase 2：验证置信度（需要 GPU，约 5 分钟）

对选出的候选 task_id，手动运行置信度检查。

### Step 2a：运行置信度查看脚本

```python
#!/usr/bin/env python3
"""查看某个 task 的 dLLM vs AR 置信度，找"AR 高置信但 dLLM 低置信"的错误 token"""
import json, torch, pathlib

OUTPUTS = pathlib.Path("outputs/base_tuteng")
TARGET_TASK = "HumanEval/46"  # 替换为候选 task_id

# 加载数据
ar_recs = {json.loads(l)["task_id"]: json.loads(l)
           for l in (OUTPUTS / "deepseek_humaneval.jsonl").read_text().splitlines() if l.strip()}
collab_recs = {json.loads(l)["task_id"]: json.loads(l)
               for l in (OUTPUTS / "deepseek_dream_remask_humaneval_t0.9.jsonl").read_text().splitlines() if l.strip()}

rec = ar_recs[TARGET_TASK]
prompt   = rec["prompt"]
draft    = rec.get("raw_completion", "") or rec.get("solution", "")
corrected = collab_recs[TARGET_TASK].get("raw_completion", "") or collab_recs[TARGET_TASK].get("solution", "")

print("=== Draft ===")
print(draft[:500])
print("\n=== Corrected ===")
print(corrected[:500])

import difflib
sm = difflib.SequenceMatcher(None, draft, corrected, autojunk=False)
print("\n=== Diff ===")
for op, a0, a1, b0, b1 in sm.get_opcodes():
    if op != "equal":
        print(f"  AR[{a0}:{a1}] = {repr(draft[a0:a1])}  →  {repr(corrected[b0:b1])}")
        fault_char_range = (a0, a1)

# dLLM confidence
from coder.models.dream_coder import DreamCoder
dream = DreamCoder(model_id="Dream-org/Dream-Coder-v0-Instruct-7B", device="cuda:0")
tok = dream.tok

comp_ids   = tok(draft,  add_special_tokens=False, return_tensors="pt")["input_ids"].to(dream.device)
prompt_ids = tok(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(dream.device)
with torch.no_grad():
    dllm_conf = dream.score_tokens(prompt_ids, comp_ids).float().cpu().numpy()

# AR confidence
from coder.locators.ar_locator import ARLocator
ar_loc = ARLocator(model_id="deepseek-ai/deepseek-coder-6.7b-instruct", device="cuda:0")
ar_conf, ar_spans = ar_loc.score(prompt, draft)

# 打印每个 token 的置信度，高亮 fault 位置
enc = tok(draft, add_special_tokens=False, return_offsets_mapping=True)
spans = enc.get("offset_mapping", [])
print("\n=== Per-token Confidence (dLLM | AR) ===")
print(f"{'Token':>20}  {'dLLM':>6}  {'AR':>6}  {'FAULT':>5}")
for i, (tok_id, (cs, ce)) in enumerate(zip(enc["input_ids"], spans)):
    tok_str = tok.decode([tok_id])
    is_fault = fault_char_range[0] <= cs < fault_char_range[1]
    d_conf = float(dllm_conf[i]) if i < len(dllm_conf) else None
    # align AR conf
    a_conf_vals = [float(ar_conf[si]) for si,(ss,se) in enumerate(ar_spans) if se>cs and ss<ce]
    a_conf = sum(a_conf_vals)/len(a_conf_vals) if a_conf_vals else None
    flag = "**FAULT**" if is_fault else ""
    print(f"{repr(tok_str):>20}  {d_conf or 0:>6.3f}  {a_conf or 0:>6.3f}  {flag}")
```

```bash
CUDA_VISIBLE_DEVICES=<GPU> python3 check_token_conf.py
```

### Step 2b：筛选标准

选择满足以下条件的 task_id：
- Fault token 的 **dLLM conf < 0.3**
- Fault token 的 **AR conf > 0.7**
- 差值 > 0.4（dLLM 和 AR 的置信度差异显著）
- 改动有直观解释性（能用一句话解释为什么 AR 看不到但 dLLM 能看到）

已知好候选（来自 `locator_scoring_clean_t09_deepseek.log`）：
- `HumanEval/46`：AR conf=0.986，dLLM conf=0.01（最好的案例，diff=1c）
- `HumanEval/64`：AR conf=0.686，dLLM conf=不明
- `Mbpp/809`：AR conf=0.999，dLLM conf=0.003（diff=1c）

**优先查看这三个**。

---

## Phase 3：撰写 Paper 展示格式

找到案例后，用以下格式在 paper §6 中展示（以 HumanEval/46 为假设案例）：

```latex
\begin{figure}[t]
  \centering
  \begin{lstlisting}[language=Python, basicstyle=\small\ttfamily]
# AR draft (fails: incorrect boundary)
def count_upper(s):
    count = 0
    for i in range(len(s) + 1):  # AR conf: 0.99; dLLM conf: 0.01
        if s[i].isupper():
            count += 1
    return count

# CoCoder correction (passes: boundary fixed)
def count_upper(s):
    count = 0
    for i in range(len(s)):      # dLLM masks "+ 1"; Dream-Coder removes it
        ...
  \end{lstlisting}
  \caption{An error invisible to the AR model but visible to dLLM.
    The token \texttt{+~1} is locally plausible (follows a common \texttt{range(n+1)}
    idiom) but globally inconsistent with subsequent indexing \texttt{s[i]}.
    The AR logprob assigns confidence 0.99; dLLM's bidirectional attention,
    seeing the downstream \texttt{s[i]} access, assigns 0.01.}
\end{figure}
```

---

## 产物列表

| 产物 | 内容 |
|------|------|
| `docs/illustrative_case.md` | 选定的 task_id、全文代码、置信度表格、paper 用 LaTeX 格式 |
| `outputs/ablation_locator/illustrative_case_conf.txt` | 完整 per-token 置信度输出 |

---

## 注意事项

1. **HumanEval/46 已知是好案例**：从现有 log 可知 AR conf=0.986，dLLM conf≈0.01。先从这个入手查看代码内容。
2. 如果 HumanEval 里没有足够直观的案例，同样查看 MBPP 的 Mbpp/809（AR conf=0.999，dLLM conf=0.003）。
3. 最终选案例的标准：**读者能在 10 秒内理解为什么 AR 看不到而 dLLM 看到了**。技术上完美但直觉上费解的案例不适合。

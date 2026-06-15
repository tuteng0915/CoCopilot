# Spec: SQL 生成可行性验证（Text-to-SQL）

> 轻量验证实验：确认 CoCoder "dLLM 通过双向置信度检测 AR 错误" 假说是否适用于 SQL 生成领域。
> **不需要完整 pipeline，只验证 fault detection ratio 是否成立。**

---

## 目标

用最小代价回答两个问题：

1. **AR 在 text-to-SQL 上犯什么样的错误？** 是否是"局部 fluent、全局违约"的结构性错误？
2. **dLLM locator 对 SQL 错误 token 的置信度是否显著低于正确 token？**（fault detection ratio >> 1）

验证成功标准：dLLM fault detection ratio ≥ 3×（对比 code 上的 23× HumanEval / 126× MBPP，SQL 预计居中）。

---

## 数据集：Spider dev set

| 属性 | 值 |
|------|-----|
| 规模 | dev: 1034 samples（取前 200 即可） |
| 评测指标 | Execution Accuracy（执行结果集匹配） |
| 数据库 | SQLite（.sqlite 文件，Python stdlib sqlite3，无需服务器） |
| Schema 格式 | JSON（tables + columns + foreign keys，随 dataset 提供） |

---

## 前提条件

```bash
source /home/tteng/miniconda3/etc/profile.d/conda.sh
conda activate code
cd /model/tteng/CoCoder
export PYTHONPATH=src
export OUTPUTS=outputs/base_tuteng
export SQL_DATA=outputs/sql_feasibility
mkdir -p $SQL_DATA
```

---

## Phase 0：下载数据（无需 GPU，~5 分钟）

```bash
# Spider dataset（含 SQLite .db 文件）
cd $SQL_DATA
wget -q https://drive.usercontent.google.com/download?id=1iRDVHLr4mX2wQKSgA9J8Pire73Jahh0m \
     -O spider.zip
unzip -q spider.zip
# 解压后结构：spider/dev.json  spider/database/*/  spider/tables.json
```

> **Fallback**（如 Google Drive 不可访问）：使用 Hugging Face `spider` dataset：
> ```python
> from datasets import load_dataset
> ds = load_dataset("spider", split="validation")
> ```

---

## Phase 1：写 SQL 执行 harness（无需 GPU，~30 分钟）

新建 `src/coder/scripts/sql_eval.py`：

```python
#!/usr/bin/env python3
"""
Minimal Spider execution accuracy evaluator.
Usage: python -m coder.scripts.sql_eval --pred pred.jsonl --spider_dir outputs/sql_feasibility/spider
"""
import json, sqlite3, pathlib, argparse, re
from typing import Optional


def execute_sql(db_path: str, sql: str) -> Optional[set]:
    """Execute SQL and return result as frozenset of sorted rows, or None on error."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return set(tuple(str(v).lower() for v in row) for row in rows)
    except Exception:
        return None


def schema_to_text(tables_info: dict) -> str:
    """Convert Spider tables.json entry to a compact schema string for prompting."""
    lines = []
    col_names = tables_info["column_names_original"]  # [(table_idx, col_name), ...]
    col_types = tables_info["column_types"]
    table_names = tables_info["table_names_original"]

    for t_idx, t_name in enumerate(table_names):
        cols = [(cn, ct) for (ti, cn), ct in zip(col_names, col_types) if ti == t_idx]
        col_str = ", ".join(f"{cn} {ct}" for cn, ct in cols)
        lines.append(f"CREATE TABLE {t_name} ({col_str});")

    fks = tables_info.get("foreign_keys", [])
    for c1_idx, c2_idx in fks:
        _, c1 = col_names[c1_idx]
        t1 = table_names[col_names[c1_idx][0]]
        _, c2 = col_names[c2_idx]
        t2 = table_names[col_names[c2_idx][0]]
        lines.append(f"-- FK: {t1}.{c1} -> {t2}.{c2}")

    return "\n".join(lines)


def make_prompt(schema_text: str, question: str) -> str:
    return (
        "Given the following SQLite database schema:\n"
        f"{schema_text}\n\n"
        f"Write a single SQL query to answer: {question}\n"
        "Output only the SQL query, no explanation.\n"
    )


def extract_sql(raw: str) -> str:
    """Strip markdown code fences and leading/trailing whitespace."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:sql)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred",       required=True, help="JSONL with task_id, pred_sql, gold_sql, db_id")
    ap.add_argument("--spider_dir", required=True, help="Path to spider/ directory")
    args = ap.parse_args()

    spider = pathlib.Path(args.spider_dir)
    n_pass = n_total = 0
    for line in pathlib.Path(args.pred).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        db_path = spider / "database" / rec["db_id"] / f"{rec['db_id']}.sqlite"
        pred_result = execute_sql(str(db_path), extract_sql(rec["pred_sql"]))
        gold_result = execute_sql(str(db_path), rec["gold_sql"])
        passed = (pred_result is not None) and (pred_result == gold_result)
        rec["exec_pass"] = passed
        print(json.dumps(rec))
        n_pass   += int(passed)
        n_total  += 1
    print(f"\n=== Execution Accuracy: {n_pass}/{n_total} = {n_pass/n_total*100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
```

---

## Phase 2：AR 生成（需要 GPU，~30 分钟 for 200 samples）

新建 `src/coder/scripts/gen_sql_ar.py`：

```python
#!/usr/bin/env python3
"""
Generate SQL drafts from Spider dev set using an AR model.
Usage:
  python -m coder.scripts.gen_sql_ar \
    --spider_dir outputs/sql_feasibility/spider \
    --out        outputs/sql_feasibility/deepseek_spider_dev.jsonl \
    --model deepseek --n_samples 200
"""
import json, argparse, pathlib, sys
sys.path.insert(0, "src")

from coder.models.deepseek_coder import DeepSeekCoder


MODELS = {
    "deepseek": ("deepseek-ai/deepseek-coder-6.7b-instruct", DeepSeekCoder),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spider_dir", required=True)
    ap.add_argument("--out",        required=True)
    ap.add_argument("--model",      default="deepseek")
    ap.add_argument("--n_samples",  type=int, default=200)
    ap.add_argument("--device",     default="cuda:0")
    args = ap.parse_args()

    spider   = pathlib.Path(args.spider_dir)
    dev_data = json.loads((spider / "dev.json").read_text())
    tables   = {t["db_id"]: t for t in json.loads((spider / "tables.json").read_text())}

    from coder.scripts.sql_eval import schema_to_text, make_prompt
    model_id, ModelCls = MODELS[args.model]
    model = ModelCls(model_id=model_id, device=args.device)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as fout:
        for i, item in enumerate(dev_data[:args.n_samples]):
            db_id    = item["db_id"]
            question = item["question"]
            gold_sql = item["query"]
            schema   = schema_to_text(tables[db_id])
            prompt   = make_prompt(schema, question)

            raw = model.generate(prompt, temperature=0.0, max_new_tokens=256)
            rec = {
                "task_id":  f"spider/dev/{i}",
                "db_id":    db_id,
                "question": question,
                "gold_sql": gold_sql,
                "prompt":   prompt,
                "pred_sql": raw,
            }
            fout.write(json.dumps(rec) + "\n")
            if (i+1) % 20 == 0:
                print(f"  {i+1}/{args.n_samples}", flush=True)

    print(f"Done → {out_path}")


if __name__ == "__main__":
    main()
```

运行：

```bash
CUDA_VISIBLE_DEVICES=0 python -m coder.scripts.gen_sql_ar \
  --spider_dir $SQL_DATA/spider \
  --out        $SQL_DATA/deepseek_spider_dev.jsonl \
  --model deepseek --n_samples 200

# 执行评测，输出 exec_pass 字段
python -m coder.scripts.sql_eval \
  --pred       $SQL_DATA/deepseek_spider_dev.jsonl \
  --spider_dir $SQL_DATA/spider \
  > $SQL_DATA/deepseek_spider_dev_eval.jsonl
```

预期 DeepSeek-Coder AR baseline：**~45–60% execution accuracy**（Spider dev 上无 fine-tune 的零样本水平）。

---

## Phase 3：dLLM Fault Detection Ratio（需要 GPU，~1 小时）

对 "AR 预测错误" 的 samples 分析：哪些 token 出错（AR pred vs gold diff）？dLLM 置信度是否低？

新建 `src/coder/analysis/sql_locator_analysis.py`：

```python
#!/usr/bin/env python3
"""
Compute dLLM fault-detection ratio on SQL drafts.
Analogous to locator_scoring.py for code.

For each AR-failed sample:
  - Align AR pred_sql vs gold_sql via difflib to find "fault chars"
  - Score pred_sql tokens with dLLM locator
  - Compute mean confidence at fault vs non-fault positions

Output: fault_detection_ratio = mean_conf_nonfault / mean_conf_fault
Expected: >> 1 if dLLM detects SQL errors
"""
import json, pathlib, difflib, argparse, sys
import numpy as np
sys.path.insert(0, "src")


def char_fault_mask(pred: str, gold: str) -> np.ndarray:
    """Return boolean mask: True = this char in pred differs from gold."""
    mask = np.zeros(len(pred), dtype=bool)
    sm = difflib.SequenceMatcher(None, pred, gold, autojunk=False)
    for op, a0, a1, *_ in sm.get_opcodes():
        if op != "equal":
            mask[a0:a1] = True
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_jsonl", required=True,  help="Output of sql_eval.py (has exec_pass field)")
    ap.add_argument("--locator",    default="dream", choices=["dream", "ar", "bert"])
    ap.add_argument("--device",     default="cuda:0")
    ap.add_argument("--max_samples", type=int, default=100)
    args = ap.parse_args()

    # Load only AR-failed samples (exec_pass=False)
    failed = []
    for line in pathlib.Path(args.eval_jsonl).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if not rec.get("exec_pass", True) and rec.get("pred_sql") and rec.get("gold_sql"):
            failed.append(rec)
    print(f"AR-failed samples: {len(failed)}, using first {args.max_samples}")
    failed = failed[:args.max_samples]

    # Load locator
    if args.locator == "dream":
        from coder.locators.dream_locator import DreamLocator
        locator = DreamLocator(model_id="Dream-org/Dream-Coder-v0-Instruct-7B", device=args.device)
    elif args.locator == "ar":
        from coder.locators.ar_locator import ARLocator
        locator = ARLocator(model_id="deepseek-ai/deepseek-coder-6.7b-instruct", device=args.device)
    elif args.locator == "bert":
        from coder.locators.bert_locator import BertLocator
        locator = BertLocator(device=args.device)

    fault_confs    = []
    nonfault_confs = []
    results = []

    for rec in failed:
        prompt   = rec["prompt"]
        pred_sql = rec["pred_sql"].strip()
        gold_sql = rec["gold_sql"].strip()
        if not pred_sql:
            continue

        conf_arr, spans = locator.score(prompt, pred_sql)
        fault_mask = char_fault_mask(pred_sql, gold_sql)

        # Map char-level fault mask → span-level
        for i, (cs, ce) in enumerate(spans):
            if i >= len(conf_arr):
                break
            is_fault = fault_mask[cs:ce].any() if ce <= len(fault_mask) else False
            if is_fault:
                fault_confs.append(float(conf_arr[i]))
            else:
                nonfault_confs.append(float(conf_arr[i]))

        results.append({
            "task_id":      rec.get("task_id"),
            "pred_sql":     pred_sql[:100],
            "gold_sql":     gold_sql[:100],
            "n_fault_toks": int(fault_mask.sum()),
        })

    if not fault_confs:
        print("No fault tokens found.")
        return

    mean_fault    = np.mean(fault_confs)
    mean_nonfault = np.mean(nonfault_confs)
    ratio         = mean_nonfault / mean_fault if mean_fault > 0 else float("inf")

    print(f"\n=== dLLM Fault Detection Ratio ({args.locator}) ===")
    print(f"  Samples analyzed:       {len(results)}")
    print(f"  Total fault spans:      {len(fault_confs)}")
    print(f"  Total non-fault spans:  {len(nonfault_confs)}")
    print(f"  Mean conf @ fault:      {mean_fault:.4f}")
    print(f"  Mean conf @ non-fault:  {mean_nonfault:.4f}")
    print(f"  Fault detection ratio:  {ratio:.1f}×")
    print()
    print("Interpretation:")
    if ratio >= 3.0:
        print(f"  ✅ ratio={ratio:.1f}× → dLLM detects SQL errors; CoCoder SQL is feasible")
    elif ratio >= 1.5:
        print(f"  ⚠️  ratio={ratio:.1f}× → weak signal; marginal feasibility")
    else:
        print(f"  ❌ ratio={ratio:.1f}× → dLLM cannot detect SQL errors")


if __name__ == "__main__":
    main()
```

运行：

```bash
# dLLM locator
CUDA_VISIBLE_DEVICES=0 python -m coder.analysis.sql_locator_analysis \
  --eval_jsonl $SQL_DATA/deepseek_spider_dev_eval.jsonl \
  --locator dream --device cuda:0 --max_samples 100

# AR logprob 对比
CUDA_VISIBLE_DEVICES=0 python -m coder.analysis.sql_locator_analysis \
  --eval_jsonl $SQL_DATA/deepseek_spider_dev_eval.jsonl \
  --locator ar --device cuda:0 --max_samples 100
```

---

## Phase 4（可选）：跑完整 CoCoder pipeline

如果 Phase 3 ratio ≥ 3×，顺手跑一次 CoCoder end-to-end：

```bash
# 将 AR draft JSONL 转换为 gen_remask 能接受的格式（重命名字段）
python3 - <<'EOF'
import json, pathlib
out = []
for line in pathlib.Path("$SQL_DATA/deepseek_spider_dev.jsonl").read_text().splitlines():
    if not line.strip(): continue
    r = json.loads(line)
    r["raw_completion"] = r["pred_sql"]   # gen_remask 需要的字段名
    out.append(json.dumps(r))
pathlib.Path("$SQL_DATA/deepseek_spider_dev_remask_input.jsonl").write_text("\n".join(out))
EOF

CUDA_VISIBLE_DEVICES=0 python -m coder.scripts.gen_remask \
  --locator dream --refiner dream \
  --input  $SQL_DATA/deepseek_spider_dev_remask_input.jsonl \
  --out    $SQL_DATA/deepseek_dream_remask_spider_dev.jsonl \
  --confidence_threshold 0.9 --device cuda:0 --resume

# 执行评测 CoCoder 结果
python -m coder.scripts.sql_eval \
  --pred       $SQL_DATA/deepseek_dream_remask_spider_dev.jsonl \
  --spider_dir $SQL_DATA/spider \
  > $SQL_DATA/deepseek_dream_remask_spider_dev_eval.jsonl
```

---

## 结果解读

| 指标 | 预期值 | 意义 |
|------|--------|------|
| AR exec accuracy | 45–60% | DeepSeek zero-shot SQL 基线合理 |
| dLLM fault detection ratio | ≥ 3× | **可行性通过**；SQL 适合 CoCoder |
| AR logprob ratio | ~1×–1.5× | 对比基线，证明 dLLM 优势来自双向注意力 |
| CoCoder exec accuracy（Phase 4） | AR + 3–8pp | 端到端提升 |

---

## 产物

| 文件 | 内容 |
|------|------|
| `outputs/sql_feasibility/deepseek_spider_dev.jsonl` | AR SQL drafts（200 条） |
| `outputs/sql_feasibility/deepseek_spider_dev_eval.jsonl` | 加了 `exec_pass` 字段 |
| `outputs/sql_feasibility/sql_locator_analysis_dream.txt` | fault detection ratio 输出 |
| `outputs/sql_feasibility/sql_locator_analysis_ar.txt` | AR logprob 对比 |

记录结论到 `docs/results.md` 新增 §SQL Feasibility 小节。

---

## 注意事项

1. **DeepSeek-Coder 的 SQL 能力**：虽然没有专门 fine-tune，但 DeepSeek-Coder-6.7b-instruct 在 Spider dev 上零样本 exec accuracy 约 40–55%，足够产生足量 fail 案例（40–80 个 fail samples）供分析。
2. **Spider 数据集下载**：如果 Google Drive 不可用，用 `datasets` 库或 `git clone https://github.com/taoyds/spider`。
3. **Schema 注入格式**：`gen_remask` 的 `prompt` 字段会被直接传给 dLLM locator——确保 prompt 包含 schema 信息，这样 dLLM 才能利用全局约束。
4. **Dream-Coder vs Dream-general**：用 Dream-Coder（代码专用）而非 dream（通用），SQL 是代码的一种。

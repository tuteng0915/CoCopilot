# Spec: Locator Calibration 分析（Calibration Plot + ROC/AUC）

> 对应 `docs/narrative_reframe.md` §四 分析 A、B。
> 数据收集需要 GPU；绘图和结果汇总无需 GPU。将 23x ratio 转化为可视化图表和标准化指标。

---

## 目标

**分析 A — Calibration Plot**：按 dLLM 置信度分桶，展示每个桶内实际 fault token 的比例。
预期：单调递减曲线（低置信 → 高 fault 比例），直观展示 dLLM 置信度的 discriminativeness。

**分析 B — ROC / AUC**：把 locator 当 binary classifier，计算三种 locator（dLLM / AR / BERT）和 Random 的 ROC 曲线与 AUC。
原始预期：dLLM AUC >> AR ≈ BERT ≈ Random(0.5)。实际结果见下方"执行结果"；changed-token proxy 下 AR/CodeBERT 也有一定信号。

---

## 前提条件

```bash
source /home/wjzhang/miniforge3/etc/profile.d/conda.sh
conda activate cocoder
cd /home/wjzhang/tt_workspace/model/CoCoder/CoCoder
export PYTHONPATH=src
```

**运行环境**：项目统一使用 `cocoder` conda 环境；不要使用旧的 `elf` 或临时 `cocoder-calib` 环境。

**已有产物**：`outputs/ablation_locator/locator_scoring_clean_t09_deepseek.log`

**问题**：现有 `locator_scoring.py` 只在"surgical fault pairs"（max_diff_chars=10）上运行，只有 3–4 个样本对，样本量太小，无法画出可信的 calibration plot 和 ROC 曲线。

**解决方案**：扩展分析脚本，放宽 fault pair 标准，收集更多样本，获取完整的 per-token confidence 分布。实际运行时 strict `AR fail + CoCoder pass + changed draft` 样本仍偏少，因此使用 `--include_collab_fail` 纳入所有 AR failed 且实际发生 remask 改动的样本；标签解释为 changed-token proxy。

---

## Phase 1：扩展数据收集（需要 GPU）

### Step 1a：收集大批量 per-token confidence 数据

新建 `src/coder/analysis/locator_calibration_data.py`：

> 下面代码块是初始设计草稿。当前仓库实现以 `src/coder/analysis/locator_calibration_data.py` 为准，已补充 `--status_field`、`--include_collab_fail`、EvalPlus 结果解析兼容和 token-span 对齐。

```python
#!/usr/bin/env python3
"""
locator_calibration_data.py — Collect per-token confidence + fault labels
                               for calibration plot and ROC/AUC analysis.

Criteria for inclusion:
  - AR draft FAILED evaluation
  - CoCoder (dLLM remask) draft PASSED evaluation
  - No character limit on diff (unlike the "surgical" criterion)

For each such pair, compute per-token confidence from all three locators,
and label each token as fault (changed) or non-fault (unchanged).

Output: JSON file with per-token records for downstream analysis.
"""
from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch


@dataclass
class TokenRecord:
    task_id: str
    token_idx: int
    char_start: int
    char_end: int
    is_fault: bool           # True if this token differs in corrected solution
    dllm_confidence: Optional[float]
    ar_confidence: Optional[float]
    bert_confidence: Optional[float]


def get_fault_char_set(draft: str, corrected: str) -> set[int]:
    """Return set of character indices in draft that are 'fault' (differ from corrected)."""
    sm = difflib.SequenceMatcher(None, draft, corrected, autojunk=False)
    fault_chars: set[int] = set()
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        if op != "equal":
            fault_chars.update(range(a0, a1))
    return fault_chars


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar_input",     required=True)
    ap.add_argument("--collab_input", required=True)
    ap.add_argument("--ar_eval",      required=True)
    ap.add_argument("--collab_eval",  required=True)
    ap.add_argument("--out",          required=True, help="Output JSON with per-token records")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--locators", nargs="+",
                    choices=["dllm", "ar", "bert"], default=["dllm", "ar", "bert"])
    ap.add_argument("--dllm_model_id",
                    default="Dream-org/Dream-Coder-v0-Instruct-7B")
    ap.add_argument("--ar_model_id",
                    default="deepseek-ai/deepseek-coder-6.7b-instruct")
    ap.add_argument("--bert_model_id",
                    default="microsoft/codebert-base-mlm")
    ap.add_argument("--limit", type=int, default=0,
                    help="Limit number of pairs (0=all)")
    args = ap.parse_args()

    # Load eval results
    def load_pass(path: str) -> dict[str, bool]:
        data = json.loads(Path(path).read_text())
        out = {}
        for tid, info in data.items():
            if isinstance(info, dict) and "eval" in info:
                out[tid] = info["eval"].get("base_status") == "pass"
        return out

    ar_pass     = load_pass(args.ar_eval)
    collab_pass = load_pass(args.collab_eval)

    ar_recs     = {json.loads(l)["task_id"]: json.loads(l)
                   for l in Path(args.ar_input).read_text().splitlines() if l.strip()}
    collab_recs = {json.loads(l)["task_id"]: json.loads(l)
                   for l in Path(args.collab_input).read_text().splitlines() if l.strip()}

    eligible = sorted(
        tid for tid in ar_recs
        if not ar_pass.get(tid, True) and collab_pass.get(tid, False)
    )
    print(f"Eligible pairs: {len(eligible)}")
    if args.limit:
        eligible = eligible[:args.limit]
        print(f"Limited to: {len(eligible)}")

    # Lazy-load models
    dllm_model = ar_model = bert_model = None

    if "dllm" in args.locators:
        from coder.models.dream_coder import DreamCoder
        dllm_model = DreamCoder(model_id=args.dllm_model_id, device=args.device)

    if "ar" in args.locators:
        from coder.locators.ar_locator import ARLocator
        ar_model = ARLocator(model_id=args.ar_model_id, device=args.device)

    if "bert" in args.locators:
        from coder.locators.bert_locator import BERTLocator
        bert_model = BERTLocator(model_id=args.bert_model_id, device=args.device)

    records: list[TokenRecord] = []

    for i, tid in enumerate(eligible):
        print(f"[{i+1}/{len(eligible)}] {tid}")
        rec = ar_recs[tid]
        draft    = rec.get("raw_completion", "") or rec.get("solution", "")
        corrected = collab_recs[tid].get("raw_completion", "") or collab_recs[tid].get("solution", "")
        prompt   = rec.get("prompt", "")

        fault_chars = get_fault_char_set(draft, corrected)

        # Get confidence from each locator
        dllm_conf = ar_conf = bert_conf = None
        dllm_spans = ar_spans = bert_spans = None

        if dllm_model is not None:
            from coder.locators.base import get_token_char_spans
            comp_ids = dllm_model.tok(draft, add_special_tokens=False,
                                      return_tensors="pt")["input_ids"].to(dllm_model.device)
            prompt_ids = dllm_model.tok(prompt, add_special_tokens=False,
                                        return_tensors="pt")["input_ids"].to(dllm_model.device)
            dllm_conf_t = dllm_model.score_tokens(prompt_ids, comp_ids)
            dllm_conf   = dllm_conf_t.float().cpu().numpy()
            dllm_spans  = get_token_char_spans(dllm_model.tok, draft)

        if ar_model is not None:
            ar_conf_np, ar_spans = ar_model.score(prompt, draft)
            ar_conf = ar_conf_np

        if bert_model is not None:
            bert_conf_np, bert_spans = bert_model.score(prompt, draft)
            bert_conf = bert_conf_np

        # Use dLLM spans as reference tokenization; fall back to AR spans
        ref_spans = dllm_spans or ar_spans or bert_spans
        if ref_spans is None:
            continue

        for tok_idx, (cs, ce) in enumerate(ref_spans):
            is_fault = any(c in fault_chars for c in range(cs, ce))

            def conf_at(conf_arr, spans, idx):
                if conf_arr is None or spans is None or idx >= len(conf_arr):
                    return None
                return float(conf_arr[idx])

            # Align AR/BERT confidence to dLLM token index via char overlap
            def align(conf_arr, spans):
                if conf_arr is None or spans is None:
                    return None
                overlapping = [
                    float(conf_arr[si])
                    for si, (ss, se) in enumerate(spans)
                    if si < len(conf_arr) and se > cs and ss < ce
                ]
                return float(np.mean(overlapping)) if overlapping else None

            records.append(TokenRecord(
                task_id=tid,
                token_idx=tok_idx,
                char_start=cs,
                char_end=ce,
                is_fault=is_fault,
                dllm_confidence=conf_at(dllm_conf, dllm_spans, tok_idx),
                ar_confidence=align(ar_conf, ar_spans),
                bert_confidence=align(bert_conf, bert_spans),
            ))

    print(f"Total token records: {len(records)}")
    print(f"Fault tokens: {sum(r.is_fault for r in records)}")
    print(f"Non-fault tokens: {sum(not r.is_fault for r in records)}")

    out = {
        "description": "Per-token confidence + fault labels for calibration/ROC analysis",
        "n_pairs": len(eligible),
        "n_tokens": len(records),
        "n_fault": sum(r.is_fault for r in records),
        "records": [asdict(r) for r in records],
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()
```

### Step 1b：运行数据收集

```bash
# HumanEval（需要 GPU，约 30–60 分钟）
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=src conda run -n cocoder python -m coder.analysis.locator_calibration_data \
  --ar_input     outputs/base_tuteng/deepseek_humaneval.jsonl \
  --collab_input outputs/base_tuteng/deepseek_dream_remask_humaneval_t0.9_timed.jsonl \
  --ar_eval      outputs/base_tuteng/deepseek_humaneval-sanitized_eval_results.json \
  --collab_eval  outputs/base_tuteng/deepseek_dream_remask_humaneval_t0.9_timed-sanitized_eval_results.json \
  --out          outputs/ablation_locator/calibration_data_humaneval.json \
  --locators dllm ar bert \
  --device cuda:0 \
  --include_collab_fail

# MBPP
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=src conda run -n cocoder python -m coder.analysis.locator_calibration_data \
  --ar_input     outputs/base_tuteng/deepseek_mbpp.jsonl \
  --collab_input outputs/base_tuteng/deepseek_dream_remask_mbpp_t0.9_timed.jsonl \
  --ar_eval      outputs/base_tuteng/deepseek_mbpp-sanitized_eval_results.json \
  --collab_eval  outputs/base_tuteng/deepseek_dream_remask_mbpp_t0.9_timed-sanitized_eval_results.json \
  --out          outputs/ablation_locator/calibration_data_mbpp.json \
  --locators dllm ar bert \
  --device cuda:0 \
  --include_collab_fail
```

---

## Phase 2：生成 Calibration Plot 和 ROC/AUC（无需 GPU）

### Step 2a：新建 `src/coder/analysis/plot_calibration.py`

> 下面代码块是初始设计草稿。当前仓库实现以 `src/coder/analysis/plot_calibration.py` 为准，已移除 scikit-learn 依赖，增加随机分数 AUC、token-count 轴和更稳健的版面设置。

```python
#!/usr/bin/env python3
"""
plot_calibration.py — Generate calibration plots and ROC/AUC from
                      locator_calibration_data.py output.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score


def plot_calibration_and_roc(data_path: str, out_dir: str, tag: str) -> None:
    data = json.loads(Path(data_path).read_text())
    records = data["records"]

    labels = np.array([r["is_fault"] for r in records], dtype=float)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Calibration Plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    locator_keys = [
        ("dllm_confidence", "dLLM (Dream-Coder)", "steelblue"),
        ("ar_confidence",   "AR logprob",          "tomato"),
        ("bert_confidence", "CodeBERT",             "seagreen"),
    ]
    n_bins = 10

    auc_results = {}
    for ax, (key, name, color) in zip(axes, locator_keys):
        conf = np.array([r[key] if r[key] is not None else 0.5 for r in records])
        bin_edges = np.linspace(0, 1, n_bins + 1)
        fault_fracs = []
        bin_centers = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (conf >= lo) & (conf < hi)
            if mask.sum() == 0:
                continue
            fault_frac = labels[mask].mean()
            fault_fracs.append(fault_frac)
            bin_centers.append((lo + hi) / 2)

        ax.bar(bin_centers, fault_fracs, width=0.09, color=color, alpha=0.7,
               edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Confidence score", fontsize=11)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(fault_fracs) * 1.2 + 0.01)

        # AUC
        try:
            auc = roc_auc_score(labels, 1 - conf)   # low confidence = predict fault
            auc_results[name] = auc
            ax.text(0.05, max(fault_fracs) * 1.1,
                    f"AUC={auc:.3f}", fontsize=9, color="black")
        except Exception:
            pass

    axes[0].set_ylabel("Fraction of fault tokens", fontsize=11)
    fig.suptitle(f"Locator Calibration — {tag}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / f"calibration_{tag}.pdf", dpi=150)
    fig.savefig(out_dir / f"calibration_{tag}.png", dpi=150)
    plt.close(fig)
    print(f"Saved calibration plot: {out_dir}/calibration_{tag}.pdf")

    # ── ROC Curves ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    for key, name, color in locator_keys:
        conf = np.array([r[key] if r[key] is not None else 0.5 for r in records])
        try:
            fpr, tpr, _ = roc_curve(labels, 1 - conf)
            auc = roc_auc_score(labels, 1 - conf)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)
        except Exception as e:
            print(f"ROC failed for {name}: {e}")

    # Random baseline
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"ROC Curves — {tag}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / f"roc_{tag}.pdf", dpi=150)
    fig.savefig(out_dir / f"roc_{tag}.png", dpi=150)
    plt.close(fig)
    print(f"Saved ROC plot: {out_dir}/roc_{tag}.pdf")

    # Print AUC summary
    print(f"\n=== AUC Summary [{tag}] ===")
    for name, auc in auc_results.items():
        print(f"  {name}: {auc:.4f}")

    return auc_results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_humaneval", required=True)
    ap.add_argument("--data_mbpp",      required=True)
    ap.add_argument("--out_dir", default="outputs/ablation_locator/plots")
    args = ap.parse_args()

    auc_he   = plot_calibration_and_roc(args.data_humaneval, args.out_dir, "humaneval")
    auc_mbpp = plot_calibration_and_roc(args.data_mbpp,      args.out_dir, "mbpp")

    # Save AUC summary JSON
    summary = {
        "humaneval": auc_he,
        "mbpp": auc_mbpp,
    }
    import json
    (Path(args.out_dir) / "auc_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\nAUC summary saved to: {args.out_dir}/auc_summary.json")


if __name__ == "__main__":
    main()
```

### Step 2b：运行绘图脚本

```bash
PYTHONPATH=src conda run -n cocoder python -m coder.analysis.plot_calibration \
  --data_humaneval outputs/ablation_locator/calibration_data_humaneval.json \
  --data_mbpp      outputs/ablation_locator/calibration_data_mbpp.json \
  --out_dir        outputs/ablation_locator/plots
```

产物：
- `outputs/ablation_locator/plots/calibration_humaneval.pdf`
- `outputs/ablation_locator/plots/calibration_mbpp.pdf`
- `outputs/ablation_locator/plots/roc_humaneval.pdf`
- `outputs/ablation_locator/plots/roc_mbpp.pdf`
- `outputs/ablation_locator/plots/auc_summary.json`

---

## 执行结果（2026-06-09）

数据收集在 `cocoder` 环境中完成，使用 `plus_status` 过滤 AR failed 样本，并开启 `--include_collab_fail`。只统计 draft 与 remask 输出实际不同的 pair，未发生改动的 pair 记为 `n_skipped_unchanged`。

| Dataset | Eligible AR-failed pairs | Changed pairs | Skipped unchanged | Token records | Fault tokens | Non-fault tokens |
|---------|--------------------------|---------------|-------------------|---------------|--------------|------------------|
| HumanEval | 71 | 9 | 62 | 981 | 12 | 969 |
| MBPP | 132 | 23 | 109 | 1696 | 27 | 1669 |

ROC/AUC 结果：

| Dataset | dLLM (Dream-Coder) | AR logprob | CodeBERT | Random scores |
|---------|--------------------|------------|----------|---------------|
| HumanEval | 0.951 | 0.862 | 0.812 | 0.398 |
| MBPP | 0.960 | 0.824 | 0.771 | 0.562 |

产物：
- `outputs/ablation_locator/calibration_data_humaneval.json`
- `outputs/ablation_locator/calibration_data_mbpp.json`
- `outputs/ablation_locator/plots/calibration_humaneval.{pdf,png}`
- `outputs/ablation_locator/plots/calibration_mbpp.{pdf,png}`
- `outputs/ablation_locator/plots/roc_humaneval.{pdf,png}`
- `outputs/ablation_locator/plots/roc_mbpp.{pdf,png}`
- `outputs/ablation_locator/plots/auc_summary.json`

结论：dLLM locator 在两个数据集上都是 AUC 最高（HumanEval 0.951，MBPP 0.960），说明低 dLLM confidence 对 changed/fault-like token 有明显区分力。HumanEval 只有 12 个 fault tokens，低于原始 `>20` 目标；MBPP 有 27 个 fault tokens，达到样本量要求。由于 `--include_collab_fail` 后标签是 changed-token proxy，AR/CodeBERT 也高于随机，因此这组 calibration/ROC 图适合作为可视化补充；主结论仍应与 surgical fault-detection ratio 一起表述。

---

## 完整矩阵扩展（2026-06-09）

用户要求不只做 DeepSeek+Dream 单组，而是完整更换 AR drafter 和 dLLM refiner。已按 `outputs/base_tuteng/model_pairs_all_t0.9.json` 中的完整 EvalPlus model-pair 表运行：

- AR drafter：DeepSeek-Coder、Qwen2.5-Coder、Llama-3.1、StarCoder2、Mistral、CodeLlama、Seed-Coder-Instruct
- dLLM refiner/locator：Dream-Coder、LLaDA
- Dataset：HumanEval、MBPP
- 总计：7 × 2 × 2 = 28 组

执行命令：

```bash
PYTHONPATH=src conda run -n cocoder python -m coder.analysis.run_locator_calibration_matrix \
  --cuda_visible_devices 7 \
  --skip_ar_for_ars 'Llama-3.1 8B'
```

说明：Llama-3.1 的 AR logprob locator 需要访问 gated HuggingFace repo；`cocoder` 当前无可用认证/cache，因此这 4 行只计算 dLLM 和 CodeBERT，AR AUC 记为缺失。其余 24 行均计算 dLLM / AR / CodeBERT / Random。运行中为 `cocoder` 补装了 `sentencepiece`，用于 Mistral/CodeLlama tokenizer。

矩阵汇总：

| Dataset | dLLM refiner | Rows | Mean dLLM AUC | Mean AR AUC | Mean CodeBERT AUC |
|---------|--------------|------|---------------|-------------|-------------------|
| HumanEval | Dream-Coder | 7 | 0.942 | 0.774 | 0.723 |
| HumanEval | LLaDA | 7 | 0.967 | 0.712 | 0.488 |
| MBPP | Dream-Coder | 7 | 0.911 | 0.769 | 0.701 |
| MBPP | LLaDA | 7 | 0.961 | 0.676 | 0.599 |

总体结论：

- dLLM AUC 在 24/24 个 AR-available 行上高于 AR logprob。
- dLLM AUC 在 28/28 行上高于 CodeBERT。
- 17/28 行 fault tokens ≥ 20；少样本行主要集中在 HumanEval 的 Qwen/Seed 组合。
- StarCoder2 changed-token 数量最大，AUC 相对更低但 dLLM 仍显著高于 AR/BERT。

矩阵产物：

- `outputs/ablation_locator/matrix/calibration_matrix_summary.json`
- `outputs/ablation_locator/matrix/calibration_data_*.json`（28 个）
- `outputs/ablation_locator/matrix_plots/calibration_*.{pdf,png}`（28 组）
- `outputs/ablation_locator/matrix_plots/roc_*.{pdf,png}`（28 组）

---

## 验收标准

| 检查项 | 期望 | 实际 |
|--------|------|------|
| `calibration_data_humaneval.json` 中 `n_fault` | > 20（要有足够样本） | 12；作为限制说明 |
| `calibration_data_mbpp.json` 中 `n_fault` | > 20（要有足够样本） | 27 |
| Calibration plot | 低置信分桶 fault 比例更高 | ✅，已生成 PNG/PDF |
| dLLM AUC | > 0.7 | HE 0.951；MBPP 0.960 |
| AR logprob AUC | 原预期接近 0.5 | HE 0.862；MBPP 0.824，changed-token proxy 下也有信号 |
| PDF 文件可正常打开 | ✅ | ✅ |

---

## 注意事项

1. **样本量问题**：如果 eligible pairs 太少（< 20），可以放宽到包含"AR failed but CoCoder also failed"的所有 AR 失败案例，用 token diff 作为 fault 定义（即使 CoCoder 没有完全修复）。
2. **dLLM score_tokens 方法**：确认 DreamCoder 模型对象有 `score_tokens(prompt_ids, comp_ids)` 方法；如 API 不同，参考 `locator_scoring.py` 中的调用方式。
3. **图形风格**：最终入 paper 的版本可以调整配色与字体，与论文其他图保持一致。

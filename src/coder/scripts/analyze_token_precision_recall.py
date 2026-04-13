#!/usr/bin/env python3
"""
analyze_token_precision_recall.py — Token-level precision/recall of confidence-based masking.

Measures how well the dLLM's per-token confidence identifies positions in the AR
draft that differ from the canonical (reference) solution.

For each task:
  1. Tokenize the AR draft and the canonical solution using DreamCoder's tokenizer.
  2. Run DreamCoder.score_tokens() on the draft to obtain per-token confidence.
  3. Use difflib.SequenceMatcher at the token level to find positions where the
     draft diverges from the reference.  These are called "true error positions."
  4. At each threshold τ, define the predicted mask as {i : confidence[i] < τ}.
  5. Compute:
       precision(τ) = |mask ∩ errors| / |mask|
       recall(τ)    = |mask ∩ errors| / |errors|
       F1(τ)        = harmonic_mean(precision, recall)

Caveats / limitations (acknowledged in the paper):
  - A draft token that differs from the canonical solution is not necessarily
    wrong; multiple correct implementations exist.  This metric is therefore
    a proxy: high recall means the dLLM flags most canonically-divergent
    positions; high precision means flagged positions are mostly divergent.
  - Comparison is done on the cleaned completion (after clean_model_completion),
    not on the full solution string.

Usage:
  python -m coder.scripts.analyze_token_precision_recall \\
    --input  outputs/deepseek_remask_humaneval.jsonl \\
    --out    outputs/analysis/precision_recall_humaneval.jsonl \\
    --dataset humaneval \\
    --thresholds 0.5 0.7 0.9

Input JSONL (from gen_remask.py or gen_evalplus.py):
  Required fields: task_id, prompt, draft_completion   (or raw_completion)

Output JSONL (one line per task):
  task_id, n_draft_tokens, n_ref_tokens, n_errors, n_masked_{tau}, precision_{tau},
  recall_{tau}, f1_{tau}, confidence_scores (list[float]), error_mask (list[bool])

Aggregate summary is written to <out>.summary.json.
"""
from __future__ import annotations

import argparse
import difflib
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from coder.models.dream_coder import DreamCoder
from coder.utils.code_cleaning import clean_model_completion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def token_error_mask(
    draft_ids: List[int],
    ref_ids: List[int],
) -> List[bool]:
    """
    Return a boolean list of length len(draft_ids).
    Position i is True if draft_ids[i] participates in a 'replace' or 'delete'
    block when aligning draft_ids → ref_ids with SequenceMatcher.
    'insert' operations (tokens in ref but not in draft) are not counted because
    they don't correspond to a draft token position.
    """
    sm = difflib.SequenceMatcher(None, draft_ids, ref_ids, autojunk=False)
    error = [False] * len(draft_ids)
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            for i in range(i1, i2):
                error[i] = True
    return error


def compute_pr(
    confidence: List[float],
    error_mask: List[bool],
    tau: float,
) -> Tuple[float | None, float | None, float | None]:
    """
    Returns (precision, recall, f1) for a single threshold.
    Returns None for undefined cases (empty sets).
    """
    n = len(confidence)
    predicted = [confidence[i] < tau for i in range(n)]
    tp = sum(1 for i in range(n) if predicted[i] and error_mask[i])
    fp = sum(1 for i in range(n) if predicted[i] and not error_mask[i])
    fn = sum(1 for i in range(n) if not predicted[i] and error_mask[i])

    n_masked = tp + fp
    n_errors = tp + fn

    precision = tp / n_masked if n_masked > 0 else None
    recall    = tp / n_errors if n_errors > 0 else None

    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None

    return precision, recall, f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Token-level precision/recall of dLLM confidence-based masking."
    )
    ap.add_argument("--input", required=True,
                    help="Input JSONL (gen_remask or gen_evalplus output).")
    ap.add_argument("--out", required=True,
                    help="Output JSONL path (one record per task).")
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True,
                    help="EvalPlus dataset to fetch canonical solutions from.")
    ap.add_argument("--model_id", default="Dream-org/Dream-Coder-v0-Instruct-7B",
                    help="DreamCoder HuggingFace model ID (for tokenizer + scoring).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.5, 0.7, 0.9],
        help="Confidence thresholds τ at which to evaluate precision/recall."
    )
    ap.add_argument(
        "--save_scores", action="store_true",
        help="Include per-token confidence_scores and error_mask lists in output JSONL."
    )
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most N tasks (0 = all). Useful for quick tests.")
    args = ap.parse_args()

    thresholds = sorted(set(args.thresholds))

    # ------------------------------------------------------------------
    # Load EvalPlus canonical solutions
    # ------------------------------------------------------------------
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    problems: Dict[str, Dict] = (
        get_human_eval_plus() if args.dataset == "humaneval" else get_mbpp_plus()
    )
    print(f"[info] loaded {len(problems)} EvalPlus problems for '{args.dataset}'")

    # ------------------------------------------------------------------
    # Load DreamCoder (tokenizer + scoring)
    # ------------------------------------------------------------------
    print(f"[info] loading DreamCoder: {args.model_id}")
    model = DreamCoder(model_id=args.model_id, device=args.device)
    tok = model.tok

    # ------------------------------------------------------------------
    # Load input records
    # ------------------------------------------------------------------
    records = list(read_jsonl(args.input))
    if args.limit > 0:
        records = records[: args.limit]
    print(f"[info] {len(records)} records from {args.input}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Per-threshold aggregation accumulators
    # We collect (precision, recall, f1) lists, skipping None values.
    agg: Dict[float, Dict[str, List[float]]] = {
        tau: {"precision": [], "recall": [], "f1": [], "n_masked": [], "n_errors": []}
        for tau in thresholds
    }
    n_skipped = 0
    n_processed = 0
    t_total0 = time.perf_counter()

    with out_path.open("w", encoding="utf-8") as fout:
        for rec in tqdm(records, desc="analyze"):
            task_id: str = rec.get("task_id", "")
            if task_id not in problems:
                n_skipped += 1
                continue

            prompt_text: str = rec.get("prompt", "")
            # Prefer draft_completion (the original AR output before refinement),
            # fall back to raw_completion for gen_evalplus records.
            raw_draft = rec.get("draft_completion") or rec.get("raw_completion") or ""
            draft = clean_model_completion(raw_draft, prompt_text)

            # Canonical reference solution (completion body only, no prompt)
            canonical = problems[task_id].get("canonical_solution", "")
            canonical = canonical.strip()

            if not draft or not canonical:
                n_skipped += 1
                continue

            # Tokenize (no chat template; plain text sequences)
            draft_ids: List[int] = tok(
                draft, add_special_tokens=False
            ).input_ids
            ref_ids: List[int] = tok(
                canonical, add_special_tokens=False
            ).input_ids

            if len(draft_ids) == 0:
                n_skipped += 1
                continue

            # Score draft tokens with DreamCoder bidirectional attention
            messages = [{"role": "user", "content": prompt_text}]
            enc = tok.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            )
            prompt_ids = enc.input_ids.to(args.device)
            comp_ids = torch.tensor([draft_ids], device=args.device)

            confidence_t: torch.Tensor = model.score_tokens(prompt_ids, comp_ids)  # [M]
            confidence: List[float] = confidence_t.cpu().tolist()

            # Align draft tokens to reference tokens → error mask
            error_mask: List[bool] = token_error_mask(draft_ids, ref_ids)
            n_errors = sum(error_mask)

            # Compute precision/recall per threshold
            out_rec: Dict = {
                "task_id": task_id,
                "n_draft_tokens": len(draft_ids),
                "n_ref_tokens": len(ref_ids),
                "n_errors": n_errors,
            }

            for tau in thresholds:
                prec, rec_val, f1 = compute_pr(confidence, error_mask, tau)
                n_masked = sum(1 for c in confidence if c < tau)
                tau_key = f"tau_{tau:.2f}".replace(".", "_")
                out_rec[f"n_masked_{tau_key}"] = n_masked
                out_rec[f"precision_{tau_key}"] = prec
                out_rec[f"recall_{tau_key}"]    = rec_val
                out_rec[f"f1_{tau_key}"]        = f1

                # Aggregate (skip None)
                agg[tau]["n_masked"].append(n_masked)
                agg[tau]["n_errors"].append(n_errors)
                if prec is not None:
                    agg[tau]["precision"].append(prec)
                if rec_val is not None:
                    agg[tau]["recall"].append(rec_val)
                if f1 is not None:
                    agg[tau]["f1"].append(f1)

            if args.save_scores:
                out_rec["confidence_scores"] = [round(c, 6) for c in confidence]
                out_rec["error_mask"] = error_mask

            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            n_processed += 1

    t_total1 = time.perf_counter()

    # ------------------------------------------------------------------
    # Aggregate summary
    # ------------------------------------------------------------------
    def _mean(lst: List[float]) -> float | None:
        return sum(lst) / len(lst) if lst else None

    summary: Dict = {
        "script": "analyze_token_precision_recall",
        "input": str(Path(args.input).resolve()),
        "out": str(out_path.resolve()),
        "dataset": args.dataset,
        "model_id": args.model_id,
        "thresholds": thresholds,
        "n_processed": n_processed,
        "n_skipped": n_skipped,
        "total_time_s": round(t_total1 - t_total0, 2),
        "aggregate": {},
    }

    def _fmt(v):
        return f"{v:.4f}" if v is not None else "  N/A"

    print(f"\n{'τ':>6}  {'#tasks':>7}  {'precision':>10}  {'recall':>8}  {'F1':>8}")
    print("-" * 50)
    for tau in thresholds:
        a = agg[tau]
        prec_avg  = _mean(a["precision"])
        rec_avg   = _mean(a["recall"])
        f1_avg    = _mean(a["f1"])
        n_tasks   = len(a["precision"])

        tau_key = f"tau_{tau:.2f}".replace(".", "_")
        summary["aggregate"][tau_key] = {
            "tau": tau,
            "n_tasks_with_data": n_tasks,
            "avg_precision": prec_avg,
            "avg_recall": rec_avg,
            "avg_f1": f1_avg,
            "total_masked_tokens": sum(a["n_masked"]),
            "total_error_tokens": sum(a["n_errors"]),
        }

        print(f"{tau:>6.2f}  {n_tasks:>7}  {_fmt(prec_avg):>10}  {_fmt(rec_avg):>8}  {_fmt(f1_avg):>8}")

    print()

    summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote {out_path}  ({n_processed} tasks)")
    print(f"[done] wrote {summary_path}")


if __name__ == "__main__":
    main()

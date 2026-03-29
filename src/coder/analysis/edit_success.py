#!/usr/bin/env python3
"""
Analyze "edit magnitude vs success".

Inputs:
  1) samples JSONL (from gen_evalplus / gen_remask / gen_locate_ar_rewrite, etc.)
  2) optional evalplus eval_results.json (from evalplus.evaluate wrapper)

Outputs:
  - per-task metrics JSONL
  - binned summary JSON (no plotting dependency)
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_evalplus_status(eval_results_json: str) -> Dict[str, bool]:
    """
    Return task_id -> passed (base pass).
    If multiple samples exist per task, treat as passed if ANY sample passed.
    """
    with open(eval_results_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    eval_map = data.get("eval", {}) or {}
    out: Dict[str, bool] = {}
    for task_id, rows in eval_map.items():
        passed_any = False
        for r in (rows or []):
            if str(r.get("base_status", "")).lower() == "pass":
                passed_any = True
                break
        out[str(task_id)] = passed_any
    return out


def levenshtein(a: str, b: str, max_cost: Optional[int] = None) -> int:
    """
    Simple Levenshtein distance (O(len(a)*len(b))).
    max_cost: optional early-exit threshold.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Ensure a is shorter to save memory
    if len(a) > len(b):
        a, b = b, a

    prev = list(range(len(a) + 1))
    for j, cb in enumerate(b, start=1):
        cur = [j]
        min_row = cur[0]
        for i, ca in enumerate(a, start=1):
            ins = cur[i - 1] + 1
            dele = prev[i] + 1
            sub = prev[i - 1] + (0 if ca == cb else 1)
            v = min(ins, dele, sub)
            cur.append(v)
            if v < min_row:
                min_row = v
        prev = cur
        if max_cost is not None and min_row > max_cost:
            return max_cost + 1
    return prev[-1]


def safe_str(x: Any) -> str:
    return x if isinstance(x, str) else ("" if x is None else str(x))


def extract_texts(obj: Dict[str, Any]) -> Tuple[str, str]:
    """
    Return (draft, final) when present; fallback to solution/raw_completion.
    """
    draft = safe_str(obj.get("draft_completion") or obj.get("raw_solution") or obj.get("raw_completion") or obj.get("solution"))
    final = safe_str(obj.get("raw_completion") or obj.get("solution") or obj.get("raw_solution"))
    return draft, final


@dataclass
class Metrics:
    task_id: str
    model: str
    passed: Optional[bool]
    draft_len: int
    final_len: int
    char_lev: int
    char_lev_norm: float
    n_masked_tokens: Optional[int]
    n_total_tokens: Optional[int]
    rounds: Optional[int]


def bin_by(values: List[float], n_bins: int) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if lo == hi:
        return [lo, hi]
    step = (hi - lo) / n_bins
    edges = [lo + i * step for i in range(n_bins)]
    edges.append(hi)
    return edges


def assign_bin(x: float, edges: List[float]) -> int:
    if not edges or len(edges) < 2:
        return 0
    # last edge inclusive
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            if a <= x <= b:
                return i
        if a <= x < b:
            return i
    return len(edges) - 2


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute edit-magnitude metrics and join with EvalPlus pass/fail if provided.")
    ap.add_argument("--samples", required=True, help="Samples JSONL path.")
    ap.add_argument("--eval_results", default=None, help="EvalPlus eval_results.json path (optional).")
    ap.add_argument("--out_metrics", required=True, help="Output per-record metrics JSONL.")
    ap.add_argument("--out_summary", required=True, help="Output binned summary JSON.")
    ap.add_argument("--bins", type=int, default=10, help="Number of bins for normalized edit distance.")
    args = ap.parse_args()

    pass_map = load_evalplus_status(args.eval_results) if args.eval_results else {}

    metrics: List[Metrics] = []
    for obj in read_jsonl(args.samples):
        task_id = safe_str(obj.get("task_id"))
        if not task_id:
            continue
        model = safe_str(obj.get("model"))
        passed = pass_map.get(task_id) if pass_map else None

        draft, final = extract_texts(obj)
        dlen, flen = len(draft), len(final)
        lev = levenshtein(draft, final, max_cost=None)
        norm = lev / max(1, max(dlen, flen))

        gen = obj.get("gen") or {}
        n_masked = gen.get("n_masked_tokens")
        n_total = gen.get("n_total_tokens")
        rounds = gen.get("rounds")

        metrics.append(
            Metrics(
                task_id=task_id,
                model=model,
                passed=passed,
                draft_len=dlen,
                final_len=flen,
                char_lev=lev,
                char_lev_norm=float(norm),
                n_masked_tokens=int(n_masked) if isinstance(n_masked, int) else None,
                n_total_tokens=int(n_total) if isinstance(n_total, int) else None,
                rounds=int(rounds) if isinstance(rounds, int) else None,
            )
        )

    out_metrics = Path(args.out_metrics)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    with out_metrics.open("w", encoding="utf-8") as f:
        for m in metrics:
            f.write(json.dumps(m.__dict__, ensure_ascii=False) + "\n")

    norms = [m.char_lev_norm for m in metrics]
    edges = bin_by(norms, n_bins=max(1, args.bins))

    # Bin stats
    bin_rows: Dict[int, Dict[str, Any]] = defaultdict(lambda: {"n": 0, "n_pass": 0, "n_fail": 0, "n_unknown": 0})
    for m in metrics:
        b = assign_bin(m.char_lev_norm, edges)
        r = bin_rows[b]
        r["n"] += 1
        if m.passed is True:
            r["n_pass"] += 1
        elif m.passed is False:
            r["n_fail"] += 1
        else:
            r["n_unknown"] += 1

    summary_bins: List[Dict[str, Any]] = []
    for b in sorted(bin_rows.keys()):
        r = bin_rows[b]
        lo = edges[b] if edges else 0.0
        hi = edges[b + 1] if edges and b + 1 < len(edges) else lo
        acc = (r["n_pass"] / (r["n_pass"] + r["n_fail"])) if (r["n_pass"] + r["n_fail"]) > 0 else None
        summary_bins.append(
            {
                "bin": b,
                "edge_lo": lo,
                "edge_hi": hi,
                "n": r["n"],
                "n_pass": r["n_pass"],
                "n_fail": r["n_fail"],
                "n_unknown": r["n_unknown"],
                "pass_rate": acc,
            }
        )

    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "samples": str(Path(args.samples).resolve()),
        "eval_results": str(Path(args.eval_results).resolve()) if args.eval_results else None,
        "n_records": len(metrics),
        "bins": summary_bins,
    }
    out_summary.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


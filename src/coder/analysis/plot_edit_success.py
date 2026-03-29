#!/usr/bin/env python3
"""
Matplotlib plots for:
  - edit magnitude vs success
  - mask ratio vs success
  - edit position along the draft (0%..100%)
  - multi-round traces (if rounds_trace exists)

This script is intentionally lightweight: it reads samples JSONL and optional
EvalPlus eval_results.json, then writes PNGs to --out_dir.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_evalplus_pass_map(eval_results_json: str) -> Dict[str, bool]:
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


def safe_str(x: Any) -> str:
    return x if isinstance(x, str) else ("" if x is None else str(x))


def levenshtein(a: str, b: str, max_cost: Optional[int] = None) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
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


def extract_draft_final(obj: Dict[str, Any]) -> Tuple[str, str]:
    draft = safe_str(obj.get("draft_completion") or obj.get("raw_solution") or obj.get("raw_completion") or obj.get("solution"))
    final = safe_str(obj.get("raw_completion") or obj.get("solution") or obj.get("raw_solution"))
    return draft, final


def compute_mask_ratio(obj: Dict[str, Any]) -> Optional[float]:
    gen = obj.get("gen") or {}
    n_masked = gen.get("n_masked_tokens")
    n_total = gen.get("n_total_tokens")
    if isinstance(n_masked, int) and isinstance(n_total, int) and n_total > 0:
        return n_masked / n_total
    return None


def extract_mask_positions_pct(masked_draft: str, mask_token: str) -> List[float]:
    """
    Return positions of each mask token occurrence as percentage in [0, 1].
    We use the *start index* of each occurrence in the string.
    """
    s = masked_draft or ""
    if not s or not mask_token:
        return []
    L = len(s)
    if L <= 0:
        return []
    out = []
    i = 0
    while True:
        j = s.find(mask_token, i)
        if j < 0:
            break
        out.append(j / L)
        i = j + max(1, len(mask_token))
    return out


def ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    return plt


def main() -> None:
    ap = argparse.ArgumentParser(description="Matplotlib plots for edit magnitude & edit position.")
    ap.add_argument("--samples", required=True, help="Samples JSONL path (e.g., locate_ar_rewrite outputs).")
    ap.add_argument("--eval_results", default=None, help="EvalPlus eval_results.json path (optional).")
    ap.add_argument("--out_dir", required=True, help="Directory to write PNGs into.")
    ap.add_argument("--mask_token", default="<MASK>", help="Mask token used in masked_draft.")
    ap.add_argument("--bins", type=int, default=20, help="Histogram bins.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pass_map = load_evalplus_pass_map(args.eval_results) if args.eval_results else {}

    xs_edit = []
    ys_pass = []
    xs_mask_ratio = []

    pos_all: List[float] = []
    pos_pass: List[float] = []
    pos_fail: List[float] = []

    per_round_positions: Dict[int, List[float]] = defaultdict(list)
    per_round_mask_ratio: Dict[int, List[float]] = defaultdict(list)

    for obj in read_jsonl(args.samples):
        task_id = safe_str(obj.get("task_id"))
        if not task_id:
            continue
        passed = pass_map.get(task_id) if pass_map else None
        draft, final = extract_draft_final(obj)
        dlen, flen = len(draft), len(final)
        lev = levenshtein(draft, final)
        lev_norm = lev / max(1, max(dlen, flen))

        xs_edit.append(lev_norm)
        ys_pass.append(1 if passed is True else (0 if passed is False else None))

        mr = compute_mask_ratio(obj)
        if mr is not None:
            xs_mask_ratio.append(mr)

        # Edit positions from masked_draft (single-round) or rounds_trace (multi-round)
        if isinstance(obj.get("rounds_trace"), list) and obj["rounds_trace"]:
            for rr in obj["rounds_trace"]:
                if not isinstance(rr, dict):
                    continue
                r = rr.get("round")
                md = safe_str(rr.get("masked_draft"))
                ps = extract_mask_positions_pct(md, mask_token=args.mask_token)
                if isinstance(r, int):
                    per_round_positions[r].extend(ps)
                    nm = rr.get("n_masked_tokens")
                    nt = rr.get("n_total_tokens")
                    if isinstance(nm, int) and isinstance(nt, int) and nt > 0:
                        per_round_mask_ratio[r].append(nm / nt)
        else:
            md = safe_str(obj.get("masked_draft"))
            ps = extract_mask_positions_pct(md, mask_token=args.mask_token)
            pos_all.extend(ps)
            if passed is True:
                pos_pass.extend(ps)
            elif passed is False:
                pos_fail.extend(ps)

    plt = ensure_matplotlib()

    # -------- Plot 1: pass rate vs edit magnitude (binned) --------
    # Only keep labeled points
    labeled = [(x, y) for x, y in zip(xs_edit, ys_pass) if y is not None]
    if labeled:
        xs, ys = zip(*labeled)
        # Bin by x into bins, compute mean(y)
        import numpy as np

        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)
        bins = max(5, int(args.bins))
        edges = np.linspace(xs.min(), xs.max(), bins + 1)
        bin_id = np.clip(np.digitize(xs, edges) - 1, 0, bins - 1)
        bin_sum = np.zeros(bins)
        bin_cnt = np.zeros(bins)
        for i in range(len(xs)):
            b = bin_id[i]
            bin_sum[b] += ys[i]
            bin_cnt[b] += 1
        bin_mean = np.divide(bin_sum, np.maximum(bin_cnt, 1), dtype=float)
        centers = (edges[:-1] + edges[1:]) / 2

        plt.figure(figsize=(7, 4))
        plt.plot(centers, bin_mean, marker="o", linewidth=2)
        plt.xlabel("Normalized edit distance (char Levenshtein / max_len)")
        plt.ylabel("Pass rate (base)")
        plt.title("Pass rate vs edit magnitude (binned)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "pass_vs_edit_magnitude.png", dpi=180)
        plt.close()

    # -------- Plot 2: histogram of edit positions (0..100%) --------
    # If we have per-round traces, prefer showing per-round; else overall.
    if per_round_positions:
        max_r = max(per_round_positions.keys())
        nrows = max_r + 1
        plt.figure(figsize=(7, 2.2 * nrows))
        for r in range(nrows):
            ax = plt.subplot(nrows, 1, r + 1)
            ps = per_round_positions.get(r, [])
            ax.hist(ps, bins=args.bins, range=(0.0, 1.0), color="#2c7fb8", alpha=0.85)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylabel(f"r={r}")
            if r == 0:
                ax.set_title("Edit locations (mask token positions) by round")
            if r != nrows - 1:
                ax.set_xticklabels([])
        plt.xlabel("Position in draft (0=begin, 1=end)")
        plt.tight_layout()
        plt.savefig(out_dir / "edit_positions_by_round.png", dpi=180)
        plt.close()
    elif pos_all:
        plt.figure(figsize=(7, 4))
        plt.hist(pos_all, bins=args.bins, range=(0.0, 1.0), color="#2c7fb8", alpha=0.85, label="all")
        if pos_pass:
            plt.hist(pos_pass, bins=args.bins, range=(0.0, 1.0), color="#31a354", alpha=0.5, label="pass")
        if pos_fail:
            plt.hist(pos_fail, bins=args.bins, range=(0.0, 1.0), color="#de2d26", alpha=0.5, label="fail")
        plt.xlabel("Position in masked_draft (0=begin, 1=end)")
        plt.ylabel("Count")
        plt.title("Where edits happen (mask token positions)")
        plt.legend()
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / "edit_positions_hist.png", dpi=180)
        plt.close()

    # -------- Plot 3: mask ratio distribution --------
    if xs_mask_ratio:
        plt.figure(figsize=(7, 4))
        plt.hist(xs_mask_ratio, bins=args.bins, range=(0.0, 1.0), color="#756bb1", alpha=0.85)
        plt.xlabel("Mask ratio (n_masked_tokens / n_total_tokens)")
        plt.ylabel("Count")
        plt.title("Mask ratio distribution")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / "mask_ratio_hist.png", dpi=180)
        plt.close()

    # -------- Plot 4: per-round mask ratio trend (if available) --------
    if per_round_mask_ratio:
        rounds = sorted(per_round_mask_ratio.keys())
        means = []
        for r in rounds:
            vals = per_round_mask_ratio[r]
            means.append(sum(vals) / len(vals) if vals else 0.0)
        plt.figure(figsize=(7, 4))
        plt.plot(rounds, means, marker="o", linewidth=2)
        plt.xlabel("Round")
        plt.ylabel("Mean mask ratio")
        plt.title("Mean mask ratio by round")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "mask_ratio_by_round.png", dpi=180)
        plt.close()

    # Write a tiny manifest
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "samples": str(Path(args.samples).resolve()),
                "eval_results": str(Path(args.eval_results).resolve()) if args.eval_results else None,
                "out_dir": str(out_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()


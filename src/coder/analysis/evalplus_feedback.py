#!/usr/bin/env python3
"""
Extract per-task EvalPlus failure feedback for Reflexion.

Input:
  - eval_results.json produced by evalplus.evaluate

Output:
  - feedback JSONL with one record per task_id, e.g.:
      {
        "task_id": "HumanEval/0",
        "passed_base": false,
        "base_status_counts": {"pass": 0, "fail": 3, ...},  # optional
        "failure_summary": "base_status=fail",
        "raw": {...}  # optional raw row snapshot
      }

Usage:
  python -m coder.analysis.evalplus_feedback \\
    --eval_results outputs/deepseek_humaneval-sanitized_eval_results.json \\
    --out_feedback outputs/deepseek_humaneval.evalplus_feedback.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_eval_results(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _first_non_empty_str(*vals: Any) -> str:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _extract_detail_message(row: Dict[str, Any]) -> str:
    # Handle common field name variants across evalplus versions.
    direct = _first_non_empty_str(
        row.get("error"),
        row.get("message"),
        row.get("details"),
        row.get("detail"),
        row.get("base_error"),
        row.get("base_message"),
    )
    if direct:
        return direct

    # Nested dict style: {"base_details": {"error": "..."}}
    for k in ("base_details", "details", "result", "failure"):
        v = row.get(k)
        if isinstance(v, dict):
            nested = _first_non_empty_str(
                v.get("error"),
                v.get("message"),
                v.get("details"),
                v.get("detail"),
                v.get("exception"),
                v.get("traceback"),
            )
            if nested:
                return nested
    return ""


def _truncate(text: str, max_chars: int) -> str:
    s = (text or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 3)] + "..."


def _build_failure_summary(rows: List[Dict[str, Any]], max_chars: int) -> str:
    if not rows:
        return "no_eval_rows"

    base_counts = Counter(str(r.get("base_status", "unknown")) for r in rows)
    top_status, top_n = base_counts.most_common(1)[0]

    detail = ""
    for r in rows:
        detail = _extract_detail_message(r)
        if detail:
            break

    prefix = f"base_status={top_status} ({top_n}/{len(rows)})"
    if detail:
        return _truncate(f"{prefix}; detail={detail}", max_chars)
    return prefix


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract EvalPlus per-task failure feedback JSONL.")
    ap.add_argument("--eval_results", required=True, help="Path to *_eval_results.json produced by eval_evalplus.py / evalplus.evaluate.")
    ap.add_argument("--out_feedback", required=True, help="Output feedback JSONL path.")
    ap.add_argument(
        "--include_raw",
        action="store_true",
        help="Include raw first row from eval map for each task (can be large).",
    )
    ap.add_argument(
        "--max_summary_chars",
        type=int,
        default=320,
        help="Max characters for failure_summary.",
    )
    args = ap.parse_args()

    data = load_eval_results(args.eval_results)
    eval_map = data.get("eval", {}) or {}

    out_path = Path(args.out_feedback)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_tasks = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for task_id, rows in eval_map.items():
            rows = rows or []
            n_tasks += 1

            # Heuristic: treat any base_status == "pass" as success.
            passed_any = any(str(r.get("base_status", "")).lower() == "pass" for r in rows)
            row0 = rows[0] if rows else {}
            base_status = str(row0.get("base_status", "unknown"))
            base_counts = Counter(str(r.get("base_status", "unknown")) for r in rows)
            failure_summary = _build_failure_summary(rows, max_chars=max(64, args.max_summary_chars))

            rec: Dict[str, Any] = {
                "task_id": task_id,
                "passed_base": bool(passed_any),
                "base_status": base_status,
                "base_status_counts": dict(base_counts),
                "failure_summary": failure_summary,
            }
            if args.include_raw and row0:
                rec["raw_details"] = row0

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[feedback] wrote {out_path} for {n_tasks} tasks")


if __name__ == "__main__":
    main()


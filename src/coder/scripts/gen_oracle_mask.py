#!/usr/bin/env python3
"""
Build oracle mask spans for the Oracle Locator experiment.

For each task where the AR draft failed but the collaborative rewrite passed,
compute character spans in the AR draft that differ from the rewritten output.
The output mirrors the AR JSONL and adds:
  oracle_mask_spans: list of [start, end] draft character spans, or None
  oracle_diff_chars: total number of differing draft characters
"""
from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def char_diff_spans(a: str, b: str) -> list[tuple[int, int]]:
    """Return character spans in a that are replaced or deleted relative to b."""
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    spans: list[tuple[int, int]] = []
    for op, a0, a1, _b0, _b1 in matcher.get_opcodes():
        if op != "equal" and a0 < a1:
            spans.append((a0, a1))
    return spans


def _status_is_pass(info: Any) -> bool:
    if isinstance(info, list):
        return any(_status_is_pass(item) for item in info)
    if not isinstance(info, dict):
        return False
    if "eval" in info and isinstance(info["eval"], dict):
        return _status_is_pass(info["eval"])
    base_status = info.get("base_status")
    if base_status is not None:
        return base_status == "pass"
    plus_status = info.get("plus_status")
    if plus_status is not None:
        return plus_status == "pass"
    return False


def load_eval_pass_map(path: str) -> dict[str, bool]:
    """Load EvalPlus result JSON in either direct or top-level-'eval' format."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    task_results = data.get("eval", data) if isinstance(data, dict) else {}
    if not isinstance(task_results, dict):
        raise ValueError(f"Unsupported eval result format: {path}")
    return {task_id: _status_is_pass(info) for task_id, info in task_results.items()}


def completion_text(rec: dict[str, Any]) -> str:
    return (
        rec.get("raw_completion")
        or rec.get("solution")
        or rec.get("draft_completion")
        or rec.get("raw_solution")
        or ""
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build oracle mask JSONL from AR drafts and successful collaborative rewrites."
    )
    ap.add_argument("--ar_input", required=True, help="AR draft JSONL")
    ap.add_argument("--collab_input", required=True, help="CoCoder result JSONL")
    ap.add_argument("--ar_eval", required=True, help="AR eval_results.json")
    ap.add_argument("--collab_eval", required=True, help="CoCoder eval_results.json")
    ap.add_argument("--out", required=True, help="Output oracle-masked JSONL")
    ap.add_argument(
        "--min_diff_chars",
        type=int,
        default=1,
        help="Skip eligible pairs where the draft-side diff is smaller than N characters.",
    )
    ap.add_argument(
        "--max_diff_chars",
        type=int,
        default=500,
        help="Skip eligible pairs where the draft-side diff is larger than N characters.",
    )
    args = ap.parse_args()

    ar_pass = load_eval_pass_map(args.ar_eval)
    collab_pass = load_eval_pass_map(args.collab_eval)

    ar_records = read_jsonl(args.ar_input)
    collab_by_id = {rec["task_id"]: rec for rec in read_jsonl(args.collab_input)}

    eligible = {
        rec["task_id"]
        for rec in ar_records
        if "task_id" in rec
        and rec["task_id"] in collab_by_id
        and not ar_pass.get(rec["task_id"], True)
        and collab_pass.get(rec["task_id"], False)
    }
    print(f"Eligible tasks (AR fail -> Collab pass): {len(eligible)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_too_small = 0
    skipped_too_large = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for ar_rec in ar_records:
            rec = dict(ar_rec)
            task_id = rec.get("task_id")
            rec["oracle_mask_spans"] = None
            rec["oracle_diff_chars"] = 0

            if task_id in eligible:
                ar_completion = completion_text(rec)
                collab_completion = completion_text(collab_by_id[task_id])
                spans = char_diff_spans(ar_completion, collab_completion)
                diff_chars = sum(end - start for start, end in spans)
                rec["oracle_diff_chars"] = diff_chars
                if diff_chars < args.min_diff_chars:
                    skipped_too_small += 1
                elif diff_chars > args.max_diff_chars:
                    skipped_too_large += 1
                else:
                    rec["oracle_mask_spans"] = [[start, end] for start, end in spans]
                    written += 1

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Written: {written} tasks with oracle mask spans")
    print(f"Skipped too small: {skipped_too_small}")
    print(f"Skipped too large: {skipped_too_large}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()

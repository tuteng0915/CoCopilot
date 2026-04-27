#!/usr/bin/env python3
"""Normalize EvalPlus JSONL samples to one solution-packaging policy.

This is an offline analysis utility. It rebuilds `solution` from the recorded
`prompt` and `raw_completion` for every record, so EvalPlus comparisons are not
confounded by older packaging differences.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from coder.utils.code_cleaning import build_evalplus_solution


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def normalize_record(rec: dict[str, Any], tag: str) -> dict[str, Any]:
    out = dict(rec)
    prompt = str(out.get("prompt") or "")
    raw = out.get("raw_completion")
    if not isinstance(raw, str):
        raw = str(out.get("solution") or "")
        out["raw_completion"] = raw

    old_solution = out.get("solution")
    new_solution = build_evalplus_solution(prompt, raw)
    out["solution"] = new_solution

    gen = dict(out.get("gen") or {})
    gen["packaging_normalized"] = True
    gen["packaging_normalizer"] = tag
    gen["packaging_solution_changed"] = bool(old_solution != new_solution)
    gen["packaging_old_solution_chars"] = len(old_solution) if isinstance(old_solution, str) else None
    gen["packaging_new_solution_chars"] = len(new_solution)
    out["gen"] = gen
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--tag",
        default="evalplus_packaging_v2_ast_imports",
        help="Metadata tag recorded in gen.packaging_normalizer.",
    )
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    changed = 0
    with args.out.open("w", encoding="utf-8") as f:
        for rec in read_jsonl(args.input):
            out = normalize_record(rec, args.tag)
            changed += int(bool((out.get("gen") or {}).get("packaging_solution_changed")))
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} records to {args.out}")
    print(f"Changed solution packaging for {changed}/{n} records")


if __name__ == "__main__":
    main()

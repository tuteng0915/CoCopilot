"""
postprocess_bigcodebench.py

Reads a raw BigCodeBench generation file (where `solution` may contain
markdown-wrapped output) and writes a cleaned version where `solution`
contains only the extracted Python code.

Usage:
    python -m coder.scripts.postprocess_bigcodebench \
        --samples outputs/base_tuteng/qwen_bigcodebench_instruct_full.jsonl \
        --out     outputs/base_tuteng/qwen_bigcodebench_instruct_full_pass1_clean.jsonl

The script applies `build_prompt_scaffold_solution(prompt, raw_solution)` to
each record, using the `prompt` and `raw_solution` fields stored by
gen_bigcodebench.py.  The resulting file is suitable for `eval_bigcodebench`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from coder.utils.code_cleaning import build_prompt_scaffold_solution


def postprocess(samples_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_already_clean = 0

    with samples_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            prompt = rec.get("prompt", "")
            # prefer raw_solution (original model output) over solution
            raw = rec.get("raw_solution") or rec.get("solution", "")

            cleaned = build_prompt_scaffold_solution(prompt, raw)

            # track whether anything changed
            if cleaned == rec.get("solution", ""):
                n_already_clean += 1

            out_rec = dict(rec)
            out_rec["raw_completion"] = cleaned
            out_rec["solution"] = cleaned
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[postprocess_bigcodebench] wrote {n_written} records → {out_path}")
    print(f"  already-clean: {n_already_clean}, modified: {n_written - n_already_clean}")


def main():
    ap = argparse.ArgumentParser(
        description="Clean raw BigCodeBench generation output (extract code from markdown)."
    )
    ap.add_argument("--samples", required=True,
                    help="Raw generation .jsonl file (e.g. qwen_bigcodebench_instruct_full.jsonl)")
    ap.add_argument("--out", default=None,
                    help="Output path. Defaults to <samples-stem>_pass1_clean.jsonl")
    args = ap.parse_args()

    samples_path = Path(args.samples)
    if not samples_path.exists():
        raise FileNotFoundError(f"samples not found: {samples_path}")

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = samples_path.with_name(
            samples_path.stem + "_pass1_clean" + samples_path.suffix
        )

    postprocess(samples_path, out_path)


if __name__ == "__main__":
    main()

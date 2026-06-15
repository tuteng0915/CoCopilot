"""gen_math_self_rewrite.py — Self-rewrite baseline for math-to-code benchmarks.

The AR model reviews its own draft `solution()` and generates a corrected version.
This is a pure AR-only refinement baseline: no dLLM, no confidence signal.
Compared against CoCoder (AR draft + dLLM remask), it isolates whether gains
come from the dLLM locator or merely from running a second generation pass.

Usage:
  python -m coder.scripts.gen_math_self_rewrite \\
    --input  outputs/math_code/deepseek_gsm8k_code.jsonl \\
    --model  deepseek \\
    --dataset gsm8k \\
    --out    outputs/math_code/deepseek_gsm8k_code_self_rewrite.jsonl

Evaluate with:
  python -m coder.scripts.eval_math_code \\
    --input outputs/math_code/deepseek_gsm8k_code_self_rewrite.jsonl \\
    --out   outputs/math_code/deepseek_gsm8k_code_self_rewrite_eval.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable

from tqdm import tqdm

from coder.scripts.gen_math import build_model
from coder.scripts.eval_math_code import ensure_solution_function
from coder.utils.schema import ModelRequest


import re as _re
_CODE_RE = _re.compile(r'def\s+solution\s*\(|```')

def extract_draft_code(raw_completion: str) -> str:
    """Return a clean `def solution():...` string from any raw_completion format."""
    return ensure_solution_function(raw_completion)


def has_code(text: str) -> bool:
    """Return True if text contains a code block or def solution()."""
    return bool(_CODE_RE.search(text))


# ── Prompt templates ──────────────────────────────────────────────────────────

SELF_REWRITE_PROMPT = """\
You previously wrote a Python function `solution()` to solve a math problem, \
but it may contain bugs.
Review your solution carefully and provide a corrected version.
Return ONLY the function definition. Do NOT add print statements or input() calls.

Problem:
{question}

Your previous solution (may contain bugs):
```python
{draft}
```

Corrected solution:
def solution():
"""

AIME_SELF_REWRITE_PROMPT = """\
You previously wrote a Python function `solution()` to solve an AIME competition math problem, \
but it may contain bugs.
The answer must be a non-negative integer between 000 and 999.
Review your solution carefully and provide a corrected version.
Return ONLY the function definition. Do NOT add print statements or input() calls.

Problem:
{question}

Your previous solution (may contain bugs):
```python
{draft}
```

Corrected solution:
def solution():
"""


def _get_prompt(dataset: str, question: str, draft: str) -> str:
    template = AIME_SELF_REWRITE_PROMPT if dataset in ("aime", "aime2025") else SELF_REWRITE_PROMPT
    return template.format(question=question, draft=draft.strip())


# ── I/O helpers ───────────────────────────────────────────────────────────────

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for rec in read_jsonl(str(path)):
        rid = rec.get("id")
        if rid:
            done.add(str(rid))
    return done


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Self-rewrite baseline: AR model reviews and fixes its own math-code draft."
    )
    ap.add_argument("--input",   required=True, help="AR draft JSONL from gen_math_code")
    ap.add_argument("--model",   required=True,
                    choices=["deepseek", "qwen", "llama31", "mistral", "starcoder2",
                             "seed-coder", "api"],
                    help="AR model to use for rewriting (should match the draft model)")
    ap.add_argument("--dataset", required=True,
                    choices=["gsm8k", "math500", "aime", "aime2025"],
                    help="Dataset name (determines prompt template)")
    ap.add_argument("--out",     required=True, help="Output JSONL path")

    ap.add_argument("--model_id",      type=str, default=None,
                    help="Override HuggingFace model id")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature",   type=float, default=0.0)
    ap.add_argument("--top_p",         type=float, default=1.0)
    ap.add_argument("--seed",          type=int, default=3407)
    ap.add_argument("--device",        type=str, default="cuda")
    ap.add_argument("--resume",        action="store_true",
                    help="Skip records already written to --out")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if args.resume and out_path.exists():
        done_ids = load_done_ids(out_path)
        print(f"[resume] {len(done_ids)} records already done, skipping.")

    records = list(read_jsonl(args.input))
    print(f"[info] {len(records)} AR drafts loaded from {args.input}")

    model = build_model(args.model, device=args.device, model_id=args.model_id)

    timing_s: list[float] = []
    n_written = 0
    t_total0 = time.perf_counter()

    with out_path.open("a", encoding="utf-8") as fout:
        for rec in tqdm(records, desc=f"self_rewrite({args.model}/{args.dataset})"):
            rid = str(rec.get("id", ""))
            if rid in done_ids:
                continue

            question: str = rec.get("question", "")
            draft: str    = extract_draft_code(rec.get("raw_completion", ""))

            refine_prompt = _get_prompt(args.dataset, question, draft)

            req = ModelRequest(
                prompt=refine_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
            )

            t0 = time.perf_counter()
            try:
                refined = model.generate(req)
            except Exception as exc:
                print(f"[warn] {rid}: generation failed ({exc}); keeping draft.")
                refined = draft
            t1 = time.perf_counter()
            timing_s.append(t1 - t0)

            # If model produced no code (e.g. "The solution is correct."), fall
            # back to the original draft — the review conclusion was "no change".
            used_draft_fallback = False
            if not has_code(refined):
                refined = draft
                used_draft_fallback = True

            out_rec: Dict[str, Any] = {
                "id":             rid,
                "question":       question,
                "answer_ref":     rec.get("answer_ref", ""),
                "raw_completion": refined,
                "draft_completion": draft,
                "fallback_to_draft": used_draft_fallback,
                "code_mode":      True,
                "model":          f"self_rewrite::{model.name}",
                "dataset":        args.dataset,
                "gen": {
                    "source_model":    rec.get("model", ""),
                    "rewrite_model":   model.name,
                    "max_new_tokens":  args.max_new_tokens,
                    "temperature":     args.temperature,
                    "top_p":           args.top_p,
                    "seed":            args.seed,
                    "timing": {"generate_s": t1 - t0},
                },
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_written += 1

    t_total1 = time.perf_counter()
    mean_t = sum(timing_s) / len(timing_s) if timing_s else 0.0

    summary = {
        "script":           "gen_math_self_rewrite",
        "model":            args.model,
        "dataset":          args.dataset,
        "input":            args.input,
        "out":              str(out_path.resolve()),
        "n_input":          len(records),
        "n_written":        n_written,
        "n_skipped_resume": len(done_ids),
        "mean_generate_s":  mean_t,
        "total_s":          t_total1 - t_total0,
    }
    timing_path = str(out_path) + ".timing_summary.json"
    Path(timing_path).write_text(json.dumps(summary, indent=2))
    print(f"[done] {n_written} records → {out_path}  ({mean_t:.1f}s/sample)")


if __name__ == "__main__":
    main()

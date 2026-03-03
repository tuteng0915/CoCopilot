#!/usr/bin/env python3
"""
gen_remask.py — Refine DeepSeek-Coder drafts using DreamCoder token remasking.

Pipeline:
  1. Load DeepSeek-Coder JSONL outputs (prompt + raw_completion per task).
  2. For each task, score each completion token with DreamCoder's forward pass.
  3. Mask low-confidence tokens (by threshold or by ratio).
  4. Regenerate only the masked positions via DreamCoder diffusion.
  5. Write refined JSONL (EvalPlus-compatible).

Input JSONL fields:  task_id, prompt, raw_completion, solution, model, gen
Output JSONL fields: task_id, prompt, draft_completion, raw_completion, solution, model, gen
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm

from coder.models.dream_coder import DreamCoder
from coder.utils.schema import ModelRequest


def build_evalplus_solution(prompt_text: str, gen: str) -> str:
    """Build a full EvalPlus-compatible solution string from a prompt + completion."""
    g = (gen or "").lstrip()
    if re.search(r"(?m)^(def|class|import|from)\s+", g):
        return g.rstrip()
    return (prompt_text.rstrip() + "\n" + gen.lstrip()).rstrip()


def read_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Refine DeepSeek drafts with DreamCoder token remasking."
    )
    ap.add_argument("--input", required=True,
                    help="Input JSONL file from DeepSeek-Coder generation.")
    ap.add_argument("--out", required=True,
                    help="Output JSONL path for refined samples.")
    ap.add_argument("--model_id", default="Dream-org/Dream-Coder-v0-Instruct-7B",
                    help="DreamCoder HuggingFace model ID.")
    ap.add_argument("--device", default="cuda")

    # Masking policy (use exactly one)
    ap.add_argument("--confidence_threshold", type=float, default=None,
                    help="Mask tokens where P(token|context) < threshold. Default 0.5.")
    ap.add_argument("--mask_ratio", type=float, default=None,
                    help="Mask the bottom K%% least-confident tokens (0.0–1.0).")

    # Diffusion generation params for the refinement step
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--max_new_tokens", type=int, default=512,
                    help="Fallback budget; actual max_new_tokens is set to draft length.")

    ap.add_argument("--resume", action="store_true",
                    help="Skip task_ids already present in --out (append mode).")
    args = ap.parse_args()

    # Resolve masking policy
    if args.confidence_threshold is None and args.mask_ratio is None:
        args.confidence_threshold = 0.5
    if args.confidence_threshold is not None and args.mask_ratio is not None:
        ap.error("Specify at most one of --confidence_threshold and --mask_ratio.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: collect already-done task_ids
    done_ids: set[str] = set()
    if args.resume and out_path.exists():
        for rec in read_jsonl(str(out_path)):
            done_ids.add(rec["task_id"])
        print(f"[resume] skipping {len(done_ids)} already-done tasks.")

    # Load DreamCoder once
    model = DreamCoder(model_id=args.model_id, device=args.device)

    records = list(read_jsonl(args.input))
    print(f"[info] {len(records)} records loaded from {args.input}")

    with out_path.open("a", encoding="utf-8") as fout:
        for rec in tqdm(records, desc="remask"):
            task_id: str = rec["task_id"]
            if task_id in done_ids:
                continue

            prompt_text: str = rec["prompt"]
            draft: str = rec.get("raw_completion", "")

            req = ModelRequest(
                prompt=prompt_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
            )

            try:
                refined = model.generate_with_remask(
                    req=req,
                    draft=draft,
                    confidence_threshold=args.confidence_threshold,
                    mask_ratio=args.mask_ratio,
                )
            except Exception as e:
                print(f"[warn] {task_id}: generate_with_remask failed ({e}); keeping draft.")
                refined = draft

            solution = build_evalplus_solution(prompt_text, refined)

            out_rec = {
                "task_id":           task_id,
                "prompt":            prompt_text,
                "draft_completion":  draft,       # original DeepSeek output
                "raw_completion":    refined,     # DreamCoder-refined output
                "solution":          solution,    # EvalPlus-compatible full solution
                "model":             f"dream_remask::{args.model_id}",
                "gen": {
                    "source_model":         rec.get("model", "unknown"),
                    "confidence_threshold": args.confidence_threshold,
                    "mask_ratio":           args.mask_ratio,
                    "temperature":          args.temperature,
                    "top_p":                args.top_p,
                    "seed":                 args.seed,
                },
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()

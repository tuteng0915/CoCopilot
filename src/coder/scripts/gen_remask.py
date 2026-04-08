#!/usr/bin/env python3
"""
gen_remask.py — Refine DeepSeek-Coder (or other AR) drafts using DreamCoder token remasking.

Pipeline:
  1. Load AR JSONL outputs (prompt + raw_completion per task).
     - EvalPlus: from scripts/gen_evalplus.py
       fields: task_id, prompt, raw_completion, solution, model, gen
     - LiveBench-Coding: from scripts/gen_livebench.py
       fields: task_id, question_id, prompt, raw_completion, solution, model, gen, meta
  2. For each task, score each completion token with DreamCoder's forward pass.
  3. Mask low-confidence tokens (by threshold or by ratio).
  4. Regenerate only the masked positions via DreamCoder diffusion.
  5. Write refined JSONL (EvalPlus/LiveBench-compatible).

Output JSONL (EvalPlus):
  task_id, prompt, draft_completion, raw_completion, solution, model, gen

Output JSONL (LiveBench):
  task_id, question_id, prompt, draft_completion, raw_completion, solution, model, gen, meta
"""
from __future__ import annotations

import argparse
import fcntl
import json
import re
import time
from pathlib import Path

from tqdm import tqdm

from coder.models import DreamCoder, LLaDACoder
from coder.utils.code_cleaning import build_prompt_scaffold_solution, clean_model_completion
from coder.utils.schema import ModelRequest
from coder.utils.sharding import take_shard, validate_shard_args


def build_evalplus_solution(prompt_text: str, gen: str) -> str:
    """Build a full EvalPlus-compatible solution string from a prompt + completion."""
    g = clean_model_completion(gen, prompt_text).lstrip()
    if re.search(r"(?m)^(def|class|import|from)\s+", g):
        return g.rstrip()
    return (prompt_text.rstrip() + "\n" + g.lstrip()).rstrip()


def read_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def infer_refiner_name(model_id: str) -> str:
    mid = (model_id or "").lower()
    if "llada" in mid:
        return "llada"
    return "dream"


def build_refiner(name: str, model_id: str, device: str):
    if name == "dream":
        return DreamCoder(model_id=model_id, device=device)
    if name == "llada":
        return LLaDACoder(model_id=model_id, device=device)
    raise ValueError(f"Unsupported refiner: {name}")


def infer_benchmark(rec: dict, task_id: str) -> str | None:
    benchmark = rec.get("benchmark")
    if isinstance(benchmark, str) and benchmark.strip():
        return benchmark.strip()
    if task_id.startswith("LiveBench/"):
        return "livebench-coding"
    if task_id.startswith("LiveCodeBench/"):
        return "livecodebench"
    if task_id.startswith("BigCodeBench/"):
        return "bigcodebench"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Refine AR drafts with a diffusion refiner via confidence-based remasking."
    )
    ap.add_argument("--input", required=True,
                    help="Input JSONL file from AR model generation.")
    ap.add_argument("--out", required=True,
                    help="Output JSONL path for refined samples.")
    ap.add_argument("--refiner", choices=["dream", "llada"], default=None,
                    help="Refiner family. Default: infer from --model_id.")
    ap.add_argument("--model_id", default="Dream-org/Dream-Coder-v0-Instruct-7B",
                    help="Refiner HuggingFace model ID.")
    ap.add_argument("--device", default="cuda")

    # Masking policy (use exactly one)
    ap.add_argument("--confidence_threshold", type=float, default=None,
                    help="Mask tokens where P(token|context) < threshold. Default 0.5.")
    ap.add_argument("--mask_ratio", type=float, default=None,
                    help="Mask the bottom K%% least-confident tokens (0.0–1.0).")
    ap.add_argument(
        "--mask_granularity",
        choices=["token", "span", "line"],
        default="token",
        help="Masking granularity: token (default), span (merge nearby masks), line (mask whole lines).",
    )
    ap.add_argument(
        "--span_merge_gap",
        type=int,
        default=0,
        help="For --mask_granularity span: also mask gaps of <= this many tokens between masked tokens.",
    )

    # Diffusion generation params for the refinement step
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--max_new_tokens", type=int, default=512,
                    help="Fallback budget; actual max_new_tokens is set to draft length.")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_idx", type=int, default=0)

    ap.add_argument("--resume", action="store_true",
                    help="Skip task_ids already present in --out (append mode).")
    args = ap.parse_args()

    # Resolve masking policy
    if args.confidence_threshold is None and args.mask_ratio is None:
        args.confidence_threshold = 0.5
    if args.confidence_threshold is not None and args.mask_ratio is not None:
        ap.error("Specify at most one of --confidence_threshold and --mask_ratio.")
    try:
        validate_shard_args(num_shards=args.num_shards, shard_idx=args.shard_idx)
    except ValueError as e:
        ap.error(str(e))

    refiner_name = args.refiner or infer_refiner_name(args.model_id)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = out_path.with_suffix(out_path.suffix + ".lock")
    lock_f = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as e:
        raise RuntimeError(f"Another gen_remask process is already writing to {out_path}") from e

    # Resume: collect already-done task_ids
    done_ids: set[str] = set()
    if args.resume and out_path.exists():
        for rec in read_jsonl(str(out_path)):
            done_ids.add(rec["task_id"])
        print(f"[resume] skipping {len(done_ids)} already-done tasks.")

    records = list(read_jsonl(args.input))
    records = take_shard(records, num_shards=args.num_shards, shard_idx=args.shard_idx)
    print(f"[info] {len(records)} records loaded from {args.input}")
    if not records:
        out_path.touch(exist_ok=True)
        summary = {
            "script": "gen_remask",
            "out": str(out_path.resolve()),
            "model": None,
            "refiner": refiner_name,
            "num_shards": args.num_shards,
            "shard_idx": args.shard_idx,
            "n_records_written": 0,
            "timing": {
                "total_s": 0.0,
                "remask_generate_s_total": 0.0,
                "pack_solution_s_total": 0.0,
                "remask_generate_s_avg": None,
            },
        }
        out_path.with_suffix(out_path.suffix + ".timing_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[timing] wrote {out_path}.timing_summary.json")
        print(f"[done] no records selected for shard {args.shard_idx}/{args.num_shards}")
        return

    # Load refiner once
    model = build_refiner(refiner_name, args.model_id, args.device)

    t_total0 = time.perf_counter()
    timing_remask_s: list[float] = []
    timing_pack_s: list[float] = []
    n_records_written = 0

    with out_path.open("a", encoding="utf-8") as fout:
        for rec in tqdm(records, desc="remask"):
            task_id: str = rec["task_id"]
            if task_id in done_ids:
                continue

            benchmark = infer_benchmark(rec, task_id) or "evalplus"
            is_livebench = benchmark in ("livebench-coding", "livecodebench")

            prompt_text: str = rec.get("prompt", "")
            draft: str = (
                rec.get("raw_completion")
                or rec.get("solution")
                or rec.get("raw_solution")
                or ""
            )
            draft = clean_model_completion(draft, prompt_text)

            req = ModelRequest(
                prompt=prompt_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
            )

            try:
                t0 = time.perf_counter()
                refined = model.generate_with_remask(
                    req=req,
                    draft=draft,
                    confidence_threshold=args.confidence_threshold,
                    mask_ratio=args.mask_ratio,
                    mask_granularity=args.mask_granularity,
                    span_merge_gap=args.span_merge_gap,
                )
                t1 = time.perf_counter()
            except Exception as e:
                print(f"[warn] {task_id}: generate_with_remask failed ({e}); keeping draft.")
                refined = draft
                t1 = time.perf_counter()
                t0 = t1

            t_pack0 = time.perf_counter()
            refined = clean_model_completion(refined, prompt_text)
            # EvalPlus: 需要把 refined completion 封装成完整 solution 程序
            # LiveBench: 官方评测直接使用 solution 字段，无需 prepend prompt。
            if is_livebench:
                solution = refined
            elif benchmark == "bigcodebench":
                solution = build_prompt_scaffold_solution(prompt_text, refined)
            else:
                solution = build_evalplus_solution(prompt_text, refined)
            t_pack1 = time.perf_counter()

            timing_remask_s.append(t1 - t0)
            timing_pack_s.append(t_pack1 - t_pack0)

            out_rec = {
                "task_id":           task_id,
                "benchmark":         benchmark,
                "prompt":            prompt_text,
                "draft_completion":  draft,       # original AR output
                "raw_completion":    refined,     # diffusion-refined output
                "solution":          solution,
                "model":             model.name,
                "gen": {
                    "source_model":         rec.get("model", "unknown"),
                    "refiner":              refiner_name,
                    "confidence_threshold": args.confidence_threshold,
                    "mask_ratio":           args.mask_ratio,
                    "mask_granularity":     args.mask_granularity,
                    "span_merge_gap":       args.span_merge_gap,
                    "temperature":          args.temperature,
                    "top_p":                args.top_p,
                    "seed":                 args.seed,
                    "timing": {
                        "remask_generate_s": t1 - t0,
                        "pack_solution_s": t_pack1 - t_pack0,
                        "total_s": (t1 - t0) + (t_pack1 - t_pack0),
                    },
                },
            }

            # LiveBench: 保留 question_id / meta，方便 eval_livebench.py 直接复用。
            if is_livebench:
                if "question_id" in rec:
                    out_rec["question_id"] = rec["question_id"]
                if "meta" in rec:
                    out_rec["meta"] = rec["meta"]
            if benchmark == "bigcodebench":
                for key in ("split", "subset", "revision", "source_prompt"):
                    if key in rec:
                        out_rec[key] = rec[key]
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_records_written += 1

    t_total1 = time.perf_counter()
    timing_path = str(out_path) + ".timing_summary.json"
    summary = {
        "script": "gen_remask",
        "out": str(out_path.resolve()),
        "model": model.name,
        "refiner": refiner_name,
        "num_shards": args.num_shards,
        "shard_idx": args.shard_idx,
        "n_records_written": n_records_written,
        "timing": {
            "total_s": t_total1 - t_total0,
            "remask_generate_s_total": float(sum(timing_remask_s)),
            "pack_solution_s_total": float(sum(timing_pack_s)),
            "remask_generate_s_avg": (float(sum(timing_remask_s)) / len(timing_remask_s)) if timing_remask_s else None,
        },
    }
    out_path.with_suffix(out_path.suffix + ".timing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[timing] wrote {timing_path}")

    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()

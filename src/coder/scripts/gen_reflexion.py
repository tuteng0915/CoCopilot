#!/usr/bin/env python3
"""
gen_reflexion.py — Reflexion baseline (Shinn et al., 2023), simplified for this repo.

Minimal workflow (1+ rounds, default 1):
  1) Given a problem prompt and a prior attempt (draft code), produce a short reflection:
     - what likely went wrong
     - concrete fixes to apply
  2) Generate a revised solution using the reflection as additional context.

Notes:
  - This is a lightweight baseline without automatic execution feedback.
  - If the input JSONL contains failure feedback (e.g., sanitizer/evaluator messages),
    you can pass --feedback_key to include it in reflection.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from tqdm import tqdm

from coder.models import (
    ApiCoder,
    DeepSeekCoder,
    QwenCoder,
    Qwen35Coder,
    LLaDACoder,
    StarCoder2Coder,
    MistralCoder,
    Llama31Coder,
    DiffuLLaMACoder,
    SeedDiffCoder,
    SeedCoder,
    CoderModel,
)
from coder.utils.schema import ModelRequest


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_model(name: str, device: str, model_id: Optional[str]) -> CoderModel:
    name = (name or "").lower()
    if name in ["deepseek", "deepseek_coder", "ds"]:
        return DeepSeekCoder(model_id=model_id or "deepseek-ai/deepseek-coder-6.7b-instruct", device=device)
    if name in ["qwen", "qwen_coder"]:
        return QwenCoder(model_id=model_id or "Qwen/Qwen2.5-Coder-7B-Instruct", device=device)
    if name in ["qwen35", "qwen35_coder", "qwen3.5"]:
        return Qwen35Coder(model_id=model_id or "Qwen/Qwen3.5-4B", device=device)
    if name in ["llada", "llada_coder"]:
        return LLaDACoder(model_id=model_id or "GSAI-ML/LLaDA-8B-Instruct", device=device)
    if name in ["starcoder2", "starcoder2_coder", "sc2"]:
        return StarCoder2Coder(model_id=model_id or "bigcode/starcoder2-7b", device=device)
    if name in ["mistral", "mistral_coder"]:
        return MistralCoder(model_id=model_id or "mistralai/Mistral-7B-Instruct-v0.3", device=device)
    if name in ["llama31", "llama31_coder", "llama3.1"]:
        return Llama31Coder(model_id=model_id or "meta-llama/Llama-3.1-8B-Instruct", device=device)
    if name in ["diffullama", "diffullama_coder", "dflm"]:
        return DiffuLLaMACoder(model_id=model_id, device=device)
    if name in ["seed-diffcoder", "seed_diffcoder", "seeddiffcoder"]:
        return SeedDiffCoder(model_id=model_id, device=device)
    if name in ["seed-coder", "seed_coder", "seedcoder"]:
        return SeedCoder(model_id=model_id, device=device)
    if name in ["api", "api_coder", "closed_api"]:
        return ApiCoder(model_id=model_id, device="api")
    raise ValueError(f"Unknown --model: {name}")


def build_evalplus_solution(prompt_text: str, gen: str) -> str:
    g = (gen or "").lstrip()
    if re.search(r"(?m)^(def|class|import|from)\s+", g):
        return g.rstrip()
    return (prompt_text.rstrip() + "\n" + gen.lstrip()).rstrip()


def get_nested(obj: Dict[str, Any], dotted_key: str) -> Any:
    """
    Support dotted keys like "eval.error" to fetch nested values.
    """
    cur: Any = obj
    for part in dotted_key.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def main() -> None:
    ap = argparse.ArgumentParser(description="Reflexion baseline (reflection + revise), default 1 round.")
    ap.add_argument("--input", required=True, help="Input JSONL from AR generation (EvalPlus or LiveBench).")
    ap.add_argument("--out", required=True, help="Output JSONL path for reflexion-refined samples.")

    ap.add_argument("--model", required=True, help="Backbone model to generate reflection/revision.")
    ap.add_argument("--model_id", default=None, help="Override model id.")
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--rounds", type=int, default=1, help="Number of reflexion rounds (default 1).")
    ap.add_argument("--feedback_key", default=None, help="Optional key (supports dotted) to include feedback text.")

    ap.add_argument("--max_new_tokens_reflection", type=int, default=256)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=3407)

    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.rounds < 1:
        ap.error("--rounds must be >= 1")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if args.resume and out_path.exists():
        for rec in read_jsonl(str(out_path)):
            tid = rec.get("task_id")
            if isinstance(tid, str) and tid:
                done_ids.add(tid)
        print(f"[resume] skipping {len(done_ids)} already-done tasks.")

    model = build_model(args.model, device=args.device, model_id=args.model_id)

    records = list(read_jsonl(args.input))
    print(f"[info] {len(records)} records loaded from {args.input}")

    t_total0 = time.perf_counter()
    timing_reflect_s: list[float] = []
    timing_revise_s: list[float] = []
    n_records_written = 0

    with out_path.open("a", encoding="utf-8") as fout:
        for rec in tqdm(records, desc=f"reflexion({model.name})"):
            task_id = rec.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                continue
            if task_id in done_ids:
                continue

            is_livebench = "question_id" in rec or str(task_id).startswith("LiveBench/") or str(task_id).startswith("LiveCodeBench/")
            prompt_text: str = rec.get("prompt", "")
            draft0: str = rec.get("raw_completion", "") or rec.get("solution", "")

            feedback_text = ""
            if args.feedback_key:
                v = get_nested(rec, args.feedback_key)
                if isinstance(v, str) and v.strip():
                    feedback_text = v.strip()

            cur = draft0
            reflections: list[str] = []
            trace: list[Dict[str, Any]] = []

            for r in range(int(args.rounds)):
                reflection_prompt = (
                    "You are analyzing a previous Python solution attempt.\n"
                    "Write a concise REFLECTION with:\n"
                    "1) Likely failure points / bugs\n"
                    "2) Specific fixes to apply\n"
                    "Do NOT output code in this reflection.\n\n"
                    "[Problem]\n"
                    f"{prompt_text}\n\n"
                    "[Previous attempt]\n"
                    f"{cur}\n"
                )
                if feedback_text:
                    reflection_prompt += f"\n[Feedback]\n{feedback_text}\n"

                ref_req = ModelRequest(
                    prompt=reflection_prompt,
                    max_new_tokens=args.max_new_tokens_reflection,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=args.seed,
                )

                t_ref0 = time.perf_counter()
                try:
                    reflection = model.generate(ref_req).strip()
                except Exception as e:
                    reflection = f"(reflection failed: {e})"
                t_ref1 = time.perf_counter()
                timing_reflect_s.append(t_ref1 - t_ref0)

                reflections.append(reflection)

                revise_prompt = (
                    "You are improving your previous Python solution.\n"
                    "Use the REFLECTION notes to fix the solution.\n"
                    "Output ONLY valid Python code (no explanations).\n\n"
                    "[Problem]\n"
                    f"{prompt_text}\n\n"
                    "[Previous attempt]\n"
                    f"{cur}\n\n"
                    "[REFLECTION]\n"
                    f"{reflection}\n"
                )

                rev_req = ModelRequest(
                    prompt=revise_prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=args.seed,
                )

                t_rev0 = time.perf_counter()
                try:
                    nxt = model.generate(rev_req).strip()
                except Exception as e:
                    nxt = cur
                t_rev1 = time.perf_counter()
                timing_revise_s.append(t_rev1 - t_rev0)

                trace.append(
                    {
                        "round": r,
                        "input": cur,
                        "reflection": reflection,
                        "output": nxt,
                        "timing": {
                            "reflect_s": t_ref1 - t_ref0,
                            "revise_s": t_rev1 - t_rev0,
                            "total_s": (t_ref1 - t_ref0) + (t_rev1 - t_rev0),
                        },
                    }
                )

                cur = nxt

            final = cur
            if is_livebench:
                solution = final
            else:
                solution = build_evalplus_solution(prompt_text, final)

            out_rec: Dict[str, Any] = {
                "task_id": task_id,
                "prompt": prompt_text,
                "draft_completion": draft0,
                "raw_completion": final,
                "solution": solution,
                "reflection": reflections[-1] if reflections else "",
                "reflexion_trace": trace,
                "model": f"reflexion::{model.name}",
                "gen": {
                    "source_model": rec.get("model", "unknown"),
                    "reflexion_model": model.name,
                    "rounds": args.rounds,
                    "max_new_tokens_reflection": args.max_new_tokens_reflection,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "seed": args.seed,
                    "feedback_key": args.feedback_key,
                    "timing": {
                        "reflect_s_total": float(sum([x["timing"]["reflect_s"] for x in trace])) if trace else 0.0,
                        "revise_s_total": float(sum([x["timing"]["revise_s"] for x in trace])) if trace else 0.0,
                        "total_s": float(sum([x["timing"]["total_s"] for x in trace])) if trace else 0.0,
                    },
                },
            }

            if is_livebench:
                if "question_id" in rec:
                    out_rec["question_id"] = rec["question_id"]
                if "meta" in rec:
                    out_rec["meta"] = rec["meta"]
                if "benchmark" in rec:
                    out_rec["benchmark"] = rec["benchmark"]

            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_records_written += 1

    t_total1 = time.perf_counter()
    timing_path = str(out_path) + ".timing_summary.json"
    summary = {
        "script": "gen_reflexion",
        "out": str(out_path.resolve()),
        "model": model.name,
        "rounds": args.rounds,
        "n_records_written": n_records_written,
        "timing": {
            "total_s": t_total1 - t_total0,
            "reflect_s_total": float(sum(timing_reflect_s)),
            "revise_s_total": float(sum(timing_revise_s)),
            "reflect_s_avg": (float(sum(timing_reflect_s)) / len(timing_reflect_s)) if timing_reflect_s else None,
            "revise_s_avg": (float(sum(timing_revise_s)) / len(timing_revise_s)) if timing_revise_s else None,
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


#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, Dict, Any

from tqdm import tqdm

from coder.utils.code_cleaning import build_evalplus_solution
from coder.utils.schema import ModelRequest
from coder.models import DeepSeekCoder, QwenCoder, Llama31Coder, StarCoder2Coder, CoderModel


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_model(name: str, device: str, model_id: str | None) -> CoderModel:
    name = name.lower()
    if name in ["deepseek", "deepseek_coder", "ds"]:
        return DeepSeekCoder(
            model_id=model_id or "deepseek-ai/deepseek-coder-6.7b-instruct",
            device=device,
        )
    if name in ["qwen", "qwen_coder"]:
        return QwenCoder(
            model_id=model_id or "Qwen/Qwen2.5-Coder-7B-Instruct",
            device=device,
        )
    if name in ["llama31", "llama31_coder", "llama3.1"]:
        return Llama31Coder(
            model_id=model_id or "meta-llama/Llama-3.1-8B-Instruct",
            device=device,
        )
    if name in ["starcoder2", "starcoder2_coder", "sc2"]:
        return StarCoder2Coder(
            model_id=model_id or "bigcode/starcoder2-7b",
            device=device,
        )
    raise ValueError(f"Unsupported self-refine backbone: {name}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Self-refine AR drafts by feeding problem + initial solution back to an AR model."
    )
    ap.add_argument("--input", required=True, help="Input JSONL from AR generation (EvalPlus or LiveBench).")
    ap.add_argument("--out", required=True, help="Output JSONL path for self-refined samples.")

    ap.add_argument("--model", required=True, help="Self-refine backbone: deepseek | qwen")
    ap.add_argument("--model_id", default=None, help="Override HuggingFace model id for the backbone.")
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=3407)

    ap.add_argument("--resume", action="store_true", help="Skip task_ids already present in --out (append mode).")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if args.resume and out_path.exists():
        for rec in read_jsonl(str(out_path)):
            done_ids.add(rec["task_id"])
        print(f"[resume] skipping {len(done_ids)} already-done tasks.")

    model = build_model(args.model, device=args.device, model_id=args.model_id)

    records = list(read_jsonl(args.input))
    print(f"[info] {len(records)} records loaded from {args.input}")

    t_total0 = time.perf_counter()
    timing_refine_s: list[float] = []
    timing_pack_s: list[float] = []
    n_records_written = 0

    with out_path.open("a", encoding="utf-8") as fout:
        for rec in tqdm(records, desc=f"self_refine({model.name})"):
            task_id: str = rec["task_id"]
            if task_id in done_ids:
                continue

            is_livebench = "question_id" in rec or str(task_id).startswith("LiveBench/")

            prompt_text: str = rec.get("prompt", "")
            draft: str = rec.get("raw_completion", "") or rec.get("solution", "")

            # Build a generic self-refinement prompt that embeds both problem and initial attempt.
            refine_prompt = (
                "You are improving your previous Python solution.\n\n"
                "[Problem]\n"
                f"{prompt_text}\n\n"
                "[Your previous attempt]\n"
                f"{draft}\n\n"
                "Please output a corrected and clean Python solution. "
                "Only output valid Python code (no explanations)."
            )

            req = ModelRequest(
                prompt=refine_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
            )

            try:
                t0 = time.perf_counter()
                refined = model.generate(req)
                t1 = time.perf_counter()
            except Exception as e:
                print(f"[warn] {task_id}: self-refine generation failed ({e}); keeping draft.")
                refined = draft
                t1 = time.perf_counter()
                t0 = t1

            t_pack0 = time.perf_counter()
            if is_livebench:
                solution = refined
            else:
                solution = build_evalplus_solution(prompt_text, refined)
            t_pack1 = time.perf_counter()

            timing_refine_s.append(t1 - t0)
            timing_pack_s.append(t_pack1 - t_pack0)

            out_rec: Dict[str, Any] = {
                "task_id":          task_id,
                "prompt":           prompt_text,
                "draft_completion": draft,
                "raw_completion":   refined,
                "solution":         solution,
                "model":            f"self_refine::{model.name}",
                "gen": {
                    "source_model":      rec.get("model", "unknown"),
                    "self_refine_model": model.name,
                    "max_new_tokens":    args.max_new_tokens,
                    "temperature":       args.temperature,
                    "top_p":             args.top_p,
                    "seed":              args.seed,
                    "timing": {
                        "refine_generate_s": t1 - t0,
                        "pack_solution_s": t_pack1 - t_pack0,
                        "total_s": (t1 - t0) + (t_pack1 - t_pack0),
                    },
                },
            }

            if is_livebench:
                if "question_id" in rec:
                    out_rec["question_id"] = rec["question_id"]
                if "meta" in rec:
                    out_rec["meta"] = rec["meta"]

            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_records_written += 1

    t_total1 = time.perf_counter()
    timing_path = str(out_path) + ".timing_summary.json"
    summary = {
        "script": "gen_self_refine",
        "out": str(out_path.resolve()),
        "model": model.name,
        "n_records_written": n_records_written,
        "timing": {
            "total_s": t_total1 - t_total0,
            "refine_generate_s_total": float(sum(timing_refine_s)),
            "pack_solution_s_total": float(sum(timing_pack_s)),
            "refine_generate_s_avg": (float(sum(timing_refine_s)) / len(timing_refine_s)) if timing_refine_s else None,
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

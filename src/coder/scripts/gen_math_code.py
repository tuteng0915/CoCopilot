"""Generation script for math benchmarks in code mode: GSM8K and MATH-500.

Usage examples:
  python -m coder.scripts.gen_math_code --model deepseek --dataset gsm8k --out outputs/math_code/deepseek_gsm8k_code.jsonl
  python -m coder.scripts.gen_math_code --model qwen --dataset math500 --out outputs/math_code/qwen_math500_code.jsonl
"""
import argparse
import json
import os
import time
from typing import Any, Dict

from tqdm import tqdm

from coder.scripts.gen_math import (
    build_model,
    load_gsm8k,
    load_math500,
    select_tasks,
)
from coder.utils.schema import ModelRequest
from coder.utils.sharding import take_shard, validate_shard_args


GSM8K_CODE_PROMPT = """\
Write a Python function `solution()` that solves the following math problem.
The function must return a single numeric value (int or float).
Do NOT use input(). Do NOT print inside the function.
Only output the function definition, no extra text.

Problem: {question}

def solution():
"""

MATH500_CODE_PROMPT = """\
Write a Python function `solution()` that solves the following math problem.
The function must return the exact answer as a string or number.
You may use sympy. Do NOT use input(). Do NOT print.
Only output the function definition, no extra text.

Problem: {question}

def solution():
"""

AIME_CODE_PROMPT = """\
Write a Python function `solution()` that solves the following AIME competition math problem.
The function must return a single non-negative integer between 000 and 999 (inclusive).
Do NOT use input(). Do NOT print inside the function.
Only output the function definition, no extra text.

Problem: {question}

def solution():
"""


def load_aime() -> list[Dict[str, Any]]:
    """Load AIME 2022-2024 problems from AI-MO/aimo-validation-aime.

    Dataset fields: problem, answer (integer string), id.
    All AIME answers are integers in [0, 999].
    """
    ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
    items = []
    for row in ds:
        raw_answer = row.get("answer", "")
        # answer may be int or string representation of int
        answer_str = str(int(str(raw_answer).strip())) if str(raw_answer).strip().isdigit() else str(raw_answer).strip()
        items.append({
            "id": f"aime/{row.get('id', len(items))}",
            "question": row["problem"],
            "answer": answer_str,
        })
    return items


def load_aime2025() -> list[Dict[str, Any]]:
    """Load AIME 2025 problems from MathArena/aime_2025.

    Dataset fields vary; we try common field names for problem and answer.
    All AIME answers are integers in [0, 999].
    """
    ds = load_dataset("MathArena/aime_2025", split="test")
    items = []
    for i, row in enumerate(ds):
        question = row.get("problem") or row.get("question") or ""
        raw_answer = row.get("answer") or row.get("solution") or ""
        answer_str = str(int(str(raw_answer).strip())) if str(raw_answer).strip().isdigit() else str(raw_answer).strip()
        pid = row.get("id") or row.get("problem_id") or i
        items.append({
            "id": f"aime2025/{pid}",
            "question": question,
            "answer": answer_str,
        })
    return items


_LOADERS = {
    "gsm8k": load_gsm8k,
    "math500": load_math500,
    "aime": load_aime,
    "aime2025": load_aime2025,
}


def build_prompt(dataset: str, item: Dict[str, Any]) -> str:
    if dataset == "gsm8k":
        return GSM8K_CODE_PROMPT.format(question=item["question"])
    if dataset in ("aime", "aime2025"):
        return AIME_CODE_PROMPT.format(question=item["question"])
    return MATH500_CODE_PROMPT.format(question=item["question"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        choices=[
            "dream", "deepseek", "qwen", "qwen35", "llada",
            "starcoder2", "mistral", "llama31", "diffullama",
            "seed-diffcoder", "seed-coder", "api",
        ],
        required=True,
    )
    ap.add_argument("--dataset", choices=["gsm8k", "math500", "aime", "aime2025"], required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--limit", type=int, default=0, help="Only run first N problems. 0 = all.")
    ap.add_argument("--task_ids", type=str, default=None, help="Comma-separated problem ids to run.")
    ap.add_argument("--task_ids_file", type=str, default=None, help="Newline-delimited file of problem ids to run.")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Skip problems already written to --out.")

    ap.add_argument("--max_new_tokens", type=int, default=512, help="Code-mode math solutions are shorter than CoT.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--num_samples", type=int, default=1, help="Samples per problem (for pass@n).")
    ap.add_argument("--num_shards", type=int, default=1, help="Total number of shards (for parallel runs).")
    ap.add_argument("--shard_idx", type=int, default=0, help="Which shard to run (0-indexed).")

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--model_id", type=str, default=None, help="Override HuggingFace model id.")

    args = ap.parse_args()
    if args.num_samples < 1:
        ap.error("--num_samples must be >= 1")
    try:
        validate_shard_args(num_shards=args.num_shards, shard_idx=args.shard_idx)
    except ValueError as e:
        ap.error(str(e))

    print(f"[data] loading {args.dataset} ...")
    items = _LOADERS[args.dataset]()
    print(f"[data] {len(items)} problems loaded")

    task_ids: list[str] | None = None
    if args.task_ids or args.task_ids_file:
        ids_set: set[str] = set()
        if args.task_ids:
            ids_set.update(x.strip() for x in args.task_ids.split(",") if x.strip())
        if args.task_ids_file:
            with open(args.task_ids_file, encoding="utf-8") as fids:
                ids_set.update(line.strip() for line in fids if line.strip())
        task_ids = list(ids_set)

    selected = select_tasks(items, limit=args.limit, task_ids=task_ids, shuffle=args.shuffle, seed=args.seed)
    selected = take_shard(selected, num_shards=args.num_shards, shard_idx=args.shard_idx)
    print(f"[data] {len(selected)} problems selected (shard {args.shard_idx}/{args.num_shards})")

    model = build_model(args.model, device=args.device, model_id=args.model_id)

    done_keys: set[tuple[str, int]] = set()
    out_path = os.path.abspath(args.out)
    if args.resume and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as existing:
            for line in existing:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    done_keys.add((obj["id"], obj.get("sample_id", 0)))
                except Exception:
                    continue
        print(f"[resume] skipping {len(done_keys)} already-done records")

    t_total0 = time.perf_counter()
    timing_generate_s: list[float] = []
    n_records_written = 0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    open_mode = "a" if args.resume else "w"
    with open(out_path, open_mode, encoding="utf-8") as fout:
        for item in tqdm(selected, desc=f"gen_code:{args.model}:{args.dataset}"):
            prompt = build_prompt(args.dataset, item)
            base_req = ModelRequest(
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
            )

            for sample_idx in range(args.num_samples):
                if (item["id"], sample_idx) in done_keys:
                    continue

                req = base_req
                if args.num_samples > 1 and req.seed is not None:
                    req = ModelRequest(
                        prompt=base_req.prompt,
                        max_new_tokens=base_req.max_new_tokens,
                        temperature=base_req.temperature,
                        top_p=base_req.top_p,
                        seed=int(args.seed) + sample_idx,
                    )

                t0 = time.perf_counter()
                raw_gen = model.generate(req)
                t1 = time.perf_counter()
                timing_generate_s.append(t1 - t0)

                rec = {
                    "id": item["id"],
                    "sample_id": sample_idx,
                    "question": item["question"],
                    "prompt": prompt,
                    "answer_ref": item["answer"],
                    "raw_completion": raw_gen,
                    "code_mode": True,
                    "model": model.name,
                    "dataset": args.dataset,
                    "gen": {
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "seed": req.seed,
                        "num_samples": args.num_samples,
                        "sample_id": sample_idx,
                        "timing": {
                            "generate_s": t1 - t0,
                            "total_s": t1 - t0,
                        },
                    },
                }
                if args.dataset == "math500":
                    rec["subject"] = item.get("subject", "")
                    rec["level"] = item.get("level", "")

                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n_records_written += 1

    t_total1 = time.perf_counter()

    summary = {
        "script": "gen_math_code",
        "out": out_path,
        "model": model.name,
        "dataset": args.dataset,
        "num_samples": args.num_samples,
        "num_shards": args.num_shards,
        "shard_idx": args.shard_idx,
        "n_records_written": n_records_written,
        "timing": {
            "total_s": t_total1 - t_total0,
            "generate_s_total": float(sum(timing_generate_s)),
            "generate_s_avg": float(sum(timing_generate_s)) / len(timing_generate_s) if timing_generate_s else None,
        },
    }
    timing_path = out_path + ".timing_summary.json"
    with open(timing_path, "w", encoding="utf-8") as tf:
        json.dump(summary, tf, ensure_ascii=False, indent=2)

    print(f"[samples] wrote {out_path}  ({n_records_written} records)")
    print(f"[timing]  wrote {timing_path}")


if __name__ == "__main__":
    main()

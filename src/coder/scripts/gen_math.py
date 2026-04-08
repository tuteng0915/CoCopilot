"""Generation script for math benchmarks: GSM8K and MATH-500.

Usage examples:
  python -m coder.scripts.gen_math --model dream --dataset gsm8k --out outputs/gsm8k_dream.jsonl
  python -m coder.scripts.gen_math --model qwen  --dataset math500 --out outputs/math500_qwen.jsonl
"""
import argparse
import json
import os
import random
import re
import time
from typing import List, Tuple, Dict, Any

from datasets import load_dataset
from tqdm import tqdm

from coder.utils.sharding import take_shard, validate_shard_args
from coder.models import (
    CoderModel,
    DreamCoder,
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
    ApiCoder,
)
from coder.utils.schema import ModelRequest


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_gsm8k() -> List[Dict[str, Any]]:
    """Load GSM8K test split. Each item: {id, question, answer, answer_number}."""
    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = []
    for i, row in enumerate(ds):
        # Ground truth: last line of answer field is "#### <number>"
        raw_answer = row["answer"]
        m = re.search(r"####\s*(.+)$", raw_answer, re.MULTILINE)
        answer_str = m.group(1).strip().replace(",", "") if m else raw_answer.strip()
        items.append({
            "id": f"gsm8k/{i}",
            "question": row["question"],
            "answer": answer_str,          # the number string, e.g. "72"
            "answer_raw": raw_answer,      # full chain-of-thought answer
        })
    return items


def load_math500() -> List[Dict[str, Any]]:
    """Load MATH-500 test split. Each item: {id, problem, answer, subject, level}."""
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    items = []
    for i, row in enumerate(ds):
        items.append({
            "id": f"math500/{row.get('unique_id', i)}",
            "question": row["problem"],
            "answer": row["answer"],       # already extracted, e.g. "\\frac{1}{2}"
            "subject": row.get("subject", ""),
            "level": row.get("level", ""),
        })
    return items


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

GSM8K_PROMPT_TMPL = (
    "Solve the following math problem step by step.\n"
    "At the end of your solution, write your final answer on a new line "
    "in the format: #### <number>\n\n"
    "Problem: {question}\n"
    "Solution:"
)

MATH500_PROMPT_TMPL = (
    "Solve the following math problem. Show your reasoning step by step.\n"
    "At the end, write your final answer inside \\boxed{{}}.\n\n"
    "Problem: {question}\n"
    "Solution:"
)


def build_prompt(dataset: str, item: Dict[str, Any]) -> str:
    if dataset == "gsm8k":
        return GSM8K_PROMPT_TMPL.format(question=item["question"])
    else:
        return MATH500_PROMPT_TMPL.format(question=item["question"])


# ---------------------------------------------------------------------------
# Model factory (mirrors gen_evalplus.py)
# ---------------------------------------------------------------------------

def build_model(name: str, device: str, model_id: str | None) -> CoderModel:
    if name == "dream":
        return DreamCoder(model_id=model_id or "Dream-org/Dream-Coder-v0-Instruct-7B", device=device)
    if name == "deepseek":
        return DeepSeekCoder(model_id=model_id or "deepseek-ai/deepseek-coder-6.7b-instruct", device=device)
    if name == "qwen":
        return QwenCoder(model_id=model_id or "Qwen/Qwen2.5-Coder-7B-Instruct", device=device)
    if name == "qwen35":
        return Qwen35Coder(model_id=model_id or "Qwen/Qwen3.5-4B", device=device)
    if name == "llada":
        return LLaDACoder(model_id=model_id or "GSAI-ML/LLaDA-8B-Instruct", device=device)
    if name == "starcoder2":
        return StarCoder2Coder(model_id=model_id or "bigcode/starcoder2-7b", device=device)
    if name == "mistral":
        return MistralCoder(model_id=model_id or "mistralai/Mistral-7B-Instruct-v0.3", device=device)
    if name == "llama31":
        return Llama31Coder(model_id=model_id or "meta-llama/Llama-3.1-8B-Instruct", device=device)
    if name == "diffullama":
        return DiffuLLaMACoder(model_id=model_id, device=device)
    if name == "seed-diffcoder":
        return SeedDiffCoder(model_id=model_id, device=device)
    if name == "seed-coder":
        return SeedCoder(model_id=model_id, device=device)
    if name == "api":
        return ApiCoder(model_id=model_id, device="api")
    raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Task selection helpers
# ---------------------------------------------------------------------------

def select_tasks(
    items: List[Dict[str, Any]],
    limit: int,
    task_ids: List[str] | None,
    shuffle: bool,
    seed: int,
) -> List[Dict[str, Any]]:
    if task_ids:
        wanted = set(task_ids)
        items = [it for it in items if it["id"] in wanted]

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(items)

    if limit > 0:
        items = items[:limit]

    return items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
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
    ap.add_argument("--dataset", choices=["gsm8k", "math500"], required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--limit", type=int, default=0, help="Only run first N problems. 0 = all.")
    ap.add_argument("--task_ids", type=str, default=None, help="Comma-separated problem ids to run.")
    ap.add_argument("--task_ids_file", type=str, default=None, help="Newline-delimited file of problem ids to run.")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Skip problems already written to --out.")

    ap.add_argument("--max_new_tokens", type=int, default=1024, help="Longer than code since math needs chain-of-thought.")
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
    items = load_gsm8k() if args.dataset == "gsm8k" else load_math500()
    print(f"[data] {len(items)} problems loaded")

    # Merge comma-sep and file-based task_id filters
    task_ids: List[str] | None = None
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

    # Resume: collect already-done (id, sample_id) pairs
    done_keys: set[tuple[str, int]] = set()
    out_path = os.path.abspath(args.out)
    if args.resume and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    _obj = json.loads(_line)
                    done_keys.add((_obj["id"], _obj.get("sample_id", 0)))
                except Exception:
                    continue
        print(f"[resume] skipping {len(done_keys)} already-done records")

    t_total0 = time.perf_counter()
    timing_generate_s: list[float] = []
    n_records_written = 0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    open_mode = "a" if args.resume else "w"
    with open(out_path, open_mode, encoding="utf-8") as f:
        for item in tqdm(selected, desc=f"gen:{args.model}:{args.dataset}"):
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
                    "answer_ref": item["answer"],          # ground-truth answer
                    "raw_completion": raw_gen,
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
                # Carry over per-dataset metadata
                if args.dataset == "math500":
                    rec["subject"] = item.get("subject", "")
                    rec["level"] = item.get("level", "")

                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n_records_written += 1

    t_total1 = time.perf_counter()

    summary = {
        "script": "gen_math",
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

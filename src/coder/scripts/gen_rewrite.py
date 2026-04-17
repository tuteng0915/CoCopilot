"""Generation script for text rewriting benchmarks: ASSET and CoEdIT.

Supported datasets:
  asset    — sentence simplification, 359 test items, 10 reference rewrites each
  coedit   — multi-task editing (GEC / paraphrase / neutralize), validation split

Usage (AR baseline):
  python -m coder.scripts.gen_rewrite --model llama31 --dataset asset \\
      --out outputs/rewrite/asset_llama31.jsonl

Usage (DreamGeneral standalone):
  python -m coder.scripts.gen_rewrite --model dream_general --dataset asset \\
      --out outputs/rewrite/asset_dream_general.jsonl

The output JSONL is compatible with gen_remask.py (id / prompt / raw_completion /
answer_ref fields), so the CoCoder pipeline can be applied with:
  python -m coder.scripts.gen_remask \\
      --input outputs/rewrite/asset_llama31.jsonl \\
      --out   outputs/rewrite/asset_llama31_dream_general_t0.9.jsonl \\
      --refiner dream_general ...
"""
import argparse
import json
import os
import random
import time
from typing import Any, Dict, List

from datasets import load_dataset
from tqdm import tqdm

from coder.models import (
    CoderModel,
    DreamCoder,
    DreamGeneral,
    LLaDACoder,
    DeepSeekCoder,
    QwenCoder,
    Llama31Coder,
    MistralCoder,
    ApiCoder,
)
from coder.utils.schema import ModelRequest
from coder.utils.sharding import take_shard, validate_shard_args


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_asset() -> List[Dict[str, Any]]:
    """ASSET sentence simplification test set (359 items, 10 refs each)."""
    ds = load_dataset("asset", "simplification", split="test")
    items = []
    for i, row in enumerate(ds):
        items.append({
            "id": f"asset/{i}",
            "original": row["original"],
            # Keep all 10 references for eval; store first as primary answer_ref
            "answer_ref": row["simplifications"][0],
            "references": row["simplifications"],  # full list for SARI
            "task": "simplification",
        })
    return items


# CoEdIT tasks to include from the validation split.
# 'simplification' is only in train, so we use gec + paraphrase here.
_COEDIT_TASKS = {"gec", "paraphrase", "neutralize"}


def load_coedit(tasks: set[str] | None = None) -> List[Dict[str, Any]]:
    """CoEdIT validation split, filtered to requested tasks."""
    if tasks is None:
        tasks = _COEDIT_TASKS
    ds = load_dataset("grammarly/coedit", split="validation")
    items = []
    for row in ds:
        if row["task"] not in tasks:
            continue
        items.append({
            "id": f"coedit/{row['_id']}",
            "original": row["src"],    # already contains instruction prefix
            "answer_ref": row["tgt"],
            "references": [row["tgt"]],
            "task": row["task"],
        })
    return items


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

ASSET_PROMPT = (
    "Simplify the following sentence. "
    "Keep the meaning but use simpler words and shorter structure.\n\n"
    "Original: {original}\n\n"
    "Simplified:"
)

# CoEdIT src already contains the instruction (e.g. "Fix grammaticality: ...")
COEDIT_PROMPT = "{original}\n\nRewritten:"


def build_prompt(dataset: str, item: Dict[str, Any]) -> str:
    if dataset == "asset":
        return ASSET_PROMPT.format(original=item["original"])
    else:
        return COEDIT_PROMPT.format(original=item["original"])


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(name: str, device: str, model_id: str | None) -> CoderModel:
    if name == "dream_general":
        return DreamGeneral(model_id=model_id or "Dream-org/Dream-v0-Instruct-7B", device=device)
    if name == "dream_coder":
        return DreamCoder(model_id=model_id or "Dream-org/Dream-Coder-v0-Instruct-7B", device=device)
    if name == "llada":
        return LLaDACoder(model_id=model_id or "GSAI-ML/LLaDA-8B-Instruct", device=device)
    if name == "deepseek":
        return DeepSeekCoder(model_id=model_id or "deepseek-ai/deepseek-coder-6.7b-instruct", device=device)
    if name == "qwen":
        return QwenCoder(model_id=model_id or "Qwen/Qwen2.5-Coder-7B-Instruct", device=device)
    if name == "llama31":
        return Llama31Coder(model_id=model_id or "meta-llama/Llama-3.1-8B-Instruct", device=device)
    if name == "mistral":
        return MistralCoder(model_id=model_id or "mistralai/Mistral-7B-Instruct-v0.3", device=device)
    if name == "api":
        return ApiCoder(model_id=model_id, device="api")
    raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Task selection
# ---------------------------------------------------------------------------

def select_tasks(
    items: List[Dict[str, Any]],
    limit: int,
    task_ids: List[str] | None,
    tasks_filter: List[str] | None,
    shuffle: bool,
    seed: int,
) -> List[Dict[str, Any]]:
    if task_ids:
        wanted = set(task_ids)
        items = [it for it in items if it["id"] in wanted]
    if tasks_filter:
        wanted_tasks = set(tasks_filter)
        items = [it for it in items if it.get("task") in wanted_tasks]
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
        choices=["dream_general", "dream_coder", "llada", "deepseek", "qwen",
                 "llama31", "mistral", "api"],
        required=True,
    )
    ap.add_argument("--dataset", choices=["asset", "coedit"], required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--coedit_tasks", type=str, default=None,
                    help="Comma-separated CoEdIT tasks to include. "
                         "Default: gec,paraphrase,neutralize")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--task_ids", type=str, default=None)
    ap.add_argument("--task_ids_file", type=str, default=None)
    ap.add_argument("--tasks_filter", type=str, default=None,
                    help="Comma-separated task names to filter (e.g. gec,paraphrase)")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--resume", action="store_true")

    ap.add_argument("--max_new_tokens", type=int, default=128,
                    help="Short — rewrite targets are sentence-level (default: 128)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--num_samples", type=int, default=1)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--model_id", type=str, default=None)
    args = ap.parse_args()

    validate_shard_args(num_shards=args.num_shards, shard_idx=args.shard_idx)

    print(f"[data] loading {args.dataset} ...")
    if args.dataset == "asset":
        items = load_asset()
    else:
        coedit_tasks = (
            set(t.strip() for t in args.coedit_tasks.split(",") if t.strip())
            if args.coedit_tasks else None
        )
        items = load_coedit(tasks=coedit_tasks)
    print(f"[data] {len(items)} items loaded")

    task_ids: List[str] | None = None
    if args.task_ids or args.task_ids_file:
        ids_set: set[str] = set()
        if args.task_ids:
            ids_set.update(x.strip() for x in args.task_ids.split(",") if x.strip())
        if args.task_ids_file:
            with open(args.task_ids_file) as f:
                ids_set.update(line.strip() for line in f if line.strip())
        task_ids = list(ids_set)

    tasks_filter = (
        [t.strip() for t in args.tasks_filter.split(",") if t.strip()]
        if args.tasks_filter else None
    )

    selected = select_tasks(
        items, args.limit, task_ids, tasks_filter, args.shuffle, args.seed
    )
    selected = take_shard(selected, num_shards=args.num_shards, shard_idx=args.shard_idx)
    print(f"[data] {len(selected)} items selected (shard {args.shard_idx}/{args.num_shards})")

    # Show task breakdown
    task_counts: Dict[str, int] = {}
    for it in selected:
        k = it.get("task", "?")
        task_counts[k] = task_counts.get(k, 0) + 1
    print(f"[data] task breakdown: {task_counts}")

    model = build_model(args.model, args.device, args.model_id)

    done_keys: set[tuple[str, int]] = set()
    out_path = os.path.abspath(args.out)
    if args.resume and os.path.exists(out_path):
        with open(out_path) as _f:
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
                    "original": item["original"],
                    "prompt": prompt,
                    # answer_ref mirrors gen_math.py convention (used by gen_remask.py)
                    "answer_ref": item["answer_ref"],
                    "references": item["references"],
                    "raw_completion": raw_gen,
                    "task": item.get("task", ""),
                    "model": model.name,
                    "dataset": args.dataset,
                    "gen": {
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "seed": req.seed,
                        "timing": {"generate_s": t1 - t0},
                    },
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n_records_written += 1

    t_total1 = time.perf_counter()

    summary = {
        "script": "gen_rewrite",
        "out": out_path,
        "model": model.name,
        "dataset": args.dataset,
        "n_records_written": n_records_written,
        "timing": {
            "total_s": t_total1 - t_total0,
            "generate_s_avg": (
                float(sum(timing_generate_s)) / len(timing_generate_s)
                if timing_generate_s else None
            ),
        },
    }
    timing_path = out_path + ".timing_summary.json"
    with open(timing_path, "w", encoding="utf-8") as tf:
        json.dump(summary, tf, indent=2)

    print(f"[samples] wrote {out_path}  ({n_records_written} records)")
    print(f"[timing]  wrote {timing_path}")


if __name__ == "__main__":
    main()

"""Generation script for deep research benchmarks: FRAMES and HotpotQA.

Both are run closed-book (no retrieval) to test the dLLM's knowledge + reasoning.

Usage:
  python -m coder.scripts.gen_research --model dream_general --dataset frames \\
      --out outputs/base_tuteng/frames_dream_general.jsonl

  python -m coder.scripts.gen_research --model llama31 --dataset hotpotqa \\
      --out outputs/base_tuteng/hotpotqa_llama31.jsonl
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

def load_frames() -> List[Dict[str, Any]]:
    """FRAMES benchmark — 824 multi-hop questions, closed-book."""
    ds = load_dataset("google/frames-benchmark", split="test")
    items = []
    for i, row in enumerate(ds):
        items.append({
            "id": f"frames/{i}",
            "question": row["Prompt"],
            "answer": row["Answer"],
            "reasoning_types": row.get("reasoning_types", ""),
        })
    return items


def load_hotpotqa() -> List[Dict[str, Any]]:
    """HotpotQA distractor validation set — 7405 multi-hop questions, closed-book."""
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    items = []
    for row in ds:
        items.append({
            "id": f"hotpotqa/{row['id']}",
            "question": row["question"],
            "answer": row["answer"],
            "type": row.get("type", ""),
            "level": row.get("level", ""),
        })
    return items


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

FRAMES_PROMPT = (
    "Answer the following question based on your knowledge. "
    "Think step by step and give a concise final answer.\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

HOTPOTQA_PROMPT = (
    "Answer the following question based on your knowledge. "
    "The answer should be a short phrase or entity.\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def build_prompt(dataset: str, item: Dict[str, Any]) -> str:
    if dataset == "frames":
        return FRAMES_PROMPT.format(question=item["question"])
    else:
        return HOTPOTQA_PROMPT.format(question=item["question"])


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
        choices=["dream_general", "dream_coder", "llada", "deepseek", "qwen", "llama31", "mistral", "api"],
        required=True,
    )
    ap.add_argument("--dataset", choices=["frames", "hotpotqa"], required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--limit", type=int, default=0, help="Only run first N items. 0 = all.")
    ap.add_argument("--task_ids", type=str, default=None)
    ap.add_argument("--task_ids_file", type=str, default=None)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--resume", action="store_true")

    ap.add_argument("--max_new_tokens", type=int, default=512)
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
    items = load_frames() if args.dataset == "frames" else load_hotpotqa()
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

    selected = select_tasks(items, args.limit, task_ids, args.shuffle, args.seed)
    selected = take_shard(selected, num_shards=args.num_shards, shard_idx=args.shard_idx)
    print(f"[data] {len(selected)} items selected (shard {args.shard_idx}/{args.num_shards})")

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
                    "question": item["question"],
                    "prompt": prompt,
                    "answer_ref": item["answer"],
                    "raw_completion": raw_gen,
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
                # carry over metadata
                if args.dataset == "frames":
                    rec["reasoning_types"] = item.get("reasoning_types", "")
                elif args.dataset == "hotpotqa":
                    rec["type"] = item.get("type", "")
                    rec["level"] = item.get("level", "")

                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n_records_written += 1

    t_total1 = time.perf_counter()

    summary = {
        "script": "gen_research",
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

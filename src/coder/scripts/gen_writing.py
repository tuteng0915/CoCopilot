"""Generation script for creative writing benchmark: WildBench Creative Writing subset.

WildBench v2 has 146 "Creative Writing" items with LLM-judge checklists.
Evaluation is via LLM-as-judge (see eval_writing.py, not yet implemented).

Usage:
  python -m coder.scripts.gen_writing --model dream_general \\
      --out outputs/base_tuteng/writing_dream_general.jsonl

  python -m coder.scripts.gen_writing --model llama31 \\
      --out outputs/base_tuteng/writing_llama31.jsonl
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
# Dataset loader
# ---------------------------------------------------------------------------

def load_wildbench_writing() -> List[Dict[str, Any]]:
    """WildBench v2 Creative Writing subset (146 items)."""
    ds = load_dataset("allenai/WildBench", "v2", split="test")
    items = []
    for row in ds:
        if row.get("primary_tag") != "Creative Writing":
            continue
        # Build a single prompt string from conversation_input turns.
        # Most items are single-turn; for multi-turn we concatenate as a transcript.
        turns = row["conversation_input"]
        if len(turns) == 1:
            prompt_text = turns[0]["content"]
        else:
            lines = []
            for t in turns:
                role = t.get("role", "user").capitalize()
                lines.append(f"{role}: {t['content']}")
            prompt_text = "\n\n".join(lines)

        items.append({
            "id": f"wildbench_writing/{row['id']}",
            "prompt_text": prompt_text,
            "conversation_input": turns,
            "checklist": row.get("checklist", []),
            "intent": row.get("intent", ""),
            "references": row.get("references", []),
        })
    return items


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

WRITING_SYSTEM = (
    "You are a skilled creative writer. "
    "Fulfill the following request thoughtfully and creatively."
)


def build_prompt(item: Dict[str, Any]) -> str:
    return f"{WRITING_SYSTEM}\n\n{item['prompt_text']}"


# ---------------------------------------------------------------------------
# Model factory (same pattern as gen_research.py)
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
    ap.add_argument("--out", required=True)

    ap.add_argument("--limit", type=int, default=0, help="Only run first N items. 0 = all.")
    ap.add_argument("--task_ids", type=str, default=None)
    ap.add_argument("--task_ids_file", type=str, default=None)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--resume", action="store_true")

    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="Higher temp for creative tasks (default: 0.7)")
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--num_samples", type=int, default=1)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--model_id", type=str, default=None)
    args = ap.parse_args()

    validate_shard_args(num_shards=args.num_shards, shard_idx=args.shard_idx)

    print("[data] loading WildBench Creative Writing ...")
    items = load_wildbench_writing()
    print(f"[data] {len(items)} creative writing items loaded")

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
        for item in tqdm(selected, desc=f"gen:{args.model}:writing"):
            prompt = build_prompt(item)
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
                    "prompt_text": item["prompt_text"],
                    "prompt": prompt,
                    "raw_completion": raw_gen,
                    "checklist": item["checklist"],
                    "intent": item["intent"],
                    "model": model.name,
                    "dataset": "wildbench_writing",
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
        "script": "gen_writing",
        "out": out_path,
        "model": model.name,
        "dataset": "wildbench_writing",
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

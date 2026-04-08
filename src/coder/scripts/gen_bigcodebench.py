import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from tqdm import tqdm

from coder.models import (
    ApiCoder,
    DeepSeekCoder,
    DiffuLLaMACoder,
    DreamCoder,
    LLaDACoder,
    Llama31Coder,
    MistralCoder,
    Qwen35Coder,
    QwenCoder,
    SeedCoder,
    SeedDiffCoder,
    StarCoder2Coder,
)
from coder.utils.schema import ModelRequest
from coder.utils.code_cleaning import build_prompt_scaffold_solution
from coder.utils.sharding import take_shard, validate_shard_args


def load_id_file(path: Optional[str]) -> Optional[set[str]]:
    if not path:
        return None
    ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if value:
                ids.add(value)
    return ids


def build_model(name: str, device: str, model_id: Optional[str] = None):
    name = name.lower()
    if name in ["dream", "dream_coder"]:
        return DreamCoder(model_id=model_id or "Dream-org/Dream-Coder-v0-Instruct-7B", device=device)
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


def load_bigcodebench_rows(subset: str, revision: str) -> List[Dict[str, Any]]:
    dataset_name = "bigcode/bigcodebench-hard" if subset == "hard" else "bigcode/bigcodebench"
    ds = load_dataset(dataset_name, split=revision)
    return list(ds)


def get_prompt(row: Dict[str, Any], split_mode: str) -> str:
    if split_mode == "instruct":
        return str(row.get("instruct_prompt", "")).strip()
    return str(row.get("complete_prompt", "")).strip()


def build_generation_prompt(prompt: str) -> str:
    prompt = (prompt or "").strip()
    instruction = (
        "Write a self-contained Python solution. "
        "Only output valid Python code. "
        "Do not include explanations, markdown fences, or surrounding text."
    )
    return f"{instruction}\n\n{prompt}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model_id", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split", choices=["instruct", "complete"], default="instruct")
    ap.add_argument("--subset", choices=["full", "hard"], default="full")
    ap.add_argument("--revision", default="v0.1.0_hf", help="HF split name, e.g. v0.1.0_hf")
    ap.add_argument("--task_ids_file", default=None, help="Optional newline-delimited task_id filter.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--max_new_tokens", type=int, default=1280)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--out", required=True, help="outputs/{model}_bigcodebench_{split}.jsonl")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    try:
        validate_shard_args(num_shards=args.num_shards, shard_idx=args.shard_idx)
    except ValueError as e:
        ap.error(str(e))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_model(args.model, device=args.device, model_id=args.model_id)
    rows = load_bigcodebench_rows(subset=args.subset, revision=args.revision)
    task_id_filter = load_id_file(args.task_ids_file)
    if task_id_filter:
        rows = [row for row in rows if str(row.get("task_id", "")).strip() in task_id_filter]

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(rows)
    rows = take_shard(rows, num_shards=args.num_shards, shard_idx=args.shard_idx)
    if args.limit is not None:
        rows = rows[:args.limit]

    done_task_ids = set()
    if args.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    tid = obj.get("task_id")
                    if isinstance(tid, str) and tid:
                        done_task_ids.add(tid)
                except Exception:
                    continue

    t_total0 = time.perf_counter()
    timing_generate_s: list[float] = []
    n_records_written = 0

    with out_path.open("a", encoding="utf-8") as fout:
        for row in tqdm(rows, desc=f"gen_bigcodebench({model.name})"):
            task_id = str(row.get("task_id", "")).strip()
            if not task_id:
                continue
            if task_id in done_task_ids:
                continue

            source_prompt = get_prompt(row, split_mode=args.split)
            if not source_prompt:
                continue
            prompt = build_generation_prompt(source_prompt)

            req = ModelRequest(
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
            )
            t0 = time.perf_counter()
            raw_gen = model.generate(req)
            t1 = time.perf_counter()
            completion = build_prompt_scaffold_solution(req.prompt, raw_gen)
            timing_generate_s.append(t1 - t0)

            rec = {
                "task_id": task_id,
                "benchmark": "bigcodebench",
                "split": args.split,
                "subset": args.subset,
                "revision": args.revision,
                "prompt": req.prompt,
                "source_prompt": source_prompt,
                "raw_generation": raw_gen,
                "raw_completion": completion,
                "solution": completion,
                "raw_solution": raw_gen,
                "model": model.name,
                "gen": {
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "seed": args.seed,
                    "timing": {
                        "generate_s": t1 - t0,
                        "total_s": t1 - t0,
                    },
                },
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_records_written += 1

    t_total1 = time.perf_counter()
    timing_path = str(out_path) + ".timing_summary.json"
    summary = {
        "script": "gen_bigcodebench",
        "out": str(out_path.resolve()),
        "model": model.name,
        "split": args.split,
        "subset": args.subset,
        "revision": args.revision,
        "num_shards": args.num_shards,
        "shard_idx": args.shard_idx,
        "n_records_written": n_records_written,
        "timing": {
            "total_s": t_total1 - t_total0,
            "generate_s_total": float(sum(timing_generate_s)),
            "generate_s_avg": (float(sum(timing_generate_s)) / len(timing_generate_s)) if timing_generate_s else None,
        },
    }
    Path(timing_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[timing] wrote {timing_path}")
    print(f"[samples] wrote {out_path}")


if __name__ == "__main__":
    main()

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional

from tqdm import tqdm
from datasets import load_dataset

from coder.utils.schema import ModelRequest
from coder.models import DreamCoder, DeepSeekCoder


def build_model(name: str, device: str, model_id: Optional[str] = None):
    name = name.lower()
    if name in ["dream", "dream_coder"]:
        return DreamCoder(
            model_id=model_id or "Dream-org/Dream-Coder-v0-Instruct-7B",
            device=device,
        )
    if name in ["deepseek", "deepseek_coder", "ds"]:
        return DeepSeekCoder(
            model_id=model_id or "deepseek-ai/deepseek-coder-6.7b-instruct",
            device=device,
        )
    raise ValueError(f"Unknown --model: {name}")


def iter_livebench_coding(
    split: str,
    limit: Optional[int],
    shuffle: bool,
    seed: int,
    task_filter: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    ds = load_dataset("livebench/coding", split=split)
    items = list(ds)

    if task_filter:
        task_filter_set = set(task_filter)
        items = [x for x in items if x.get("task") in task_filter_set]

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(items)

    if limit is not None:
        items = items[:limit]

    return items


def read_existing_qids(path: Path) -> set:
    if not path.exists():
        return set()
    qids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                qid = obj.get("question_id")
                if qid:
                    qids.add(qid)
            except Exception:
                continue
    return qids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="dream | deepseek")
    ap.add_argument("--model_id", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split", default="test", choices=["test"])
    ap.add_argument("--task", action="append", default=None, help="Filter by task name, can repeat.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=3407)

    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)

    ap.add_argument("--out", required=True, help="outputs/{model}_livebench.jsonl")
    ap.add_argument("--resume", action="store_true", help="Skip question_ids already in --out")

    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_model(args.model, device=args.device, model_id=args.model_id)

    questions = iter_livebench_coding(
        split=args.split,
        limit=args.limit,
        shuffle=args.shuffle,
        seed=args.seed,
        task_filter=args.task,
    )

    done_qids = read_existing_qids(out_path) if args.resume else set()

    with out_path.open("a", encoding="utf-8") as fout:
        for q in tqdm(questions, desc=f"gen_livebench({model.name})"):
            qid = q["question_id"]
            if qid in done_qids:
                continue

            prompt = q["turns"][0]  # LiveBench: one-turn prompt for coding dataset

            req = ModelRequest(
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
            )
            completion = model.generate(req)

            # 统一成你 evalplus 的 samples 风格：task_id / solution / model / gen
            row = {
                "task_id": f"LiveBench/{qid}",
                "question_id": qid,
                "solution": completion,
                "model": model.name,
                "gen": {
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "seed": args.seed,
                },
                "meta": {
                    "task": q.get("task"),
                    "question_title": q.get("question_title"),
                    "release_date": str(q.get("release_date", "")),
                    "livebench_release_date": str(q.get("livebench_release_date", "")),
                },
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()


if __name__ == "__main__":
    main()

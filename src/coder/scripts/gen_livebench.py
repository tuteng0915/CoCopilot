import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional

from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files

from coder.utils.schema import ModelRequest
from coder.models import (
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


def resolve_dataset_name(benchmark: str) -> str:
    if benchmark == "livebench-coding":
        return "livebench/coding"
    if benchmark == "livecodebench":
        return "livecodebench/code_generation_lite"
    raise ValueError(f"Unsupported benchmark: {benchmark}")


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
    if name in ["qwen", "qwen_coder"]:
        return QwenCoder(
            model_id=model_id or "Qwen/Qwen2.5-Coder-7B-Instruct",
            device=device,
        )
    if name in ["qwen35", "qwen35_coder", "qwen3.5"]:
        return Qwen35Coder(
            model_id=model_id or "Qwen/Qwen3.5-4B",
            device=device,
        )
    if name in ["llada", "llada_coder"]:
        return LLaDACoder(
            model_id=model_id or "GSAI-ML/LLaDA-8B-Instruct",
            device=device,
        )
    if name in ["starcoder2", "starcoder2_coder", "sc2"]:
        return StarCoder2Coder(
            model_id=model_id or "bigcode/starcoder2-7b",
            device=device,
        )
    if name in ["mistral", "mistral_coder"]:
        return MistralCoder(
            model_id=model_id or "mistralai/Mistral-7B-Instruct-v0.3",
            device=device,
        )
    if name in ["llama31", "llama31_coder", "llama3.1"]:
        return Llama31Coder(
            model_id=model_id or "meta-llama/Llama-3.1-8B-Instruct",
            device=device,
        )
    if name in ["diffullama", "diffullama_coder", "dflm"]:
        return DiffuLLaMACoder(
            model_id=model_id,
            device=device,
        )
    if name in ["seed-diffcoder", "seed_diffcoder", "seeddiffcoder"]:
        return SeedDiffCoder(
            model_id=model_id,
            device=device,
        )
    if name in ["seed-coder", "seed_coder", "seedcoder"]:
        return SeedCoder(
            model_id=model_id,
            device=device,
        )
    if name in ["api", "api_coder", "closed_api"]:
        return ApiCoder(
            model_id=model_id,
            device="api",
        )
    raise ValueError(f"Unknown --model: {name}")


def iter_livebench_coding(
    benchmark: str,
    split: str,
    limit: Optional[int],
    shuffle: bool,
    seed: int,
    task_filter: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if benchmark == "livecodebench":
        if split != "test":
            raise ValueError("livecodebench currently supports split=test only")
        repo_id = resolve_dataset_name(benchmark)
        files = list_repo_files(repo_id=repo_id, repo_type="dataset")
        jsonl_files = sorted([f for f in files if f.endswith(".jsonl") and f.startswith("test")])
        items: List[Dict[str, Any]] = []
        for filename in jsonl_files:
            local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename)
            with open(local_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    items.append(json.loads(line))
    else:
        ds = load_dataset(resolve_dataset_name(benchmark), split=split)
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


def get_question_id(row: Dict[str, Any], idx: int, benchmark: str) -> str:
    qid = row.get("question_id") or row.get("id") or row.get("task_id")
    if isinstance(qid, (int, float)):
        qid = str(qid)
    if isinstance(qid, str) and qid.strip():
        return qid.strip()
    return f"{benchmark}_{idx}"


def get_prompt(row: Dict[str, Any]) -> str:
    turns = row.get("turns")
    if isinstance(turns, list) and turns:
        first = turns[0]
        if isinstance(first, str):
            return first
    for key in ("prompt", "question_content", "question", "description"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def get_task_name(row: Dict[str, Any]) -> str:
    for key in ("task", "task_type", "type", "category"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # LiveCodeBench code_generation_lite rows don't carry explicit task type.
    # Default to LCB_generation so eval script can route scorer correctly.
    if row.get("public_test_cases") is not None or row.get("private_test_cases") is not None:
        return "LCB_generation"
    return ""


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
    ap.add_argument(
        "--benchmark",
        default="livebench-coding",
        choices=["livebench-coding", "livecodebench"],
        help="Dataset backend to generate from.",
    )
    ap.add_argument("--split", default="test")
    ap.add_argument("--task", action="append", default=None, help="Filter by task name, can repeat.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=3407)

    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples per question_id. Default 1 keeps old behavior. "
             "Note: eval_livebench.py uses last-write-wins per question_id.",
    )

    ap.add_argument("--out", required=True, help="outputs/{model}_livebench.jsonl")
    ap.add_argument("--resume", action="store_true", help="Skip question_ids already in --out")

    args = ap.parse_args()
    if args.num_samples < 1:
        ap.error("--num_samples must be >= 1")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_model(args.model, device=args.device, model_id=args.model_id)

    questions = iter_livebench_coding(
        benchmark=args.benchmark,
        split=args.split,
        limit=args.limit,
        shuffle=args.shuffle,
        seed=args.seed,
        task_filter=args.task,
    )

    done_qids = read_existing_qids(out_path) if args.resume else set()

    t_total0 = time.perf_counter()
    timing_generate_s: list[float] = []
    n_records_written = 0

    with out_path.open("a", encoding="utf-8") as fout:
        for idx, q in enumerate(tqdm(questions, desc=f"gen_{args.benchmark}({model.name})")):
            qid = get_question_id(q, idx=idx, benchmark=args.benchmark)
            if qid in done_qids:
                continue

            prompt = get_prompt(q)
            if not prompt:
                continue

            base_req = ModelRequest(
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
            )

            task_prefix = "LiveBench" if args.benchmark == "livebench-coding" else "LiveCodeBench"
            for sample_idx in range(args.num_samples):
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
                completion = model.generate(req)
                t1 = time.perf_counter()
                timing_generate_s.append(t1 - t0)

                row = {
                    "task_id": f"{task_prefix}/{qid}",
                    "question_id": qid,
                    "sample_id": sample_idx,
                    "benchmark": args.benchmark,
                    "prompt": prompt,
                    "raw_completion": completion,
                    # LiveBench 官方评测只看 solution 字段，这里保持向后兼容。
                    "solution": completion,
                    "model": model.name,
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
                    "meta": {
                        "task": get_task_name(q),
                        "question_title": q.get("question_title"),
                        "release_date": str(q.get("release_date", "")),
                        "livebench_release_date": str(q.get("livebench_release_date", "")),
                    },
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                n_records_written += 1

    t_total1 = time.perf_counter()
    timing_path = str(out_path) + ".timing_summary.json"
    summary = {
        "script": "gen_livebench",
        "out": str(out_path.resolve()),
        "model": model.name,
        "benchmark": args.benchmark,
        "num_samples": args.num_samples,
        "n_records_written": n_records_written,
        "timing": {
            "total_s": t_total1 - t_total0,
            "generate_s_total": float(sum(timing_generate_s)),
            "generate_s_avg": (float(sum(timing_generate_s)) / len(timing_generate_s)) if timing_generate_s else None,
        },
    }
    Path(timing_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[timing] wrote {timing_path}")


if __name__ == "__main__":
    main()

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files


def resolve_dataset_name(benchmark: str) -> str:
    if benchmark == "livebench-coding":
        return "livebench/coding"
    if benchmark == "livecodebench":
        return "livecodebench/code_generation_lite"
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def load_samples(samples_path: Path) -> Tuple[Dict[str, str], Dict[str, Any], str]:
    """
    Returns:
      qid2solution: question_id -> solution text
      meta: one representative meta dict (for printing/debug)
      model_name: samples file's model name (from first line)
    """
    qid2solution: Dict[str, str] = {}
    model_name = "unknown_model"
    meta_any: Dict[str, Any] = {}

    with samples_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("question_id")
            sol = obj.get("solution", "")
            if qid is None:
                continue
            qid2solution[qid] = sol  # last-write-wins
            model_name = obj.get("model", model_name)
            meta_any = obj.get("meta", meta_any)

    return qid2solution, meta_any, model_name


def infer_benchmark_from_samples(samples_path: Path) -> Optional[str]:
    with samples_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            b = obj.get("benchmark")
            if isinstance(b, str) and b.strip():
                return b.strip()
            task_id = obj.get("task_id")
            if isinstance(task_id, str):
                if task_id.startswith("LiveCodeBench/"):
                    return "livecodebench"
                if task_id.startswith("LiveBench/"):
                    return "livebench-coding"
            break
    return None


def get_question_id(row: Dict[str, Any], idx: int, benchmark: str) -> str:
    qid = row.get("question_id") or row.get("id") or row.get("task_id")
    if isinstance(qid, (int, float)):
        qid = str(qid)
    if isinstance(qid, str) and qid.strip():
        return qid.strip()
    return f"{benchmark}_{idx}"


def get_task_name(row: Dict[str, Any]) -> str:
    for key in ("task", "task_type", "type", "category"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    if row.get("public_test_cases") is not None or row.get("private_test_cases") is not None:
        return "LCB_generation"
    return ""


def load_questions(benchmark: str, split: str) -> List[Dict[str, Any]]:
    if benchmark == "livecodebench":
        if split != "test":
            raise ValueError("livecodebench currently supports split=test only")
        repo_id = resolve_dataset_name(benchmark)
        files = list_repo_files(repo_id=repo_id, repo_type="dataset")
        jsonl_files = sorted([f for f in files if f.endswith(".jsonl") and f.startswith("test")])
        rows: List[Dict[str, Any]] = []
        for filename in jsonl_files:
            local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename)
            with open(local_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    original = json.loads(line)
                    row = dict(original)
                    # LiveCodeBench official scorer expects this key.
                    row["original_json"] = original
                    rows.append(row)
        return rows

    ds = load_dataset(resolve_dataset_name(benchmark), split=split)
    rows = []
    for r in ds:
        row = dict(r)
        row["original_json"] = r
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="outputs/{model}_livebench.jsonl")
    ap.add_argument(
        "--benchmark",
        default=None,
        choices=["livebench-coding", "livecodebench"],
        help="Dataset backend to evaluate against. Default: infer from samples.",
    )
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--out_judgments", required=True, help="outputs/{model}_livebench_judgments.jsonl")
    ap.add_argument("--out_summary", required=True, help="outputs/{model}_livebench_summary.json")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    samples_path = Path(args.samples)
    out_j = Path(args.out_judgments)
    out_s = Path(args.out_summary)
    out_j.parent.mkdir(parents=True, exist_ok=True)
    out_s.parent.mkdir(parents=True, exist_ok=True)

    # LiveBench official scorer (installed from official repo)
    try:
        from livebench.process_results.coding.utils import (
            LCB_generation_process_results,
            code_generation_process_results,
        )
    except Exception as e:
        raise RuntimeError(
            "cannot find `livebench`。\n"
            "  pip install git+https://github.com/LiveBench/LiveBench.git\n"
        ) from e

    qid2solution, _, model_name = load_samples(samples_path)
    benchmark = args.benchmark or infer_benchmark_from_samples(samples_path) or "livebench-coding"

    questions = load_questions(benchmark=benchmark, split=args.split)

    if args.shuffle:
        import random
        rng = random.Random(args.seed)
        rng.shuffle(questions)
    if args.limit is not None:
        questions = questions[:args.limit]

    scores: List[int] = []
    per_task = defaultdict(list)

    with out_j.open("w", encoding="utf-8") as fout:
        for idx, q in enumerate(tqdm(questions, desc=f"eval_{benchmark}({model_name})")):
            qid = get_question_id(q, idx=idx, benchmark=benchmark)
            task = get_task_name(q)
            if qid not in qid2solution:
                row = {
                    "question_id": qid,
                    "benchmark": benchmark,
                    "task": task,
                    "model": model_name,
                    "score": None,
                    "status": "missing_answer",
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            ans = qid2solution[qid]

            try:
                if task in ("LCB_generation", "coding_completion"):
                    score = int(LCB_generation_process_results(q, ans, debug=args.debug))
                elif task in ("code_generation", "code_completion"):
                    score = int(code_generation_process_results(q, ans, debug=args.debug))
                elif task == "agentic_coding":
                    score = None
                else:
                    score = None
            except Exception as e:
                score = None
                if args.debug:
                    print(f"[ERROR] qid={qid} task={task}: {e}")

            row = {
                "question_id": qid,
                "benchmark": benchmark,
                "task": task,
                "model": model_name,
                "score": score,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

            if score is not None:
                scores.append(score)
                per_task[task].append(score)

    summary = {
        "model": model_name,
        "benchmark": benchmark,
        "n_total_questions": len(questions),
        "n_scored": len(scores),
        "accuracy": (sum(scores) / len(scores)) if scores else None,
        "by_task": {t: (sum(v) / len(v) if v else None) for t, v in per_task.items()},
    }

    out_s.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

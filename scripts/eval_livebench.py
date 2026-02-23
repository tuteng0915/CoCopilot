import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from tqdm import tqdm
from datasets import load_dataset


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="outputs/{model}_livebench.jsonl")
    ap.add_argument("--split", default="test", choices=["test"])
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

    ds = load_dataset("livebench/coding", split=args.split)
    questions = list(ds)

    if args.shuffle:
        import random
        rng = random.Random(args.seed)
        rng.shuffle(questions)
    if args.limit is not None:
        questions = questions[:args.limit]

    scores: List[int] = []
    per_task = defaultdict(list)

    with out_j.open("w", encoding="utf-8") as fout:
        for q in tqdm(questions, desc=f"eval_livebench({model_name})"):
            qid = q["question_id"]
            task = q.get("task", "")
            if qid not in qid2solution:
                row = {
                    "question_id": qid,
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
        "n_total_questions": len(questions),
        "n_scored": len(scores),
        "accuracy": (sum(scores) / len(scores)) if scores else None,
        "by_task": {t: (sum(v) / len(v) if v else None) for t, v in per_task.items()},
    }

    out_s.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

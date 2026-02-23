import argparse
import gzip
import json
import os
import random
from typing import Dict, Tuple, List

from tqdm import tqdm
from evalplus.data import get_human_eval_plus, get_mbpp_plus

from coder.utils.schema import ModelRequest, SampleRecord
from coder.models import DreamCoder, DeepSeekCoder, CoderModel
import re


def build_model(name: str, device: str, model_id: str | None) -> CoderModel:
    if name == "dream":
        return DreamCoder(model_id=model_id or "Dream-org/Dream-Coder-v0-Instruct-7B", device=device)
    if name == "deepseek":
        return DeepSeekCoder(model_id=model_id or "deepseek-ai/deepseek-coder-6.7b-instruct", device=device)
    raise ValueError(f"Unknown model: {name}")


def select_tasks(
    problems: Dict[str, Dict],
    limit: int,
    task_ids: List[str] | None,
    shuffle: bool,
    seed: int,
) -> List[Tuple[str, Dict]]:
    items = list(problems.items())
    items.sort(key=lambda x: x[0])  # deterministic order

    if task_ids:
        wanted = set(task_ids)
        items = [(tid, p) for tid, p in items if tid in wanted]

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(items)

    if limit > 0:
        items = items[:limit]

    return items


def write_override_dataset_gz(out_path: str, selected: List[Tuple[str, Dict]]):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for _, problem in selected:
            f.write(json.dumps(problem, ensure_ascii=False) + "\n")

def clean_model_completion(text: str, prompt: str | None = None) -> str:
    if not text:
        return ""

    s = text.strip()

    # Prefer fenced code block if present
    fence_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", s, flags=re.S | re.I)
    if fence_blocks:
        s = fence_blocks[-1].strip()

    s = s.replace("```python", "").replace("```", "").strip()

    # Remove exact prompt echo if present
    if prompt:
        p = prompt.strip()
        if p and s.startswith(p):
            s = s[len(p):].lstrip()

    # Remove common assistant-style preamble
    s = re.sub(r"^\s*(assistant|response)\s*:\s*", "", s, flags=re.I)

    # If we can find a code start, cut to it.
    # (Do NOT force this when the model returns body-only completion.)
    m = re.search(r"(?m)^(def|class|import|from|@)\s+", s)
    if m:
        s = s[m.start():].lstrip()

    return s.strip()


def build_evalplus_solution(prob: dict, gen: str) -> str:
    """
    EvalPlus accepts a full `solution` program.
    If gen already looks like complete code, use it directly.
    Otherwise, treat gen as completion and prepend prompt.
    """
    g = (gen or "").lstrip()
    prompt = prob["prompt"].rstrip()

    # If the model already returned full code, don't prepend the prompt again.
    if re.search(r"(?m)^(def|class|import|from)\s+", g):
        return g.rstrip()

    # Otherwise assume completion (e.g., body-only for HumanEval)
    return (prompt + "\n" + gen.lstrip()).rstrip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["dream", "deepseek"], required=True)
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)

    ap.add_argument("--out", required=True)
    ap.add_argument("--override_out", default=None, help="If set, also write override dataset (.jsonl.gz) for subset eval.")

    ap.add_argument("--limit", type=int, default=0, help="Only generate first N tasks after filtering/shuffle. 0 = all.")
    ap.add_argument("--task_ids", type=str, default=None, help="Comma-separated task_ids to run (overrides full set).")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle tasks before taking --limit.")

    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=3407)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--model_id", type=str, default=None, help="Override HuggingFace model id.")

    args = ap.parse_args()

    problems = get_human_eval_plus() if args.dataset == "humaneval" else get_mbpp_plus()
    task_ids = [x.strip() for x in args.task_ids.split(",")] if args.task_ids else None

    selected = select_tasks(
        problems=problems,
        limit=args.limit,
        task_ids=task_ids,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    # If user wants subset, strongly recommend writing override dataset so evalplus.evaluate won't assert.
    if (args.limit > 0 or task_ids) and args.override_out is None:
        base = os.path.splitext(os.path.basename(args.out))[0]
        args.override_out = f"outputs/{args.dataset}_override_{base}.jsonl.gz"

    if args.override_out:
        write_override_dataset_gz(args.override_out, selected)
        print(f"[override] wrote {args.override_out}")
        if args.dataset == "humaneval":
            print(f'  export HUMANEVAL_OVERRIDE_PATH="{os.path.abspath(args.override_out)}"')
        else:
            print(f'  export MBPP_OVERRIDE_PATH="{os.path.abspath(args.override_out)}"')

    model = build_model(args.model, device=args.device, model_id=args.model_id)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for task_id, prob in tqdm(selected, desc=f"gen:{args.model}:{args.dataset}"):
            prompt = (
                "Complete the following Python function. "
                "Only output valid Python code (no explanation).\n\n"
                f"{prob['prompt']}\n"
            )
            req = ModelRequest(
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
            )
            raw_gen = model.generate(req)
            gen = clean_model_completion(raw_gen, prompt=req.prompt)
            solution = build_evalplus_solution(prob, gen)


            rec = SampleRecord(
                task_id=task_id,
                solution=solution,
                model=model.name,
                gen={
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "seed": args.seed,
                },
            )
            f.write(json.dumps(rec.to_json(), ensure_ascii=False) + "\n")

    print(f"[samples] wrote {args.out}")


if __name__ == "__main__":
    main()

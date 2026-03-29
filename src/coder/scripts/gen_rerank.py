#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from typing import Dict, Any, List, Tuple

from tqdm import tqdm
from evalplus.data import get_human_eval_plus, get_mbpp_plus

from coder.utils.schema import ModelRequest
from coder.models import DeepSeekCoder, QwenCoder, CoderModel


def build_model(name: str, device: str, model_id: str | None) -> CoderModel:
    name = name.lower()
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
    raise ValueError(f"Unsupported rerank backbone: {name}")


def select_tasks(
    problems: Dict[str, Dict],
    limit: int,
    task_ids: List[str] | None,
    shuffle: bool,
    seed: int,
) -> List[Tuple[str, Dict]]:
    items = list(problems.items())
    items.sort(key=lambda x: x[0])

    if task_ids:
        wanted = set(task_ids)
        items = [(tid, p) for tid, p in items if tid in wanted]

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(items)

    if limit > 0:
        items = items[:limit]

    return items


def clean_model_completion(text: str, prompt: str | None = None) -> str:
    if not text:
        return ""

    s = text.strip()

    fence_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", s, flags=re.S | re.I)
    if fence_blocks:
        s = fence_blocks[-1].strip()

    s = s.replace("```python", "").replace("```", "").strip()

    if prompt:
        p = prompt.strip()
        if p and s.startswith(p):
            s = s[len(p) :].lstrip()

    s = re.sub(r"^\s*(assistant|response)\s*:\s*", "", s, flags=re.I)

    m = re.search(r"(?m)^(def|class|import|from|@)\s+", s)
    if m:
        s = s[m.start() :].lstrip()

    return s.strip()


def build_evalplus_solution(prob: dict, gen: str) -> str:
    g = (gen or "").lstrip()
    prompt = prob["prompt"].rstrip()
    if re.search(r"(?m)^(def|class|import|from)\s+", g):
        return g.rstrip()
    return (prompt + "\n" + gen.lstrip()).rstrip()


def score_candidate(code: str) -> float:
    """
    A lightweight heuristic scorer for reranking:
      - prefer syntactically-looking Python (def/import/class)
      - lightly penalize overly long or overly short generations
    """
    if not code:
        return -1e9

    score = 0.0
    if re.search(r"(?m)^(def|class)\s+", code):
        score += 1.0
    if "import " in code:
        score += 0.3

    n_lines = code.count("\n") + 1
    target = 30.0
    score -= 0.01 * abs(n_lines - target)

    return score


def main() -> None:
    ap = argparse.ArgumentParser(
        description="AR + reranking baseline: sample multiple candidates and keep the best one."
    )
    ap.add_argument(
        "--model",
        choices=["deepseek", "qwen"],
        required=True,
    )
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)

    ap.add_argument("--out", required=True)

    ap.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="Number of AR samples per task for reranking.",
    )

    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--task_ids", type=str, default=None)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=3407)

    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--model_id", type=str, default=None)

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

    model = build_model(args.model, device=args.device, model_id=args.model_id)

    t_total0 = time.perf_counter()
    timing_generate_s: list[float] = []
    timing_score_s: list[float] = []
    timing_select_s: list[float] = []
    n_records_written = 0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for task_id, prob in tqdm(selected, desc=f"gen_rerank:{args.model}:{args.dataset}"):
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

            candidates: List[str] = []
            cleaned: List[str] = []
            scores: List[float] = []

            for i in range(args.num_samples):
                # To get diverse samples, jitter the seed slightly per sample.
                if req.seed is not None:
                    req.seed = args.seed + i
                t0 = time.perf_counter()
                raw_gen = model.generate(req)
                t1 = time.perf_counter()
                gen = clean_model_completion(raw_gen, prompt=req.prompt)
                t2 = time.perf_counter()
                sc = score_candidate(gen)
                t3 = time.perf_counter()

                candidates.append(raw_gen)
                cleaned.append(gen)
                scores.append(sc)

                timing_generate_s.append(t1 - t0)
                timing_score_s.append(t3 - t2)

            t_sel0 = time.perf_counter()
            best_idx = int(max(range(len(scores)), key=lambda i: scores[i])) if scores else 0
            best_gen = cleaned[best_idx] if cleaned else ""
            solution = build_evalplus_solution(prob, best_gen)
            t_sel1 = time.perf_counter()
            timing_select_s.append(t_sel1 - t_sel0)

            rec: Dict[str, Any] = {
                "task_id":        task_id,
                "prompt":         req.prompt,
                "raw_completion": best_gen,
                "solution":       solution,
                "model":          f"rerank::{model.name}",
                "gen": {
                    "backbone_model":  model.name,
                    "num_samples":     args.num_samples,
                    "max_new_tokens":  args.max_new_tokens,
                    "temperature":     args.temperature,
                    "top_p":           args.top_p,
                    "seed_base":       args.seed,
                    "timing": {
                        "generate_s_total": float(sum(timing_generate_s[-args.num_samples:])),
                        "score_s_total": float(sum(timing_score_s[-args.num_samples:])),
                        "select_s": t_sel1 - t_sel0,
                        "total_s": float(sum(timing_generate_s[-args.num_samples:])) + float(sum(timing_score_s[-args.num_samples:])) + (t_sel1 - t_sel0),
                    },
                },
                # For analysis: keep all candidates + their heuristic scores.
                "rerank_candidates": [
                    {
                        "raw": candidates[i],
                        "clean": cleaned[i],
                        "score": scores[i],
                    }
                    for i in range(len(candidates))
                ],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_records_written += 1

    t_total1 = time.perf_counter()
    timing_path = args.out + ".timing_summary.json"
    summary = {
        "script": "gen_rerank",
        "out": os.path.abspath(args.out),
        "model": model.name,
        "dataset": args.dataset,
        "n_records_written": n_records_written,
        "timing": {
            "total_s": t_total1 - t_total0,
            "generate_s_total": float(sum(timing_generate_s)),
            "score_s_total": float(sum(timing_score_s)),
            "select_s_total": float(sum(timing_select_s)),
            "generate_s_avg": (float(sum(timing_generate_s)) / len(timing_generate_s)) if timing_generate_s else None,
        },
    }
    with open(timing_path, "w", encoding="utf-8") as tf:
        json.dump(summary, tf, ensure_ascii=False, indent=2)

    print(f"[samples] wrote {args.out}")
    print(f"[timing] wrote {timing_path}")


if __name__ == "__main__":
    main()


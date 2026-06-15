#!/usr/bin/env python3
"""
Generate SQL drafts from Spider dev with an autoregressive model.

Example:
  python -m coder.scripts.gen_sql_ar \
    --spider_dir outputs/sql_feasibility/spider \
    --out outputs/sql_feasibility/deepseek_spider_dev.jsonl \
    --model deepseek --n_samples 200
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

from tqdm import tqdm

from coder.models.codellama_coder import CodeLlamaCoder
from coder.models.deepseek_coder import DeepSeekCoder
from coder.models.llama31_coder import Llama31Coder
from coder.models.mistral_coder import MistralCoder
from coder.models.qwen_coder import QwenCoder
from coder.models.seed_coder import SeedCoder
from coder.models.starcoder2_coder import StarCoder2Coder
from coder.scripts.sql_eval import extract_sql, make_prompt, schema_to_text
from coder.utils.schema import ModelRequest


MODELS = {
    "deepseek": ("deepseek-ai/deepseek-coder-6.7b-instruct", DeepSeekCoder),
    "codellama": ("codellama/CodeLlama-7b-Instruct-hf", CodeLlamaCoder),
    "llama31": ("meta-llama/Llama-3.1-8B-Instruct", Llama31Coder),
    "mistral": ("mistralai/Mistral-7B-Instruct-v0.3", MistralCoder),
    "qwen": ("Qwen/Qwen2.5-Coder-7B-Instruct", QwenCoder),
    "seed-coder": ("ByteDance-Seed/Seed-Coder-8B-Instruct", SeedCoder),
    "starcoder2": ("bigcode/starcoder2-7b", StarCoder2Coder),
}


def _read_done_ids(out_path: pathlib.Path) -> set[str]:
    done: set[str] = set()
    if not out_path.exists():
        return done
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_id = rec.get("task_id")
            if task_id:
                done.add(str(task_id))
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spider_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", choices=sorted(MODELS), default="deepseek")
    ap.add_argument("--model_id", default=None, help="Override HuggingFace model id.")
    ap.add_argument("--n_samples", type=int, default=200, help="0 means all dev samples.")
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    spider = pathlib.Path(args.spider_dir)
    dev_data = json.loads((spider / "dev.json").read_text(encoding="utf-8"))
    tables = {
        t["db_id"]: t
        for t in json.loads((spider / "tables.json").read_text(encoding="utf-8"))
    }

    if args.start_idx < 0:
        ap.error("--start_idx must be >= 0")
    end_idx = len(dev_data) if args.n_samples == 0 else args.start_idx + args.n_samples
    selected = list(enumerate(dev_data))[args.start_idx:end_idx]

    model_id, model_cls = MODELS[args.model]
    model = model_cls(model_id=args.model_id or model_id, device=args.device)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = _read_done_ids(out_path) if args.resume else set()
    mode = "a" if args.resume else "w"

    timing_generate_s: list[float] = []
    n_records_written = 0
    t_total0 = time.perf_counter()

    with out_path.open(mode, encoding="utf-8") as fout:
        for dev_idx, item in tqdm(selected, desc=f"sql:{args.model}:spider"):
            task_id = f"spider/dev/{dev_idx}"
            if task_id in done_ids:
                continue

            db_id = item["db_id"]
            question = item["question"]
            gold_sql = item["query"]
            schema = schema_to_text(tables[db_id])
            prompt = make_prompt(schema, question)
            req = ModelRequest(
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
            )

            t0 = time.perf_counter()
            raw_output = model.generate(req)
            t1 = time.perf_counter()
            pred_sql = extract_sql(raw_output)
            timing_generate_s.append(t1 - t0)

            rec = {
                "task_id": task_id,
                "benchmark": "spider",
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "prompt": prompt,
                "pred_sql": pred_sql,
                "raw_completion": pred_sql,
                "raw_model_output": raw_output,
                "model": model.name,
                "gen": {
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "seed": args.seed,
                    "timing": {
                        "generate_s": t1 - t0,
                    },
                },
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_records_written += 1

    t_total1 = time.perf_counter()
    summary = {
        "script": "gen_sql_ar",
        "out": str(out_path.resolve()),
        "model": model.name,
        "n_records_written": n_records_written,
        "timing": {
            "total_s": t_total1 - t_total0,
            "generate_s_total": float(sum(timing_generate_s)),
            "generate_s_avg": (
                float(sum(timing_generate_s)) / len(timing_generate_s)
                if timing_generate_s
                else None
            ),
        },
    }
    timing_path = out_path.with_suffix(out_path.suffix + ".timing_summary.json")
    timing_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[samples] wrote {out_path}")
    print(f"[timing] wrote {timing_path}")


if __name__ == "__main__":
    main()

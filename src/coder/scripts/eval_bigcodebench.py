from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


def infer_split_subset(samples_path: Path) -> tuple[str, str]:
    with samples_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            split = obj.get("split")
            subset = obj.get("subset")
            if split in ("instruct", "complete") and subset in ("full", "hard"):
                return split, subset
            break
    return "instruct", "full"


def infer_model_name(samples_path: Path) -> str:
    with samples_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            model = obj.get("model")
            if isinstance(model, str) and model.strip():
                return model.strip()
            break
    return "unknown_model"


def load_pass_at_k(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="Samples jsonl path.")
    ap.add_argument("--split", choices=["instruct", "complete"], default=None)
    ap.add_argument("--subset", choices=["full", "hard"], default=None)
    ap.add_argument("--execution", choices=["local", "gradio", "e2b"], default="local")
    ap.add_argument("--parallel", type=int, default=None)
    ap.add_argument("--pass_k", default="1,5,10")
    ap.add_argument("--no_gt", action="store_true")
    ap.add_argument("--out_summary", required=True)
    ap.add_argument("--out_eval_results", default=None)
    args = ap.parse_args()

    samples = Path(args.samples).resolve()
    if not samples.exists():
        raise FileNotFoundError(f"samples not found: {samples}")

    inferred_split, inferred_subset = infer_split_subset(samples)
    split = args.split or inferred_split
    subset = args.subset or inferred_subset

    cmd = [
        "bigcodebench.evaluate",
        "--execution",
        args.execution,
        "--split",
        split,
        "--subset",
        subset,
        "--samples",
        str(samples),
        "--pass_k",
        args.pass_k,
        "--save_pass_rate",
    ]
    if args.parallel is not None:
        cmd += ["--parallel", str(args.parallel)]
    if args.no_gt:
        cmd += ["--no-gt"]

    print("[cmd]", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "找不到 `bigcodebench.evaluate` 命令。请先安装：\n"
            "  pip install bigcodebench --upgrade\n"
            "  pip install -I -r https://raw.githubusercontent.com/bigcode-project/bigcodebench/main/Requirements/requirements-eval.txt"
        ) from e

    eval_path = Path(str(samples).replace(".jsonl", "_eval_results.json"))
    passk_path = Path(str(samples).replace(".jsonl", "_pass_at_k.json"))

    if args.out_eval_results and eval_path.exists():
        dst = Path(args.out_eval_results)
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(eval_path, dst)
        eval_path = dst

    pass_at_k = load_pass_at_k(passk_path)
    summary = {
        "model": infer_model_name(samples),
        "benchmark": "bigcodebench",
        "split": split,
        "subset": subset,
        "samples": str(samples),
        "eval_results_file": str(eval_path) if eval_path.exists() else None,
        "pass_at_k_file": str(passk_path) if passk_path.exists() else None,
        "pass_at_k": pass_at_k,
    }

    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[summary] wrote {out_summary}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _safe_read_json(path: Path) -> Dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _extract_model(obj: Dict[str, Any], filename: str) -> str:
    m = obj.get("model")
    if isinstance(m, str) and m.strip():
        return m.strip()
    # fallback: <model>_xxx_summary.json
    return filename.split("_", 1)[0]


def _extract_track(obj: Dict[str, Any], filename: str) -> str:
    # unify different summary schemas
    dataset = obj.get("dataset")
    benchmark = obj.get("benchmark")
    split = obj.get("split")
    subset = obj.get("subset")

    if isinstance(benchmark, str) and benchmark.strip() == "bigcodebench":
        sp = split if isinstance(split, str) else "unknown"
        sb = subset if isinstance(subset, str) else "unknown"
        return f"bigcodebench/{sp}/{sb}"

    if isinstance(dataset, str) and dataset.strip():
        return dataset.strip()

    if isinstance(benchmark, str) and benchmark.strip():
        return benchmark.strip()

    # fallback by filename
    if "_humaneval_summary.json" in filename:
        return "humaneval"
    if "_mbpp_summary.json" in filename:
        return "mbpp"
    if "_livecodebench_summary.json" in filename:
        return "livecodebench"
    if "_livebench_summary.json" in filename:
        return "livebench-coding"
    return "unknown"


def _extract_metric(obj: Dict[str, Any]) -> Dict[str, Any]:
    # EvalPlus wrapper summary schema
    if "pass_at_k" in obj:
        pass_at_k = obj.get("pass_at_k")
        if isinstance(pass_at_k, dict):
            return {"pass_at_k": pass_at_k}
        return {"pass_at_k": None}

    # LiveBench summary schema
    if "accuracy" in obj:
        return {
            "accuracy": obj.get("accuracy"),
            "n_total_questions": obj.get("n_total_questions"),
            "n_scored": obj.get("n_scored"),
            "by_task": obj.get("by_task"),
        }

    # BigCodeBench summary schema
    if "pass_at_k_file" in obj or "eval_results_file" in obj:
        return {
            "pass_at_k": obj.get("pass_at_k"),
            "eval_results_file": obj.get("eval_results_file"),
            "pass_at_k_file": obj.get("pass_at_k_file"),
        }

    # generic fallback
    out = {}
    for k in ("score", "metric", "result"):
        if k in obj:
            out[k] = obj.get(k)
    return out


def merge_summaries(summary_files: List[Path]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        "models": defaultdict(dict),
        "sources": [],
        "skipped_files": [],
    }

    for p in sorted(summary_files):
        obj = _safe_read_json(p)
        if obj is None:
            merged["skipped_files"].append(str(p))
            continue

        model = _extract_model(obj, p.name)
        track = _extract_track(obj, p.name)
        metric = _extract_metric(obj)

        merged["models"][model][track] = {
            "summary_file": str(p),
            "metric": metric,
        }
        merged["sources"].append(str(p))

    # defaultdict -> dict
    merged["models"] = dict(merged["models"])
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--summary_dir",
        default="outputs/base_tuteng",
        help="Directory containing *_summary.json files.",
    )
    ap.add_argument(
        "--out",
        default="outputs/base_tuteng/all_summaries_merged.json",
        help="Merged JSON output path.",
    )
    args = ap.parse_args()

    summary_dir = Path(args.summary_dir).resolve()
    if not summary_dir.exists():
        raise FileNotFoundError(f"summary_dir not found: {summary_dir}")

    files = sorted(summary_dir.glob("*_summary.json"))
    merged = merge_summaries(files)
    merged["summary_dir"] = str(summary_dir)
    merged["n_summary_files"] = len(files)
    merged["n_models"] = len(merged["models"])

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[merged] wrote {out_path}")
    print(f"[merged] models={merged['n_models']} files={merged['n_summary_files']}")


if __name__ == "__main__":
    main()


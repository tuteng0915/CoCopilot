#!/usr/bin/env python3
"""Run locator calibration over all completed EvalPlus model-pair rows."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from coder.analysis.plot_calibration import process_dataset


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PAIRS = REPO_ROOT / "outputs" / "base_tuteng" / "model_pairs_all_t0.9.json"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "ablation_locator" / "matrix"
DEFAULT_PLOT_DIR = REPO_ROOT / "outputs" / "ablation_locator" / "matrix_plots"

AR_MODEL_IDS = {
    "DeepSeek-Coder 6.7B": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "Qwen2.5-Coder 7B": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Llama-3.1 8B": "meta-llama/Llama-3.1-8B-Instruct",
    "StarCoder2 7B": "bigcode/starcoder2-7b",
    "Mistral 7B": "mistralai/Mistral-7B-Instruct-v0.3",
    "CodeLlama 7B": "codellama/CodeLlama-7b-Instruct-hf",
    "Seed-Coder-Instruct 8B": "ByteDance-Seed/Seed-Coder-8B-Instruct",
}

DLLM_MODEL_IDS = {
    "Dream-Coder 7B": ("dream", "Dream-org/Dream-Coder-v0-Instruct-7B"),
    "LLaDA 8B": ("llada", "GSAI-ML/LLaDA-8B-Instruct"),
}


def _repo_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _select_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    data = _load_json(Path(args.model_pairs))
    rows: list[dict[str, Any]] = []
    datasets = set(args.datasets or [])
    ar_filters = set(args.ars or [])
    dllm_filters = set(args.dllms or [])
    slug_filters = set(args.slugs or [])

    for row in data.get("rows", []):
        if datasets and row.get("dataset") not in datasets:
            continue
        if ar_filters and row.get("ar_drafter") not in ar_filters and _slug_label(row.get("ar_drafter", "")) not in ar_filters:
            continue
        if dllm_filters and row.get("dllm_refiner") not in dllm_filters and _slug_label(row.get("dllm_refiner", "")) not in dllm_filters:
            continue
        if slug_filters and row.get("slug") not in slug_filters:
            continue
        if not row.get("collab_status", {}).get("is_fully_complete", False):
            continue
        rows.append(row)
    return rows


def _slug_label(text: str) -> str:
    return (
        text.lower()
        .replace("2.5", "25")
        .replace("3.1", "31")
        .replace("+", "plus")
        .replace("-", "_")
        .replace(" ", "_")
        .replace(".", "")
    )


def _out_path(row: dict[str, Any], out_dir: Path) -> Path:
    return out_dir / f"calibration_data_{row['slug']}.json"


def _command_for_row(row: dict[str, Any], args: argparse.Namespace, out_path: Path) -> list[str]:
    ar_label = row["ar_drafter"]
    dllm_label = row["dllm_refiner"]
    if ar_label not in AR_MODEL_IDS:
        raise KeyError(f"No AR model id mapping for {ar_label!r}")
    if dllm_label not in DLLM_MODEL_IDS:
        raise KeyError(f"No dLLM model id mapping for {dllm_label!r}")

    dllm_backend, dllm_model_id = DLLM_MODEL_IDS[dllm_label]
    locators = list(args.locators)
    skip_ar_labels = set(args.skip_ar_for_ars or [])
    if "ar" in locators and (ar_label in skip_ar_labels or _slug_label(ar_label) in skip_ar_labels):
        locators.remove("ar")

    paths = row.get("paths") or {}
    required = ["ar_input", "collab_output", "ar_eval_results", "collab_eval_results"]
    missing = [key for key in required if not paths.get(key) or not _repo_path(paths[key]).exists()]
    if missing:
        raise FileNotFoundError(f"{row['slug']} missing paths: {', '.join(missing)}")

    cmd = [
        sys.executable,
        "-m",
        "coder.analysis.locator_calibration_data",
        "--ar_input",
        str(_repo_path(paths["ar_input"])),
        "--collab_input",
        str(_repo_path(paths["collab_output"])),
        "--ar_eval",
        str(_repo_path(paths["ar_eval_results"])),
        "--collab_eval",
        str(_repo_path(paths["collab_eval_results"])),
        "--out",
        str(out_path),
        "--device",
        args.device,
        "--dllm_backend",
        dllm_backend,
        "--dllm_model_id",
        dllm_model_id,
        "--ar_model_id",
        AR_MODEL_IDS[ar_label],
        "--status_field",
        args.status_field,
        "--locators",
        *locators,
    ]
    if args.include_collab_fail:
        cmd.append("--include_collab_fail")
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    return cmd


def _auc(value: Any) -> float | None:
    return None if value is None else float(value)


def _summarize_row(row: dict[str, Any], data_path: Path, auc: dict[str, Any]) -> dict[str, Any]:
    data = _load_json(data_path)
    return {
        "slug": row.get("slug"),
        "dataset": row.get("dataset"),
        "ar_drafter": row.get("ar_drafter"),
        "dllm_refiner": row.get("dllm_refiner"),
        "n_eligible": data.get("n_eligible"),
        "n_changed_pairs": data.get("n_pairs"),
        "n_skipped_unchanged": data.get("n_skipped_unchanged"),
        "n_tokens": data.get("n_tokens"),
        "n_fault": data.get("n_fault"),
        "n_nonfault": data.get("n_nonfault"),
        "include_collab_fail": data.get("include_collab_fail"),
        "dllm_auc": _auc(auc.get(row.get("dllm_refiner_auc_key", "dLLM"))),
        "ar_auc": _auc(auc.get("AR logprob")),
        "bert_auc": _auc(auc.get("CodeBERT")),
        "random_auc": _auc(auc.get("Random")),
        "data_path": _display_path(data_path),
    }


def _dllm_auc_key(row: dict[str, Any]) -> str:
    if row.get("dllm_refiner") == "Dream-Coder 7B":
        return "dLLM (Dream-Coder)"
    if row.get("dllm_refiner") == "LLaDA 8B":
        return "dLLM (LLaDA)"
    return "dLLM"


def _normalize_auc_keys(row: dict[str, Any], auc: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(auc)
    dream_key = "dLLM (Dream-Coder)"
    if row.get("dllm_refiner") == "LLaDA 8B" and dream_key in normalized:
        normalized["dLLM (LLaDA)"] = normalized.pop(dream_key)
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_pairs", default=str(DEFAULT_MODEL_PAIRS))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--plot_dir", default=str(DEFAULT_PLOT_DIR))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda_visible_devices", default="")
    parser.add_argument("--status_field", default="plus_status", choices=["plus_status", "base_status"])
    parser.add_argument("--locators", nargs="+", choices=["dllm", "ar", "bert"], default=["dllm", "ar", "bert"])
    parser.add_argument("--datasets", nargs="+", choices=["humaneval", "mbpp"])
    parser.add_argument("--ars", nargs="+", help="Filter by full AR label or normalized slug label.")
    parser.add_argument("--dllms", nargs="+", help="Filter by full dLLM label or normalized slug label.")
    parser.add_argument("--slugs", nargs="+", help="Filter exact model-pair slugs.")
    parser.add_argument(
        "--skip_ar_for_ars",
        nargs="+",
        help="For these AR labels/slug labels, do not load the AR logprob locator.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--include_collab_fail", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    plot_dir = Path(args.plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    rows = _select_rows(args)
    if not rows:
        raise SystemExit("No completed model-pair rows selected.")

    print(f"Selected rows: {len(rows)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    summary_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, 1):
        out_path = _out_path(row, out_dir)
        print(f"\n[{idx}/{len(rows)}] {row['slug']} -> {_display_path(out_path)}")
        cmd = _command_for_row(row, args, out_path)
        print(" ".join(cmd))
        if args.dry_run:
            continue
        if out_path.exists() and not args.force:
            print(f"[skip] existing output: {out_path}")
        else:
            subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)

        if args.skip_plots:
            auc = {}
        else:
            auc = process_dataset(out_path, plot_dir, row["slug"], args.n_bins, args.random_seed)
            auc = _normalize_auc_keys(row, auc)
        summary = _summarize_row(row, out_path, auc)
        dllm_key = _dllm_auc_key(row)
        summary["dllm_auc"] = _auc(auc.get(dllm_key))
        summary_rows.append(summary)

    if not args.dry_run:
        summary = {
            "description": "Locator calibration matrix over completed EvalPlus model pairs.",
            "status_field": args.status_field,
            "include_collab_fail": args.include_collab_fail,
            "locators": args.locators,
            "n_rows": len(summary_rows),
            "rows": summary_rows,
        }
        summary_path = out_dir / "calibration_matrix_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSaved matrix summary: {_display_path(summary_path)}")


if __name__ == "__main__":
    main()

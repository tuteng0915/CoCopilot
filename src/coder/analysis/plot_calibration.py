#!/usr/bin/env python3
"""
Generate calibration plots and ROC/AUC summaries from locator calibration data.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LOCATORS = [
    ("dllm_confidence", "dLLM (Dream-Coder)", "steelblue"),
    ("ar_confidence", "AR logprob", "tomato"),
    ("bert_confidence", "CodeBERT", "seagreen"),
]


def _locator_keys(data: dict) -> list[tuple[str, str, str]]:
    dllm_label = "dLLM (Dream-Coder)"
    backend = str(data.get("dllm_backend") or "").lower()
    model_id = str(data.get("dllm_model_id") or "").lower()
    if backend == "llada" or "llada" in model_id:
        dllm_label = "dLLM (LLaDA)"
    return [
        ("dllm_confidence", dllm_label, "steelblue"),
        ("ar_confidence", "AR logprob", "tomato"),
        ("bert_confidence", "CodeBERT", "seagreen"),
    ]


def _values(records: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    labels: list[bool] = []
    scores: list[float] = []
    for rec in records:
        value = rec.get(key)
        if value is None:
            continue
        labels.append(bool(rec["is_fault"]))
        scores.append(float(value))
    return np.asarray(labels, dtype=bool), np.asarray(scores, dtype=float)


def _auc(labels: np.ndarray, confidence: np.ndarray) -> float | None:
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return None
    fpr, tpr = _roc_curve(labels, 1.0 - confidence)
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def _roc_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute ROC points for binary labels without requiring scikit-learn."""
    y = labels.astype(bool)
    positives = int(y.sum())
    negatives = int((~y).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("ROC requires both positive and negative labels.")

    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y[order]
    scores_sorted = scores[order]

    tpr = [0.0]
    fpr = [0.0]
    tp = 0
    fp = 0
    i = 0
    n = len(scores_sorted)
    while i < n:
        score = scores_sorted[i]
        j = i
        while j < n and scores_sorted[j] == score:
            if y_sorted[j]:
                tp += 1
            else:
                fp += 1
            j += 1
        tpr.append(tp / positives)
        fpr.append(fp / negatives)
        i = j

    if fpr[-1] != 1.0 or tpr[-1] != 1.0:
        fpr.append(1.0)
        tpr.append(1.0)
    return np.asarray(fpr, dtype=float), np.asarray(tpr, dtype=float)


def plot_calibration(data_path: Path, out_dir: Path, tag: str, n_bins: int) -> dict[str, float | None]:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    records = data.get("records", [])
    if not records:
        raise ValueError(f"No records in {data_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    auc_results: dict[str, float | None] = {}
    locators = _locator_keys(data)

    fig, axes = plt.subplots(
        1,
        len(locators),
        figsize=(16, 4.8),
        sharey=True,
        constrained_layout=True,
    )
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    for ax, (key, name, color) in zip(axes, locators):
        labels, confidence = _values(records, key)
        if len(labels) == 0:
            ax.set_title(f"{name}\n(no scores)", fontsize=12, fontweight="bold")
            auc_results[name] = None
            continue

        fault_fracs: list[float] = []
        bin_centers: list[float] = []
        bin_counts: list[int] = []
        for idx, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            if idx == n_bins - 1:
                mask = (confidence >= lo) & (confidence <= hi)
            else:
                mask = (confidence >= lo) & (confidence < hi)
            if not mask.any():
                continue
            fault_fracs.append(float(labels[mask].mean()))
            bin_centers.append(float((lo + hi) / 2.0))
            bin_counts.append(int(mask.sum()))

        auc = _auc(labels, confidence)
        auc_results[name] = auc
        ax.bar(
            bin_centers,
            fault_fracs,
            width=0.9 / n_bins,
            color=color,
            alpha=0.75,
            edgecolor="black",
            linewidth=0.4,
        )
        ax.set_xlabel("Confidence score", fontsize=11)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.0)
        if auc is not None:
            ax.text(
                0.04,
                0.9,
                f"AUC={auc:.3f}",
                fontsize=9,
                transform=ax.transAxes,
            )

        twin = ax.twinx()
        twin.plot(bin_centers, bin_counts, color="0.25", marker=".", linewidth=1.0, alpha=0.45)
        twin.set_ylim(0, max(bin_counts) * 1.25 if bin_counts else 1)
        twin.tick_params(axis="y", labelsize=8, colors="0.35")
        if ax is not axes[-1]:
            twin.set_yticklabels([])
        else:
            twin.set_ylabel("Token count", fontsize=9, color="0.35")

    axes[0].set_ylabel("Fraction of fault tokens", fontsize=11)
    fig.suptitle(
        f"Locator Calibration - {tag} "
        f"(pairs={data.get('n_pairs', '?')}, fault={data.get('n_fault', '?')})",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out_dir / f"calibration_{tag}.pdf", dpi=150)
    fig.savefig(out_dir / f"calibration_{tag}.png", dpi=150)
    plt.close(fig)
    return auc_results


def plot_roc(data_path: Path, out_dir: Path, tag: str, seed: int) -> dict[str, float | None]:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    records = data.get("records", [])
    out_dir.mkdir(parents=True, exist_ok=True)

    auc_results: dict[str, float | None] = {}
    fig, ax = plt.subplots(figsize=(6, 5))

    locators = _locator_keys(data)
    for key, name, color in locators:
        labels, confidence = _values(records, key)
        auc = _auc(labels, confidence)
        auc_results[name] = auc
        if auc is None:
            print(f"ROC skipped for {tag}/{name}: need both fault and non-fault labels.")
            continue
        fpr, tpr = _roc_curve(labels, 1.0 - confidence)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=2.0)

    labels_all = np.asarray([bool(rec["is_fault"]) for rec in records], dtype=bool)
    if len(np.unique(labels_all)) >= 2:
        rng = np.random.default_rng(seed)
        random_confidence = rng.uniform(0.0, 1.0, size=len(labels_all))
        random_auc = _auc(labels_all, random_confidence)
        auc_results["Random"] = random_auc
        if random_auc is not None:
            fpr, tpr = _roc_curve(labels_all, 1.0 - random_confidence)
            ax.plot(
                fpr,
                tpr,
                label=f"Random scores (AUC={random_auc:.3f})",
                color="0.45",
                linewidth=1.2,
                alpha=0.8,
            )
    else:
        auc_results["Random"] = None

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Chance (AUC=0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"ROC Curves - {tag}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / f"roc_{tag}.pdf", dpi=150)
    fig.savefig(out_dir / f"roc_{tag}.png", dpi=150)
    plt.close(fig)
    return auc_results


def process_dataset(data_path: Path, out_dir: Path, tag: str, n_bins: int, seed: int) -> dict[str, float | None]:
    calibration_auc = plot_calibration(data_path, out_dir, tag, n_bins)
    roc_auc = plot_roc(data_path, out_dir, tag, seed)
    merged = dict(calibration_auc)
    merged.update(roc_auc)
    print(f"\n=== AUC Summary [{tag}] ===")
    for name, value in merged.items():
        text = "n/a" if value is None else f"{value:.4f}"
        print(f"  {name}: {text}")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_humaneval")
    parser.add_argument("--data_mbpp")
    parser.add_argument("--data", action="append", default=[], help="Additional DATA_PATH:TAG input.")
    parser.add_argument("--out_dir", default="outputs/ablation_locator/plots")
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets: list[tuple[Path, str]] = []
    if args.data_humaneval:
        datasets.append((Path(args.data_humaneval), "humaneval"))
    if args.data_mbpp:
        datasets.append((Path(args.data_mbpp), "mbpp"))
    for item in args.data:
        try:
            path_text, tag = item.rsplit(":", 1)
        except ValueError as exc:
            raise SystemExit("--data entries must use DATA_PATH:TAG") from exc
        datasets.append((Path(path_text), tag))
    if not datasets:
        raise SystemExit("Provide --data_humaneval/--data_mbpp or at least one --data DATA_PATH:TAG.")

    out_dir = Path(args.out_dir)
    summary = {
        tag: process_dataset(path, out_dir, tag, args.n_bins, args.random_seed)
        for path, tag in datasets
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "auc_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved plots to: {out_dir}")
    print(f"AUC summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

"""Evaluate FRAMES / HotpotQA JSONL outputs with EM and token F1.

The scoring follows the common SQuAD-style normalization:
lowercase, remove punctuation, remove English articles, then collapse spaces.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def infer_dataset(records: list[dict[str, Any]], input_path: str) -> str:
    if records:
        dataset = records[0].get("dataset")
        if isinstance(dataset, str) and dataset:
            return dataset
        rec_id = str(records[0].get("id", ""))
        if rec_id.startswith("frames/"):
            return "frames"
        if rec_id.startswith("hotpotqa/"):
            return "hotpotqa"
    name = Path(input_path).name.lower()
    if "frames" in name:
        return "frames"
    if "hotpotqa" in name:
        return "hotpotqa"
    return "unknown"


def extract_answer(raw_completion: str) -> str:
    """Return text after the last Answer: marker, or the first 50 words."""
    text = raw_completion or ""
    idx = text.lower().rfind("answer:")
    if idx >= 0:
        tail = text[idx + len("answer:") :].strip()
        if tail:
            for line in tail.splitlines():
                line = line.strip()
                if line:
                    return line

    words = text.strip().split()
    return " ".join(words[:50]).strip()


def normalize_answer(text: Any) -> str:
    """Lower text and remove punctuation, articles, and extra whitespace."""
    s = str(text).lower()
    exclude = set(string.punctuation)
    s = "".join(ch for ch in s if ch not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def exact_match_score(prediction: str, ground_truth: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_candidates(answer_ref: Any) -> list[str]:
    if isinstance(answer_ref, list):
        values: list[str] = []
        for item in answer_ref:
            values.extend(answer_candidates(item))
        return values or [""]
    if isinstance(answer_ref, dict):
        values = []
        for key in ("answer", "text", "value"):
            if key in answer_ref:
                values.extend(answer_candidates(answer_ref[key]))
        return values or [json.dumps(answer_ref, ensure_ascii=False)]
    if answer_ref is None:
        return [""]
    return [str(answer_ref)]


def metric_max_over_ground_truths(
    prediction: str,
    ground_truths: list[str],
) -> tuple[int, float]:
    if not ground_truths:
        ground_truths = [""]
    em = max(exact_match_score(prediction, gt) for gt in ground_truths)
    f1 = max(token_f1_score(prediction, gt) for gt in ground_truths)
    return em, f1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL file from gen_research.py or gen_remask.py")
    ap.add_argument("--out", required=True, help="Path to write evaluation JSON")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    records = read_jsonl(args.input)
    dataset = infer_dataset(records, args.input)

    per_item: list[dict[str, Any]] = []
    total_em = 0
    total_f1 = 0.0

    for idx, rec in enumerate(records):
        pred = extract_answer(rec.get("raw_completion", ""))
        refs = answer_candidates(rec.get("answer_ref", rec.get("answer", "")))
        em, f1 = metric_max_over_ground_truths(pred, refs)
        total_em += em
        total_f1 += f1
        per_item.append({
            "id": rec.get("id", f"{dataset}/{idx}"),
            "em": em,
            "f1": f1,
        })

    n_total = len(records)
    summary = {
        "dataset": dataset,
        "model": records[0].get("model", "unknown") if records else "unknown",
        "n_total": n_total,
        "exact_match": (total_em / n_total) if n_total else 0.0,
        "token_f1": (total_f1 / n_total) if n_total else 0.0,
        "per_item": per_item,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[eval] dataset={dataset} records={n_total}")
    print(f"[eval] EM={summary['exact_match']:.4f} F1={summary['token_f1']:.4f}")
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()

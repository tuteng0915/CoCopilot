"""Evaluate rewriting outputs (ASSET / CoEdIT) with SARI and BLEU-4."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


PREFIX_RE = re.compile(
    r"^\s*(?:simplified|rewritten|rewrite|output|answer|corrected|paraphrase)\s*:\s*",
    re.IGNORECASE,
)


def read_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


def ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def counter_intersection(a: Counter, b: Counter) -> Counter:
    return a & b


def counter_subtract_floor(a: Counter, b: Counter) -> Counter:
    out = a.copy()
    out.subtract(b)
    return Counter({k: v for k, v in out.items() if v > 0})


def counter_total(c: Counter) -> int:
    return int(sum(c.values()))


def f1(precision: float, recall: float) -> float:
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def union_reference_ngrams(refs: list[str], n: int) -> Counter:
    union: Counter = Counter()
    for ref in refs:
        ref_counts = ngrams(tokenize(ref), n)
        for gram, count in ref_counts.items():
            union[gram] = max(union[gram], count)
    return union


def compute_sari(src: str, pred: str, refs: list[str]) -> float:
    """Compute sentence-level SARI on a 0-100 scale."""
    if not refs:
        refs = [""]
    src_tokens = tokenize(src)
    pred_tokens = tokenize(pred)
    scores: list[float] = []

    for n in range(1, 5):
        src_ng = ngrams(src_tokens, n)
        pred_ng = ngrams(pred_tokens, n)
        ref_ng = union_reference_ngrams(refs, n)

        add_sys = counter_subtract_floor(pred_ng, src_ng)
        add_ref = counter_subtract_floor(ref_ng, src_ng)
        add_good = counter_intersection(add_sys, add_ref)
        add_p = counter_total(add_good) / counter_total(add_sys) if counter_total(add_sys) else 0.0
        add_r = counter_total(add_good) / counter_total(add_ref) if counter_total(add_ref) else 0.0
        add_score = f1(add_p, add_r)

        keep_sys = counter_intersection(pred_ng, src_ng)
        keep_ref = counter_intersection(ref_ng, src_ng)
        keep_good = counter_intersection(keep_sys, keep_ref)
        keep_p = counter_total(keep_good) / counter_total(keep_sys) if counter_total(keep_sys) else 0.0
        keep_r = counter_total(keep_good) / counter_total(keep_ref) if counter_total(keep_ref) else 0.0
        keep_score = f1(keep_p, keep_r)

        del_sys = counter_subtract_floor(src_ng, pred_ng)
        del_good = counter_subtract_floor(del_sys, ref_ng)
        delete_score = counter_total(del_good) / counter_total(del_sys) if counter_total(del_sys) else 0.0

        scores.append((add_score + keep_score + delete_score) / 3.0)

    return 100.0 * (sum(scores) / len(scores))


def compute_bleu4(predictions: list[str], references: list[list[str]]) -> float:
    """Compute average sentence BLEU-4 on a 0-100 scale."""
    if not predictions:
        return 0.0
    smoothie = SmoothingFunction().method1
    scores = []
    for pred, refs in zip(predictions, references):
        ref_tokens = [tokenize(ref) for ref in (refs or [""])]
        pred_tokens = tokenize(pred)
        scores.append(sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie))
    return 100.0 * (sum(scores) / len(scores))


def extract_prediction(raw_completion: str) -> str:
    text = raw_completion or ""
    line = ""
    for candidate in text.splitlines():
        candidate = candidate.strip()
        if candidate:
            line = candidate
            break
    if not line:
        line = text.strip()
    line = PREFIX_RE.sub("", line).strip()
    words = line.split()
    return " ".join(words[:150]).strip()


def extract_src_text(original: str) -> str:
    """Remove CoEdIT instruction prefixes like 'Fix grammaticality: ...'."""
    text = (original or "").strip()
    if ":" not in text:
        return text
    prefix, rest = text.split(":", 1)
    prefix_norm = prefix.strip().lower()
    instruction_markers = (
        "fix", "rewrite", "paraphrase", "neutralize", "make", "change",
        "simplify", "correct", "grammar", "grammaticality",
    )
    if any(marker in prefix_norm for marker in instruction_markers):
        return rest.strip()
    return text


def infer_dataset(records: list[dict[str, Any]], input_path: str) -> str:
    if records:
        dataset = records[0].get("dataset")
        if isinstance(dataset, str) and dataset:
            return dataset
        rec_id = str(records[0].get("id", ""))
        if rec_id.startswith("asset/"):
            return "asset"
        if rec_id.startswith("coedit/"):
            return "coedit"
    name = Path(input_path).name.lower()
    if "asset" in name:
        return "asset"
    if "coedit" in name:
        return "coedit"
    return "rewrite"


def references_for(rec: dict[str, Any]) -> list[str]:
    refs = rec.get("references")
    if isinstance(refs, list) and refs:
        return [str(ref) for ref in refs]
    ref = rec.get("answer_ref", "")
    if isinstance(ref, list):
        return [str(x) for x in ref]
    return [str(ref)]


def summarize(items: list[dict[str, Any]]) -> dict[str, float | int]:
    if not items:
        return {"n": 0, "sari": 0.0, "bleu4": 0.0}
    return {
        "n": len(items),
        "sari": sum(float(item["sari"]) for item in items) / len(items),
        "bleu4": sum(float(item["bleu4"]) for item in items) / len(items),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL file from gen_rewrite.py or gen_remask.py")
    ap.add_argument("--out", required=True, help="Path to write evaluation JSON")
    ap.add_argument("--by_task", action="store_true", help="Report metrics grouped by task")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    records = read_jsonl(args.input)
    dataset = infer_dataset(records, args.input)

    per_item: list[dict[str, Any]] = []
    by_task_items: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for idx, rec in enumerate(records):
        refs = references_for(rec)
        src = extract_src_text(str(rec.get("original", "")))
        pred = extract_prediction(str(rec.get("raw_completion", "")))
        sari = compute_sari(src, pred, refs)
        bleu4 = compute_bleu4([pred], [refs])
        task = str(rec.get("task") or ("simplification" if dataset == "asset" else "rewrite"))
        item = {
            "id": rec.get("id", f"{dataset}/{idx}"),
            "sari": sari,
            "bleu4": bleu4,
            "prediction": pred,
            "task": task,
        }
        per_item.append(item)
        by_task_items[task].append(item)

    overall = summarize(per_item)
    summary: dict[str, Any] = {
        "dataset": dataset,
        "model": records[0].get("model", "unknown") if records else "unknown",
        "n_total": len(records),
        "sari": overall["sari"],
        "bleu4": overall["bleu4"],
        "per_item": per_item,
    }
    if args.by_task or dataset == "asset":
        summary["by_task"] = {
            task: summarize(items)
            for task, items in sorted(by_task_items.items())
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[eval] dataset={dataset} records={len(records)}")
    print(f"[eval] SARI={summary['sari']:.2f} BLEU4={summary['bleu4']:.2f}")
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()

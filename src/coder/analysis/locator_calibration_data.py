#!/usr/bin/env python3
"""
Collect per-token confidence and fault labels for locator calibration analysis.

The primary inclusion criterion is:
  - AR draft failed evaluation
  - dLLM-remasked/collab completion passed evaluation

Unlike locator_scoring.py, this script does not restrict pairs to tiny
"surgical" diffs.  It emits one JSON record per reference-token span with
confidence from each requested locator and a boolean changed-token label.
"""
from __future__ import annotations

import argparse
import difflib
import json
import inspect
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from coder.locators import align_confidence_to_spans, get_token_char_spans


@dataclass
class TokenRecord:
    task_id: str
    token_idx: int
    char_start: int
    char_end: int
    is_fault: bool
    dllm_confidence: float | None
    ar_confidence: float | None
    bert_confidence: float | None


def _read_jsonl(path: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            task_id = rec.get("task_id")
            if task_id:
                records[str(task_id)] = rec
    return records


def _status_from_entry(entry: Any, status_field: str) -> str | None:
    if isinstance(entry, list):
        entry = entry[0] if entry else None
    if not isinstance(entry, dict):
        return None
    if "eval" in entry and isinstance(entry["eval"], dict):
        return _status_from_entry(entry["eval"], status_field)
    value = entry.get(status_field)
    if value is None and status_field != "base_status":
        value = entry.get("base_status")
    if value is None and status_field != "plus_status":
        value = entry.get("plus_status")
    if value is None:
        value = entry.get("status")
    return str(value).lower() if value is not None else None


def _load_pass_map(path: str, status_field: str) -> dict[str, bool]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    eval_data = data.get("eval", data) if isinstance(data, dict) else {}
    out: dict[str, bool] = {}
    if not isinstance(eval_data, dict):
        return out
    for task_id, entry in eval_data.items():
        status = _status_from_entry(entry, status_field)
        if status is not None:
            out[str(task_id)] = status == "pass"
    return out


def _completion_text(rec: dict[str, Any], prefer_draft: bool = False) -> str:
    keys = (
        ("draft_completion", "raw_completion", "solution")
        if prefer_draft
        else ("raw_completion", "solution", "draft_completion")
    )
    for key in keys:
        value = rec.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def get_fault_char_set(draft: str, corrected: str) -> set[int]:
    """Return draft character indices whose spans differ from corrected text."""
    matcher = difflib.SequenceMatcher(None, draft, corrected, autojunk=False)
    fault_chars: set[int] = set()
    for op, a0, a1, _b0, _b1 in matcher.get_opcodes():
        if op != "equal":
            fault_chars.update(range(a0, a1))
    return fault_chars


def _infer_dllm_backend(backend: str, model_id: str) -> str:
    if backend != "auto":
        return backend
    lower = model_id.lower()
    if "llada" in lower:
        return "llada"
    return "dream"


def _score_dllm(
    dllm_model: Any,
    backend: str,
    prompt: str,
    draft: str,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    messages = [{"role": "user", "content": prompt}]
    attention_mask = None
    if backend == "llada":
        prompt_text = dllm_model.tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt_enc = dllm_model.tok(
            prompt_text,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        prompt_ids = prompt_enc["input_ids"].to(dllm_model.device)
        attention_mask = prompt_enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(dllm_model.device)
    else:
        prompt_enc = dllm_model.tok.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        prompt_ids = prompt_enc.input_ids.to(dllm_model.device)

    comp_ids = dllm_model.tok(
        draft,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(dllm_model.device)
    score_sig = inspect.signature(dllm_model.score_tokens)
    if "attention_mask" in score_sig.parameters:
        confidence_t = dllm_model.score_tokens(prompt_ids, comp_ids, attention_mask)
    else:
        confidence_t = dllm_model.score_tokens(prompt_ids, comp_ids)
    confidence = confidence_t.float().cpu().numpy()
    spans = get_token_char_spans(dllm_model.tok, draft)
    return confidence.astype(np.float32), spans


def _aligned_at(
    confidence: np.ndarray | None,
    src_spans: list[tuple[int, int]] | None,
    ref_spans: list[tuple[int, int]],
    idx: int,
) -> float | None:
    if confidence is None or src_spans is None:
        return None
    if src_spans is ref_spans and idx < len(confidence):
        return float(confidence[idx])
    aligned = align_confidence_to_spans(confidence, src_spans, [ref_spans[idx]])
    return float(aligned[0]) if len(aligned) else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ar_input", required=True)
    parser.add_argument("--collab_input", required=True)
    parser.add_argument("--ar_eval", required=True)
    parser.add_argument("--collab_eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--locators",
        nargs="+",
        choices=["dllm", "ar", "bert"],
        default=["dllm", "ar", "bert"],
    )
    parser.add_argument("--dllm_model_id", default="Dream-org/Dream-Coder-v0-Instruct-7B")
    parser.add_argument("--dllm_backend", default="auto", choices=["auto", "dream", "llada"])
    parser.add_argument("--ar_model_id", default="deepseek-ai/deepseek-coder-6.7b-instruct")
    parser.add_argument("--bert_model_id", default="microsoft/codebert-base-mlm")
    parser.add_argument("--status_field", default="plus_status", choices=["plus_status", "base_status"])
    parser.add_argument("--include_collab_fail", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Limit eligible pairs after filtering; 0 means all.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ar_pass = _load_pass_map(args.ar_eval, args.status_field)
    collab_pass = _load_pass_map(args.collab_eval, args.status_field)
    ar_records = _read_jsonl(args.ar_input)
    collab_records = _read_jsonl(args.collab_input)

    eligible = [
        task_id
        for task_id in sorted(ar_records)
        if task_id in collab_records
        and not ar_pass.get(task_id, True)
        and (args.include_collab_fail or collab_pass.get(task_id, False))
    ]
    if args.limit:
        eligible = eligible[: args.limit]

    print(f"Eligible pairs: {len(eligible)}")
    if not eligible:
        raise SystemExit("No eligible pairs found; check eval paths/status_field or use --include_collab_fail.")

    dllm_model = ar_model = bert_model = None
    dllm_backend = _infer_dllm_backend(args.dllm_backend, args.dllm_model_id)
    if "dllm" in args.locators:
        if dllm_backend == "llada":
            from coder.models.llada_coder import LLaDACoder

            print(f"[model] loading dLLM/LLaDA: {args.dllm_model_id}")
            dllm_model = LLaDACoder(model_id=args.dllm_model_id, device=args.device)
        else:
            from coder.models.dream_coder import DreamCoder

            print(f"[model] loading dLLM/Dream: {args.dllm_model_id}")
            dllm_model = DreamCoder(model_id=args.dllm_model_id, device=args.device)
    if "ar" in args.locators:
        from coder.locators.ar_locator import ARLocator

        print(f"[model] loading AR locator: {args.ar_model_id}")
        ar_model = ARLocator(model_id=args.ar_model_id, device=args.device)
    if "bert" in args.locators:
        from coder.locators.bert_locator import BERTLocator

        print(f"[model] loading BERT locator: {args.bert_model_id}")
        bert_model = BERTLocator(model_id=args.bert_model_id, device=args.device)

    records: list[TokenRecord] = []
    used_pairs = 0
    skipped_unchanged = 0

    for pair_idx, task_id in enumerate(eligible, 1):
        ar_rec = ar_records[task_id]
        collab_rec = collab_records[task_id]
        draft = _completion_text(collab_rec, prefer_draft=True) or _completion_text(ar_rec)
        corrected = _completion_text(collab_rec)
        prompt = str(collab_rec.get("prompt") or ar_rec.get("prompt") or "")

        if not draft or not corrected or draft == corrected:
            skipped_unchanged += 1
            continue

        print(f"[{pair_idx}/{len(eligible)}] {task_id}")
        fault_chars = get_fault_char_set(draft, corrected)

        dllm_conf = ar_conf = bert_conf = None
        dllm_spans = ar_spans = bert_spans = None

        if dllm_model is not None:
            dllm_conf, dllm_spans = _score_dllm(dllm_model, dllm_backend, prompt, draft)
        if ar_model is not None:
            ar_conf, ar_spans = ar_model.score(prompt, draft)
        if bert_model is not None:
            bert_conf, bert_spans = bert_model.score(prompt, draft)

        ref_spans = dllm_spans or ar_spans or bert_spans
        if not ref_spans:
            continue

        used_pairs += 1
        for token_idx, (char_start, char_end) in enumerate(ref_spans):
            is_fault = any(pos in fault_chars for pos in range(char_start, char_end))
            records.append(
                TokenRecord(
                    task_id=task_id,
                    token_idx=token_idx,
                    char_start=char_start,
                    char_end=char_end,
                    is_fault=is_fault,
                    dllm_confidence=_aligned_at(dllm_conf, dllm_spans, ref_spans, token_idx),
                    ar_confidence=_aligned_at(ar_conf, ar_spans, ref_spans, token_idx),
                    bert_confidence=_aligned_at(bert_conf, bert_spans, ref_spans, token_idx),
                )
            )

    n_fault = sum(record.is_fault for record in records)
    out = {
        "description": "Per-token locator confidence and changed-token labels for calibration/ROC analysis.",
        "status_field": args.status_field,
        "include_collab_fail": args.include_collab_fail,
        "dllm_backend": dllm_backend,
        "dllm_model_id": args.dllm_model_id,
        "ar_model_id": args.ar_model_id,
        "n_eligible": len(eligible),
        "n_pairs": used_pairs,
        "n_skipped_unchanged": skipped_unchanged,
        "n_tokens": len(records),
        "n_fault": n_fault,
        "n_nonfault": len(records) - n_fault,
        "records": [asdict(record) for record in records],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Total token records: {len(records)}")
    print(f"Fault tokens: {n_fault}")
    print(f"Non-fault tokens: {len(records) - n_fault}")
    print(f"Skipped unchanged pairs: {skipped_unchanged}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compute fault-detection ratio for SQL drafts.

For AR-failed Spider examples, this script aligns generated SQL against gold
SQL with a character diff, scores generated tokens with a locator, and reports:

  mean_conf_nonfault / mean_conf_fault

Values well above 1.0 mean the locator assigns lower confidence to fault
positions than to the rest of the SQL draft.
"""
from __future__ import annotations

import argparse
import difflib
import json
import pathlib
import sys
from typing import Protocol

import numpy as np
import torch

from coder.locators import ARLocator, BERTLocator, get_token_char_spans
from coder.models import DreamCoder, LLaDACoder
from coder.scripts.sql_eval import extract_sql


class LocatorLike(Protocol):
    def score(self, prompt_text: str, draft_text: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
        ...


class DreamCoderLocator:
    """Use DreamCoder's production single-pass token confidence as a locator."""

    def __init__(self, model_id: str, device: str):
        self.model_id = model_id
        self.device = device
        self.model = DreamCoder(model_id=model_id, device=device)

    @torch.inference_mode()
    def score(self, prompt_text: str, draft_text: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
        messages = [{"role": "user", "content": prompt_text}]
        prompt_enc = self.model.tok.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        prompt_ids = prompt_enc.input_ids.to(self.device)
        comp_ids = self.model.tok(
            draft_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(self.device)
        if comp_ids.shape[1] == 0:
            return np.array([], dtype=np.float32), []
        full_ids = torch.cat([prompt_ids, comp_ids], dim=1)
        logits = self.model.model(full_ids).logits
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
        comp_logits = logits[0, prompt_ids.shape[1]:, :].float()
        probs = torch.softmax(comp_logits, dim=-1)
        confidence = probs[torch.arange(comp_ids.shape[1], device=self.device), comp_ids[0]]
        spans = get_token_char_spans(self.model.tok, draft_text)
        return confidence.cpu().numpy().astype(np.float32), spans


class LLaDALocator:
    """Use LLaDA's single-pass token confidence as a SQL locator."""

    def __init__(self, model_id: str, device: str):
        self.model_id = model_id
        self.device = device
        self.model = LLaDACoder(model_id=model_id, device=device)

    @torch.inference_mode()
    def score(self, prompt_text: str, draft_text: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
        messages = [{"role": "user", "content": prompt_text}]
        prompt_text_chat = self.model.tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt_enc = self.model.tok(
            prompt_text_chat,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        prompt_ids = prompt_enc["input_ids"].to(self.device)
        attention_mask = prompt_enc["attention_mask"].to(self.device)
        comp_ids = self.model.tok(
            draft_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(self.device)
        if comp_ids.shape[1] == 0:
            return np.array([], dtype=np.float32), []
        confidence = self.model.score_tokens(prompt_ids, comp_ids, attention_mask)
        spans = get_token_char_spans(self.model.tok, draft_text)
        return confidence.cpu().numpy().astype(np.float32), spans


def char_fault_mask(pred: str, gold: str) -> np.ndarray:
    """Return a char mask over pred; True means this char differs from gold."""
    mask = np.zeros(len(pred), dtype=bool)
    matcher = difflib.SequenceMatcher(None, pred, gold, autojunk=False)
    for op, a0, a1, _b0, _b1 in matcher.get_opcodes():
        if op == "equal":
            continue
        if a0 < a1:
            mask[a0:a1] = True
        elif len(pred) > 0:
            # Gold has an insertion relative to pred. Mark the nearest
            # neighboring generated character so omission-only diffs are counted.
            pos = max(0, min(a0, len(pred) - 1))
            mask[pos:pos + 1] = True
    return mask


def _jsonl_records(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] skipping non-JSON line {line_no}: {line[:80]}", file=sys.stderr)


def _build_locator(args: argparse.Namespace) -> LocatorLike:
    if args.locator == "dream":
        return DreamCoderLocator(
            model_id=args.locator_model_id or "Dream-org/Dream-Coder-v0-Instruct-7B",
            device=args.device,
        )
    if args.locator == "llada":
        return LLaDALocator(
            model_id=args.locator_model_id or "GSAI-ML/LLaDA-8B-Instruct",
            device=args.device,
        )
    if args.locator == "ar":
        return ARLocator(
            model_id=args.locator_model_id or "deepseek-ai/deepseek-coder-6.7b-instruct",
            device=args.device,
        )
    if args.locator == "bert":
        return BERTLocator(
            model_id=args.locator_model_id or "microsoft/codebert-base-mlm",
            device=args.device,
        )
    raise ValueError(f"Unsupported locator: {args.locator}")


def _mean(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval_jsonl", required=True, help="Output JSONL from sql_eval.py.")
    ap.add_argument("--locator", choices=["dream", "llada", "ar", "bert"], default="dream")
    ap.add_argument("--locator_model_id", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_samples", type=int, default=100)
    ap.add_argument("--summary_out", default=None, help="Optional JSON summary path.")
    args = ap.parse_args()

    failed: list[dict] = []
    for rec in _jsonl_records(pathlib.Path(args.eval_jsonl)):
        if rec.get("exec_pass") is False and rec.get("prompt"):
            pred_sql = extract_sql(rec.get("pred_sql") or rec.get("raw_completion") or "")
            gold_sql = str(rec.get("gold_sql") or "")
            if pred_sql and gold_sql:
                rec = dict(rec)
                rec["pred_sql"] = pred_sql
                rec["gold_sql"] = gold_sql.strip()
                failed.append(rec)

    print(f"AR-failed samples: {len(failed)}, using first {args.max_samples}")
    failed = failed[: args.max_samples]
    if not failed:
        print("No failed samples to analyze.")
        return

    locator = _build_locator(args)

    fault_confs: list[float] = []
    nonfault_confs: list[float] = []
    per_sample: list[dict] = []

    for sample_idx, rec in enumerate(failed, start=1):
        prompt = rec["prompt"]
        pred_sql = rec["pred_sql"]
        gold_sql = rec["gold_sql"]
        fault_mask = char_fault_mask(pred_sql, gold_sql)
        if len(fault_mask) == 0:
            continue

        conf_arr, spans = locator.score(prompt, pred_sql)
        n_fault_spans = 0
        n_nonfault_spans = 0

        for idx, (char_start, char_end) in enumerate(spans):
            if idx >= len(conf_arr):
                break
            conf_value = float(conf_arr[idx])
            if not np.isfinite(conf_value):
                continue
            start = max(0, min(int(char_start), len(fault_mask)))
            end = max(start, min(int(char_end), len(fault_mask)))
            is_fault = bool(fault_mask[start:end].any()) if end > start else False
            if is_fault:
                fault_confs.append(conf_value)
                n_fault_spans += 1
            else:
                nonfault_confs.append(conf_value)
                n_nonfault_spans += 1

        per_sample.append({
            "task_id": rec.get("task_id"),
            "n_fault_chars": int(fault_mask.sum()),
            "n_fault_spans": n_fault_spans,
            "n_nonfault_spans": n_nonfault_spans,
        })
        print(
            f"[{sample_idx}/{len(failed)}] {rec.get('task_id')} "
            f"fault_spans={n_fault_spans} nonfault_spans={n_nonfault_spans}"
        )

    if not fault_confs:
        summary = {
            "script": "sql_locator_analysis",
            "eval_jsonl": args.eval_jsonl,
            "locator": args.locator,
            "locator_model_id": getattr(locator, "model_id", args.locator_model_id),
            "samples_analyzed": len(per_sample),
            "total_fault_spans": 0,
            "total_nonfault_spans": len(nonfault_confs),
            "mean_conf_fault": None,
            "mean_conf_nonfault": _mean(nonfault_confs),
            "fault_detection_ratio": None,
            "per_sample": per_sample,
        }
        if args.summary_out:
            out_path = pathlib.Path(args.summary_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        print("No fault tokens found.")
        return

    mean_fault = _mean(fault_confs)
    mean_nonfault = _mean(nonfault_confs)
    ratio = (
        mean_nonfault / mean_fault
        if mean_fault > 0 and not np.isnan(mean_nonfault)
        else float("inf")
    )

    print(f"\n=== dLLM Fault Detection Ratio ({args.locator}) ===")
    print(f"  Samples analyzed:       {len(per_sample)}")
    print(f"  Total fault spans:      {len(fault_confs)}")
    print(f"  Total non-fault spans:  {len(nonfault_confs)}")
    print(f"  Mean conf @ fault:      {mean_fault:.6f}")
    print(f"  Mean conf @ non-fault:  {mean_nonfault:.6f}")
    print(f"  Fault detection ratio:  {ratio:.2f}x")
    print()
    print("Interpretation:")
    if ratio >= 3.0:
        print(f"  ratio={ratio:.2f}x -> dLLM detects SQL errors; CoCoder SQL is feasible")
    elif ratio >= 1.5:
        print(f"  ratio={ratio:.2f}x -> weak signal; marginal feasibility")
    else:
        print(f"  ratio={ratio:.2f}x -> weak/no SQL fault-detection signal")

    summary = {
        "script": "sql_locator_analysis",
        "eval_jsonl": args.eval_jsonl,
        "locator": args.locator,
        "locator_model_id": getattr(locator, "model_id", args.locator_model_id),
        "samples_analyzed": len(per_sample),
        "total_fault_spans": len(fault_confs),
        "total_nonfault_spans": len(nonfault_confs),
        "mean_conf_fault": mean_fault,
        "mean_conf_nonfault": mean_nonfault,
        "fault_detection_ratio": ratio,
        "per_sample": per_sample,
    }
    if args.summary_out:
        out_path = pathlib.Path(args.summary_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

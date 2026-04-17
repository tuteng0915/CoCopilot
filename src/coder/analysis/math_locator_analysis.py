#!/usr/bin/env python3
"""
math_locator_analysis.py — Does LLaDA confidence signal locate errors in math CoT?

For each math completion (gen_math.py output), scores tokens via a single LLaDA
forward pass, segments the completion into line-level "steps", and checks:

  1. Overall: are incorrect completions less confident than correct ones?
  2. Step-level: does the min-confidence step in incorrect completions tend to
     be the actual error location?

The analysis uses a single forward pass (same as the production locator in
gen_remask), NOT leave-one-out masking — results are directly comparable to
what the real pipeline would see.

Usage:
  python -m coder.analysis.math_locator_analysis \\
      --input outputs/base_tuteng/llama31_gsm8k.jsonl \\
      --dataset gsm8k \\
      --device cuda \\
      --out outputs/math/llama31_gsm8k_locator_analysis.json

  # Run without loading LLaDA (score only; use --no_model for CPU-only quick check)
  python -m coder.analysis.math_locator_analysis \\
      --input outputs/base_tuteng/llama31_gsm8k.jsonl \\
      --dataset gsm8k --no_model
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


# ── Eval helpers (mirrors eval_math.py) ──────────────────────────────────────

import re
import unicodedata


def _normalize_number(s: str) -> str:
    s = s.strip().replace(",", "")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def _normalize_latex(s: str) -> str:
    s = s.strip().strip("$").strip()
    s = re.sub(r"\s+", " ", s)
    return unicodedata.normalize("NFC", s)


def _extract_gsm8k_answer(text: str) -> Optional[str]:
    matches = re.findall(r"####\s*([^\n]+)", text)
    if matches:
        return _normalize_number(matches[-1].strip())
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return _normalize_number(nums[-1]) if nums else None


def _extract_math500_answer(text: str) -> Optional[str]:
    last_start = None
    for m in re.finditer(r"\\boxed\{", text):
        last_start = m.end()
    if last_start is None:
        return None
    depth, i = 1, last_start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return _normalize_latex(text[last_start: i - 1]) if depth == 0 else None


def is_correct(dataset: str, completion: str, ref: str) -> bool:
    if dataset == "gsm8k":
        pred = _extract_gsm8k_answer(completion)
        return pred == _normalize_number(ref) if pred else False
    else:
        pred = _extract_math500_answer(completion)
        if pred is None:
            return False
        p, r = _normalize_latex(pred), _normalize_latex(ref)
        if p == r:
            return True
        try:
            return abs(float(p) - float(r)) < 1e-6
        except (ValueError, TypeError):
            return False


# ── Scoring ───────────────────────────────────────────────────────────────────

@dataclass
class StepInfo:
    line_idx:   int
    text:       str
    mean_conf:  float
    min_conf:   float
    n_tokens:   int


@dataclass
class RecordAnalysis:
    rec_id:      str
    correct:     bool
    mean_conf:   float           # mean over all completion tokens
    n_steps:     int
    worst_step:  int             # line index with lowest mean_conf
    steps:       list[StepInfo]  # per-line info


def _apply_chat_template(tokenizer, prompt_text: str) -> str:
    """Wrap prompt in chat template, returning the formatted string."""
    try:
        messages = [{"role": "user", "content": prompt_text}]
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
    except Exception:
        return prompt_text


@torch.inference_mode()
def score_steps(
    model,
    tokenizer,
    prompt_text: str,
    completion: str,
    device: str,
    logit_shift: bool = False,
) -> list[StepInfo]:
    """
    Single dLLM forward pass → per-token confidence → step-level aggregation.

    logit_shift=True for Dream-family (logits[:, t, :] predicts token t via
    left-shift convention); logit_shift=False for LLaDA (direct alignment).
    Returns one StepInfo per non-empty line.
    """
    formatted_prompt = _apply_chat_template(tokenizer, prompt_text)

    prompt_ids = tokenizer(
        formatted_prompt, add_special_tokens=False, return_tensors="pt",
    ).input_ids.to(device)

    # Try fast-tokenizer offset mapping; fall back to greedy matching for slow tokenizers
    try:
        comp_enc = tokenizer(
            completion, add_special_tokens=False,
            return_offsets_mapping=True, return_tensors="pt",
        )
        comp_ids = comp_enc.input_ids.to(device)
        M = comp_ids.shape[1]
        if M == 0:
            return []
        raw_offsets = comp_enc.get("offset_mapping")
        char_spans = [(int(s), int(e)) for s, e in raw_offsets[0].tolist()] \
            if raw_offsets is not None else None
    except (NotImplementedError, Exception):
        comp_enc = tokenizer(completion, add_special_tokens=False, return_tensors="pt")
        comp_ids = comp_enc.input_ids.to(device)
        M = comp_ids.shape[1]
        if M == 0:
            return []
        char_spans = None

    # Greedy character-span fallback (for slow tokenizers like Dream)
    if char_spans is None:
        char_spans = []
        pos = 0
        for tid in comp_ids[0].tolist():
            tok_str = tokenizer.decode([tid], skip_special_tokens=False)
            clean = tok_str.replace("\u2581", " ").replace("##", "").replace("\u0120", " ")
            idx = completion.find(clean, pos)
            if idx == -1:
                char_spans.append((pos, pos + max(len(clean), 1)))
                pos += max(len(clean), 1)
            else:
                char_spans.append((idx, idx + len(clean)))
                pos = idx + len(clean)

    # Single forward pass — same as production gen_remask locator
    full_ids = torch.cat([prompt_ids, comp_ids], dim=1)
    logits = model(full_ids).logits
    if logit_shift:
        # Dream convention: shift logits left so logits[:, t, :] → P(token_t)
        logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
    comp_logits = logits[0, prompt_ids.shape[1]:, :].float()
    probs = torch.softmax(comp_logits, dim=-1)
    conf = probs[torch.arange(M, device=device), comp_ids[0]].cpu().numpy()

    # Map each token to its line index (split by \n in completion text)
    line_ends: list[int] = []
    pos = 0
    for line in completion.split("\n"):
        pos += len(line) + 1  # +1 for the \n
        line_ends.append(pos - 1)

    def char_to_line(char_start: int) -> int:
        for li, le in enumerate(line_ends):
            if char_start <= le:
                return li
        return len(line_ends) - 1

    # Aggregate confidence by line
    line_confs: dict[int, list[float]] = {}
    for tok_idx, (cs, _ce) in enumerate(char_spans):
        li = char_to_line(cs)
        line_confs.setdefault(li, []).append(float(conf[tok_idx]))

    lines = completion.split("\n")
    steps: list[StepInfo] = []
    for li, line_text in enumerate(lines):
        if not line_text.strip():
            continue
        lc = line_confs.get(li, [])
        if not lc:
            continue
        steps.append(StepInfo(
            line_idx=li,
            text=line_text,
            mean_conf=float(np.mean(lc)),
            min_conf=float(np.min(lc)),
            n_tokens=len(lc),
        ))

    return steps


def analyze_record(
    model,
    tokenizer,
    rec: dict,
    dataset: str,
    device: str,
    logit_shift: bool = False,
) -> RecordAnalysis:
    rec_id   = rec.get("id", rec.get("task_id", "?"))
    prompt   = rec.get("prompt", "")
    raw_comp = rec.get("raw_completion", "")
    ref      = rec.get("answer_ref", rec.get("answer", ""))
    correct  = is_correct(dataset, raw_comp, ref)

    steps = score_steps(model, tokenizer, prompt, raw_comp, device, logit_shift=logit_shift)

    if not steps:
        return RecordAnalysis(
            rec_id=rec_id, correct=correct,
            mean_conf=float("nan"), n_steps=0, worst_step=-1, steps=[],
        )

    all_confs = [c for s in steps for _ in range(s.n_tokens)
                 for c in [s.mean_conf]]  # approximate
    mean_conf = float(np.mean([s.mean_conf for s in steps]))
    worst_step = int(np.argmin([s.mean_conf for s in steps]))

    return RecordAnalysis(
        rec_id=rec_id, correct=correct,
        mean_conf=mean_conf, n_steps=len(steps), worst_step=worst_step, steps=steps,
    )


# ── Summary stats ─────────────────────────────────────────────────────────────

def print_summary(analyses: list[RecordAnalysis]) -> None:
    correct   = [a for a in analyses if a.correct]
    incorrect = [a for a in analyses if not a.correct]

    def _stats(group: list[RecordAnalysis], label: str) -> None:
        if not group:
            print(f"  {label}: (empty)")
            return
        mc = [a.mean_conf for a in group if not math.isnan(a.mean_conf)]
        ws = [s.mean_conf
              for a in group
              for i, s in enumerate(a.steps) if i == a.worst_step]
        print(f"  {label} (n={len(group)})")
        print(f"    mean_conf (over whole completion): "
              f"{np.mean(mc):.4f} ± {np.std(mc):.4f}")
        print(f"    worst_step_conf:                   "
              f"{np.mean(ws):.4f} ± {np.std(ws):.4f}")

    print("\n" + "=" * 70)
    print("  Math Locator Analysis — LLaDA confidence vs. correctness")
    print("  " + "-" * 66)
    _stats(correct, "Correct  ")
    _stats(incorrect, "Incorrect")
    print()

    # Step-position distribution: where does the worst step tend to fall?
    if incorrect:
        total_steps = [a.n_steps for a in incorrect if a.n_steps > 0]
        worst_pos_frac = [
            a.worst_step / (a.n_steps - 1)
            for a in incorrect
            if a.n_steps > 1
        ]
        print(f"  Worst-step position (incorrect, as fraction of total steps):")
        if worst_pos_frac:
            hist, edges = np.histogram(worst_pos_frac, bins=[0, 0.25, 0.5, 0.75, 1.01])
            for i, count in enumerate(hist):
                lo, hi = edges[i], edges[i + 1]
                bar = "█" * int(count * 30 / max(hist, default=1))
                print(f"    [{lo:.2f}, {hi:.2f}) {bar} {count}")
        print(f"  Mean steps per (incorrect) problem: {np.mean(total_steps):.1f}")

    print("=" * 70 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="JSONL from gen_math.py")
    ap.add_argument("--dataset", choices=["gsm8k", "math500"], default=None,
                    help="If omitted, inferred from records.")
    ap.add_argument("--model_family",
                    choices=["llada", "dream", "dream_coder"],
                    default="llada",
                    help="dLLM family: llada (default), dream (general), dream_coder.")
    ap.add_argument("--model_id", default=None,
                    help="Override HuggingFace model ID. "
                         "Defaults: llada=GSAI-ML/LLaDA-8B-Instruct, "
                         "dream=Dream-org/Dream-v0-Instruct-7B, "
                         "dream_coder=Dream-org/Dream-Coder-v0-Instruct-7B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0,
                    help="Only analyse first N records (0 = all).")
    ap.add_argument("--out", default=None,
                    help="Write per-record JSON analysis here (optional).")
    ap.add_argument("--no_model", action="store_true",
                    help="Skip model loading; just report correctness distribution.")
    args = ap.parse_args()

    # Load records
    records: list[dict] = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("[error] no records found in", args.input)
        sys.exit(1)

    # Infer dataset
    dataset = args.dataset
    if dataset is None:
        first_id = records[0].get("id", "")
        if "gsm8k" in first_id:
            dataset = "gsm8k"
        elif "math500" in first_id:
            dataset = "math500"
        else:
            dataset = records[0].get("dataset", "gsm8k")
    print(f"[data] {len(records)} records loaded, dataset={dataset}")

    if args.limit > 0:
        records = records[:args.limit]
        print(f"[data] limited to {len(records)} records")

    # Quick correctness count (no model needed)
    n_correct = sum(
        1 for r in records
        if is_correct(dataset, r.get("raw_completion", ""), r.get("answer_ref", r.get("answer", "")))
    )
    print(f"[eval] {n_correct}/{len(records)} correct "
          f"({100*n_correct/len(records):.1f}%)")

    if args.no_model:
        print("[skip] --no_model: skipping model scoring")
        return

    # Resolve model ID and logit convention
    _DEFAULTS = {
        "llada":       "GSAI-ML/LLaDA-8B-Instruct",
        "dream":       "Dream-org/Dream-v0-Instruct-7B",
        "dream_coder": "Dream-org/Dream-Coder-v0-Instruct-7B",
    }
    model_id   = args.model_id or _DEFAULTS[args.model_family]
    logit_shift = args.model_family in ("dream", "dream_coder")
    print(f"\n[model] loading {model_id} (family={args.model_family}, "
          f"logit_shift={logit_shift}) …")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(args.device).eval()
    print("[model] loaded\n")

    analyses: list[RecordAnalysis] = []
    for i, rec in enumerate(records):
        rec_id = rec.get("id", rec.get("task_id", f"#{i}"))
        print(f"[{i+1}/{len(records)}] {rec_id}", end="", flush=True)
        a = analyze_record(model, tokenizer, rec, dataset, args.device,
                           logit_shift=logit_shift)
        analyses.append(a)
        status = "✓" if a.correct else "✗"
        worst_conf = a.steps[a.worst_step].mean_conf if a.steps and a.worst_step >= 0 else float("nan")
        print(f"  {status}  mean_conf={a.mean_conf:.3f}  "
              f"worst_step={a.worst_step}  worst_conf={worst_conf:.3f}")

    print_summary(analyses)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": dataset,
                    "n_correct": n_correct,
                    "n_total": len(records),
                    "records": [asdict(a) for a in analyses],
                },
                f, ensure_ascii=False, indent=2,
            )
        print(f"[out] wrote {out_path}")


if __name__ == "__main__":
    main()

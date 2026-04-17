#!/usr/bin/env python3
"""
math_lso_analysis.py — Leave-Sentence-Out (LSO) locator analysis for math CoT.

Unlike single-pass token scoring (math_locator_analysis.py), LSO asks:
  "How surprised is the dLLM by this sentence, given everything else?"

For each step/line in a completion, we:
  1. Mask all tokens belonging to that line (set them to [MASK]).
  2. Run one dLLM forward pass — the model sees full bidirectional context
     around the masked region.
  3. Compute:
       - recon_nll:  -mean log P(original_token | masked context)   [lower = model can reconstruct it]
       - recon_acc:  fraction of masked tokens where argmax == original token
  4. The step with the highest recon_nll is the "suspected error step".

Hypothesis: a wrong arithmetic step (e.g., "24 × 2/3 = 15") should have
higher reconstruction NLL than a correct one ("24 × 2/3 = 16"), because the
dLLM — seeing the surrounding context bidirectionally — would predict the
correct continuation.

This is ~N_steps times slower than single-pass scoring but uses the dLLM's
bidirectional attention properly.

Usage:
  python -m coder.analysis.math_lso_analysis \\
      --input  outputs/base_tuteng/llama31_gsm8k.jsonl \\
      --dataset gsm8k \\
      --model_family llada \\
      --limit 200 \\
      --device cuda \\
      --out outputs/math/llama31_gsm8k_lso_llada.json
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
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# ── Eval helpers (same as math_locator_analysis.py) ──────────────────────────

import re, unicodedata

def _normalize_number(s):
    s = s.strip().replace(",", "")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s

def _normalize_latex(s):
    s = s.strip().strip("$").strip()
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", s))

def _extract_gsm8k_answer(text):
    ms = re.findall(r"####\s*([^\n]+)", text)
    if ms:
        return _normalize_number(ms[-1].strip())
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return _normalize_number(nums[-1]) if nums else None

def _extract_math500_answer(text):
    last = None
    for m in re.finditer(r"\\boxed\{", text):
        last = m.end()
    if last is None:
        return None
    depth, i = 1, last
    while i < len(text) and depth > 0:
        depth += (text[i] == "{") - (text[i] == "}")
        i += 1
    return _normalize_latex(text[last:i-1]) if depth == 0 else None

def is_correct(dataset, completion, ref):
    pred = _extract_gsm8k_answer(completion) if dataset == "gsm8k" \
           else _extract_math500_answer(completion)
    if pred is None:
        return False
    if dataset == "gsm8k":
        return pred == _normalize_number(ref)
    p, r = _normalize_latex(pred), _normalize_latex(ref)
    if p == r:
        return True
    try:
        return abs(float(p) - float(r)) < 1e-6
    except (ValueError, TypeError):
        return False


# ── Tokenization helpers ──────────────────────────────────────────────────────

def _apply_chat_template(tokenizer, prompt_text):
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True, tokenize=False,
        )
    except Exception:
        return prompt_text


def _get_char_spans(tokenizer, text):
    """Return per-token (start, end) character spans in `text`."""
    try:
        enc = tokenizer(text, add_special_tokens=False,
                        return_offsets_mapping=True, return_tensors="pt")
        spans = enc.get("offset_mapping")
        if spans is not None:
            return enc.input_ids, [(int(s), int(e)) for s, e in spans[0].tolist()]
    except (NotImplementedError, Exception):
        pass

    # Slow-tokenizer fallback: greedy character matching
    enc = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    ids = enc.input_ids
    spans, pos = [], 0
    for tid in ids[0].tolist():
        tok_str = tokenizer.decode([tid], skip_special_tokens=False)
        clean = tok_str.replace("\u2581", " ").replace("##", "").replace("\u0120", " ")
        idx = text.find(clean, pos)
        if idx == -1:
            spans.append((pos, pos + max(len(clean), 1)))
            pos += max(len(clean), 1)
        else:
            spans.append((idx, idx + len(clean)))
            pos = idx + len(clean)
    return ids, spans


def _build_line_token_map(completion, char_spans):
    """Map each token index to its line index (split by \\n)."""
    lines = completion.split("\n")
    line_end_char = []
    pos = 0
    for line in lines:
        pos += len(line) + 1
        line_end_char.append(pos - 1)

    def char_to_line(c):
        for li, le in enumerate(line_end_char):
            if c <= le:
                return li
        return len(line_end_char) - 1

    token_to_line = [char_to_line(cs) for cs, _ in char_spans]
    line_to_tokens: dict[int, list[int]] = {}
    for ti, li in enumerate(token_to_line):
        line_to_tokens.setdefault(li, []).append(ti)

    return lines, line_to_tokens


# ── LSO scoring ───────────────────────────────────────────────────────────────

@dataclass
class LSOStepInfo:
    line_idx:    int
    text:        str
    recon_nll:   float   # -mean log P(orig token | rest masked) — higher = model more surprised
    recon_acc:   float   # fraction argmax == original token
    n_tokens:    int


@dataclass
class LSORecordAnalysis:
    rec_id:      str
    correct:     bool
    worst_step:  int             # line index with highest recon_nll
    worst_nll:   float
    mean_nll:    float           # mean recon_nll across all steps
    steps:       list[LSOStepInfo]


@torch.inference_mode()
def score_steps_lso(
    model,
    tokenizer,
    prompt_text: str,
    completion: str,
    device: str,
    mask_id: int,
    logit_shift: bool,
    min_tokens_per_step: int = 3,
) -> list[LSOStepInfo]:
    """
    For each non-trivial line, mask its tokens and run one forward pass.
    Returns LSOStepInfo per scored line.
    """
    formatted = _apply_chat_template(tokenizer, prompt_text)
    prompt_ids = tokenizer(
        formatted, add_special_tokens=False, return_tensors="pt",
    ).input_ids.to(device)
    P = prompt_ids.shape[1]

    comp_ids, char_spans = _get_char_spans(tokenizer, completion)
    comp_ids = comp_ids.to(device)
    M = comp_ids.shape[1]
    if M == 0:
        return []

    lines, line_to_tokens = _build_line_token_map(completion, char_spans)
    base_full = torch.cat([prompt_ids, comp_ids], dim=1)  # [1, P+M]

    results: list[LSOStepInfo] = []

    for li, tok_indices in line_to_tokens.items():
        line_text = lines[li] if li < len(lines) else ""
        if not line_text.strip() or len(tok_indices) < min_tokens_per_step:
            continue

        # Mask this line's tokens
        masked = base_full.clone()
        for ti in tok_indices:
            masked[0, P + ti] = mask_id

        logits = model(masked).logits  # [1, P+M, V]
        if logit_shift:
            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)

        comp_logits = logits[0, P:, :].float()          # [M, V]
        probs = torch.softmax(comp_logits, dim=-1)       # [M, V]

        tok_idx_t = torch.tensor(tok_indices, device=device)
        orig_ids  = comp_ids[0, tok_idx_t]              # original token ids

        orig_probs = probs[tok_idx_t, orig_ids]          # P(orig | masked context)
        recon_nll  = float(-torch.log(orig_probs.clamp(min=1e-9)).mean())
        recon_acc  = float((probs[tok_idx_t].argmax(dim=-1) == orig_ids).float().mean())

        results.append(LSOStepInfo(
            line_idx=li,
            text=line_text,
            recon_nll=recon_nll,
            recon_acc=recon_acc,
            n_tokens=len(tok_indices),
        ))

    return results


def analyze_record_lso(model, tokenizer, rec, dataset, device, mask_id, logit_shift):
    rec_id   = rec.get("id", rec.get("task_id", "?"))
    prompt   = rec.get("prompt", "")
    raw_comp = rec.get("raw_completion", "")
    ref      = rec.get("answer_ref", rec.get("answer", ""))
    correct  = is_correct(dataset, raw_comp, ref)

    steps = score_steps_lso(model, tokenizer, prompt, raw_comp, device,
                             mask_id, logit_shift)
    if not steps:
        return LSORecordAnalysis(rec_id=rec_id, correct=correct,
                                 worst_step=-1, worst_nll=float("nan"),
                                 mean_nll=float("nan"), steps=[])

    worst_idx = int(np.argmax([s.recon_nll for s in steps]))
    return LSORecordAnalysis(
        rec_id=rec_id, correct=correct,
        worst_step=steps[worst_idx].line_idx,
        worst_nll=steps[worst_idx].recon_nll,
        mean_nll=float(np.mean([s.recon_nll for s in steps])),
        steps=steps,
    )


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(analyses: list[LSORecordAnalysis], model_family: str) -> None:
    correct   = [a for a in analyses if a.correct]
    incorrect = [a for a in analyses if not a.correct]

    print(f"\n{'='*70}")
    print(f"  LSO Math Locator Analysis  —  {model_family}  "
          f"(n_correct={len(correct)}, n_incorrect={len(incorrect)})")
    print(f"  {'Metric':<30}  {'Correct':>10}  {'Incorrect':>10}  {'Δ':>8}  {'direction':>10}")
    print(f"  {'-'*68}")

    def _row(label, fn, higher_is_fault=True):
        c  = [fn(a) for a in correct   if not math.isnan(fn(a))]
        ic = [fn(a) for a in incorrect if not math.isnan(fn(a))]
        if not c or not ic:
            return
        mc, mic = np.mean(c), np.mean(ic)
        delta = mic - mc
        ok = (delta > 0) == higher_is_fault
        direction = "✓" if ok else "✗"
        print(f"  {label:<30}  {mc:>10.4f}  {mic:>10.4f}  {delta:>+8.4f}  {direction:>10}")

    _row("worst_step recon_nll",  lambda a: a.worst_nll,  higher_is_fault=True)
    _row("mean_step  recon_nll",  lambda a: a.mean_nll,   higher_is_fault=True)
    _row("worst_step recon_acc",  lambda a: a.steps[
            next(i for i,s in enumerate(a.steps) if s.line_idx == a.worst_step)
        ].recon_acc if a.steps else float("nan"),          higher_is_fault=False)

    # Position distribution of worst step for incorrect
    if incorrect:
        pos_fracs = [
            next((i for i, s in enumerate(a.steps) if s.line_idx == a.worst_step), 0)
            / max(len(a.steps) - 1, 1)
            for a in incorrect if a.steps
        ]
        print(f"\n  Worst-step position (incorrect, fraction of all steps):")
        hist, edges = np.histogram(pos_fracs, bins=[0, 0.25, 0.5, 0.75, 1.01])
        for i, count in enumerate(hist):
            bar = "█" * int(count * 30 / max(hist.max(), 1))
            print(f"    [{edges[i]:.2f}, {edges[i+1]:.2f})  {bar} {count}")

    print(f"{'='*70}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

_DEFAULTS = {
    "llada":       ("GSAI-ML/LLaDA-8B-Instruct",         126336, False),
    "dream":       ("Dream-org/Dream-v0-Instruct-7B",     None,   True),
    "dream_coder": ("Dream-org/Dream-Coder-v0-Instruct-7B", None, True),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input",   required=True)
    ap.add_argument("--dataset", choices=["gsm8k", "math500"], default=None)
    ap.add_argument("--model_family", choices=list(_DEFAULTS.keys()), default="llada")
    ap.add_argument("--model_id",  default=None)
    ap.add_argument("--device",    default="cuda")
    ap.add_argument("--limit",     type=int, default=0)
    ap.add_argument("--out",       default=None)
    ap.add_argument("--min_tokens_per_step", type=int, default=3,
                    help="Skip steps with fewer tokens (likely blank lines / headers).")
    args = ap.parse_args()

    # Load records
    records = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        print("[error] no records"); sys.exit(1)

    dataset = args.dataset or records[0].get("dataset",
              "gsm8k" if "gsm8k" in records[0].get("id","") else "math500")
    if args.limit > 0:
        records = records[:args.limit]

    n_correct = sum(1 for r in records
                    if is_correct(dataset, r.get("raw_completion",""),
                                  r.get("answer_ref", r.get("answer",""))))
    print(f"[data] {len(records)} records, dataset={dataset}, "
          f"correct={n_correct}/{len(records)} ({100*n_correct/len(records):.1f}%)")

    # Resolve model config
    default_id, default_mask_id, logit_shift = _DEFAULTS[args.model_family]
    model_id = args.model_id or default_id

    print(f"[model] loading {model_id} …")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(args.device).eval()

    # Resolve mask token id
    mask_id = default_mask_id
    if mask_id is None:
        mask_id = getattr(model.config, "mask_token_id", None)
    if mask_id is None:
        # Try tokenizer
        mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
    print(f"[model] loaded  mask_id={mask_id}  logit_shift={logit_shift}")

    analyses: list[LSORecordAnalysis] = []
    for i, rec in enumerate(tqdm(records, desc="lso")):
        rec_id = rec.get("id", f"#{i}")
        a = analyze_record_lso(model, tokenizer, rec, dataset,
                               args.device, mask_id, logit_shift)
        analyses.append(a)
        status = "✓" if a.correct else "✗"
        tqdm.write(f"[{i+1}/{len(records)}] {rec_id}  {status}  "
                   f"worst_nll={a.worst_nll:.3f}  worst_step={a.worst_step}")

    print_summary(analyses, args.model_family)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "model_family": args.model_family,
            "model_id": model_id,
            "dataset": dataset,
            "n_correct": n_correct,
            "n_total": len(records),
            "records": [asdict(a) for a in analyses],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[out] wrote {out_path}")


if __name__ == "__main__":
    main()

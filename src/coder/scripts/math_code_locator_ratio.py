"""math_code_locator_ratio.py — Phase 2: fault detection ratio on math-to-code.

Computes the dLLM fault-detection ratio for "surgical" correction pairs:
  ratio = P(non-fault) / P(fault)

where P(fault) is the dLLM confidence assigned to the error token(s) and
P(non-fault) is the mean confidence on all other tokens.

A high ratio (>10×) means the dLLM can distinguish error tokens from correct ones.
Compare to: text-CoT ≈ 1.15×, coding benchmark ≈ 23–126×.

Usage:
  python -m coder.scripts.math_code_locator_ratio \\
    --ar_file    outputs/math_code/deepseek_gsm8k_code.jsonl \\
    --collab_file outputs/math_code/deepseek_gsm8k_code_dream_t0.9.jsonl \\
    --ar_eval    outputs/math_code/deepseek_gsm8k_code_eval.json \\
    --col_eval   outputs/math_code/deepseek_gsm8k_code_dream_t0.9_eval.json \\
    --out        outputs/math_code/deepseek_gsm8k_code_locator_ratio.json
"""
from __future__ import annotations

import argparse
import difflib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from coder.models import DreamCoder
from coder.locators import get_token_char_spans


# ── Pair selection ────────────────────────────────────────────────────────────

def load_eval_correct(path: str) -> Dict[str, bool]:
    d = json.loads(Path(path).read_text())
    results = d.get("results", [])
    return {str(r["id"]): bool(r.get("correct", False)) for r in results}


def load_jsonl_by_id(path: str) -> Dict[str, Dict[str, Any]]:
    recs: Dict[str, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rid = str(r.get("id", ""))
            if rid:
                recs[rid] = r
    return recs


def find_surgical_pairs(
    ar_recs: Dict[str, Dict[str, Any]],
    col_recs: Dict[str, Dict[str, Any]],
    ar_correct: Dict[str, bool],
    col_correct: Dict[str, bool],
    max_diff_chars: int = 60,
) -> List[Tuple[str, str, str]]:
    """Return list of (id, ar_code, col_code) where:
      - AR failed, CoCoder succeeded
      - char-level diff <= max_diff_chars (surgical correction)
    """
    pairs = []
    for rid in ar_recs:
        if rid not in col_recs:
            continue
        if ar_correct.get(rid, True):   # AR already correct — skip
            continue
        if not col_correct.get(rid, False):  # CoCoder also failed — skip
            continue
        ar_code  = ar_recs[rid].get("raw_completion", "")
        col_code = col_recs[rid].get("raw_completion", "")
        sm = difflib.SequenceMatcher(None, ar_code, col_code, autojunk=False)
        diff_len = sum(a1 - a0
                       for op, a0, a1, b0, b1 in sm.get_opcodes()
                       if op != "equal")
        if 1 <= diff_len <= max_diff_chars:
            pairs.append((rid, ar_code, col_code))
    return pairs


# ── Token-level confidence ────────────────────────────────────────────────────

def score_code_confidence(
    model: DreamCoder,
    question: str,
    code: str,
    dataset: str,
) -> Optional[torch.Tensor]:
    """Return per-token dLLM confidence tensor for `code`, or None on error."""
    # Use same prompt as gen_math_code so context matches
    from coder.scripts.gen_math_code import build_prompt, _LOADERS
    # Build a dummy item to reuse build_prompt
    item = {"question": question}
    try:
        prompt = build_prompt(dataset, item)
    except Exception:
        prompt = question + "\n\ndef solution():\n"

    try:
        prompt_ids = model.tok.encode(prompt, return_tensors="pt").to(model.device)
        comp_ids   = model.tok.encode(code,   return_tensors="pt").to(model.device)
        if comp_ids.shape[1] == 0:
            return None
        conf = model.score_tokens(prompt_ids, comp_ids)
        return conf.float().cpu()
    except Exception as exc:
        print(f"[warn] scoring failed: {exc}")
        return None


def locate_fault_token_indices(
    ar_code: str,
    col_code: str,
    ar_conf: torch.Tensor,
    model: DreamCoder,
) -> Tuple[List[int], List[int]]:
    """
    Identify which token indices in ar_code correspond to changed characters.
    Returns (fault_indices, nonfault_indices).
    """
    sm = difflib.SequenceMatcher(None, ar_code, col_code, autojunk=False)
    changed_chars: set[int] = set()
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        if op != "equal":
            changed_chars.update(range(a0, a1))

    tok_spans = get_token_char_spans(model.tok, ar_code)
    fault_idxs: List[int] = []
    nonfault_idxs: List[int] = []
    for i, (start, end) in enumerate(tok_spans):
        if i >= len(ar_conf):
            break
        overlap = any(c in changed_chars for c in range(start, end))
        if overlap:
            fault_idxs.append(i)
        else:
            nonfault_idxs.append(i)

    return fault_idxs, nonfault_idxs


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute dLLM fault-detection ratio on math-to-code correction pairs."
    )
    ap.add_argument("--ar_file",     required=True, help="AR draft JSONL")
    ap.add_argument("--collab_file", required=True, help="CoCoder output JSONL")
    ap.add_argument("--ar_eval",     required=True, help="AR eval JSON")
    ap.add_argument("--col_eval",    required=True, help="CoCoder eval JSON")
    ap.add_argument("--dataset",     required=True,
                    choices=["gsm8k", "math500", "aime", "aime2025"])
    ap.add_argument("--out",         required=True, help="Output JSON path")
    ap.add_argument("--max_diff_chars", type=int, default=60,
                    help="Max changed characters to qualify as surgical pair")
    ap.add_argument("--model_id",    type=str, default="Dream-org/Dream-Coder-v0-Instruct-7B")
    ap.add_argument("--device",      type=str, default="cuda")
    args = ap.parse_args()

    # Load data
    ar_recs     = load_jsonl_by_id(args.ar_file)
    col_recs    = load_jsonl_by_id(args.collab_file)
    ar_correct  = load_eval_correct(args.ar_eval)
    col_correct = load_eval_correct(args.col_eval)

    pairs = find_surgical_pairs(
        ar_recs, col_recs, ar_correct, col_correct,
        max_diff_chars=args.max_diff_chars,
    )
    print(f"[info] {len(pairs)} surgical correction pairs found "
          f"(AR fail → CoCoder pass, diff ≤ {args.max_diff_chars} chars)")

    if not pairs:
        result = {
            "n_pairs": 0,
            "ratio": None,
            "message": "No surgical pairs found; try increasing --max_diff_chars",
        }
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"[done] wrote {args.out}")
        return

    # Load dLLM
    print(f"[info] loading Dream-Coder from {args.model_id} …")
    model = DreamCoder(model_id=args.model_id, device=args.device)

    fault_confs: List[float] = []
    nonfault_confs: List[float] = []
    per_pair: List[Dict[str, Any]] = []

    for rid, ar_code, col_code in tqdm(pairs, desc="scoring pairs"):
        question = ar_recs[rid].get("question", "")
        ar_conf = score_code_confidence(model, question, ar_code, args.dataset)
        if ar_conf is None or len(ar_conf) == 0:
            continue

        fault_idxs, nonfault_idxs = locate_fault_token_indices(
            ar_code, col_code, ar_conf, model
        )
        if not fault_idxs or not nonfault_idxs:
            continue

        mean_fault    = float(ar_conf[fault_idxs].mean().item())
        mean_nonfault = float(ar_conf[nonfault_idxs].mean().item())
        ratio_pair    = mean_nonfault / mean_fault if mean_fault > 0 else float("inf")

        fault_confs.extend(ar_conf[fault_idxs].tolist())
        nonfault_confs.extend(ar_conf[nonfault_idxs].tolist())
        per_pair.append({
            "id":              rid,
            "n_fault_tokens":  len(fault_idxs),
            "n_nonfault_tokens": len(nonfault_idxs),
            "mean_fault_conf":    mean_fault,
            "mean_nonfault_conf": mean_nonfault,
            "ratio":           ratio_pair,
        })

    if not fault_confs:
        result = {
            "n_pairs": len(pairs),
            "n_scored": 0,
            "ratio": None,
            "message": "Could not score any pairs (tokenization or model error)",
        }
    else:
        overall_fault    = sum(fault_confs)    / len(fault_confs)
        overall_nonfault = sum(nonfault_confs) / len(nonfault_confs)
        overall_ratio    = overall_nonfault / overall_fault if overall_fault > 0 else float("inf")

        result = {
            "dataset":             args.dataset,
            "n_pairs_found":       len(pairs),
            "n_pairs_scored":      len(per_pair),
            "n_fault_tokens":      len(fault_confs),
            "n_nonfault_tokens":   len(nonfault_confs),
            "mean_fault_conf":     overall_fault,
            "mean_nonfault_conf":  overall_nonfault,
            "ratio":               overall_ratio,
            "per_pair":            per_pair,
        }
        print(f"\n[result] fault conf = {overall_fault:.4f}, "
              f"non-fault conf = {overall_nonfault:.4f}, "
              f"ratio = {overall_ratio:.1f}×")
        print(f"[compare] text CoT ≈ 1.15×  |  code (HumanEval) ≈ 23×  |  this = {overall_ratio:.1f}×")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
locator_scoring.py — Compare per-token confidence from different model families
                     on "surgical fault pairs" from the remask pipeline.

For each (draft, remasked) pair where the draft failed and the remasked
version passed, we check whether each model assigns *lower* confidence to
the tokens that actually changed (fault tokens) vs the rest.

Scoring methods
---------------
  DLLM     — DreamCoder:  leave-one-out masking  → P(t_i | all others)
  MLM      — CodeBERT:    leave-one-out masking  → P(t_i | all others)
  AR       — DeepSeek-Coder: teacher-forced      → P(t_i | t_0…t_{i-1})

  Note: DLLM and MLM use LOO (each token masked in turn) for maximum
  accuracy in this analysis.  The production locators in coder/locators/
  use faster single-pass approximations; the difference is intentional.

Output
------
  Prints a summary table of mean P(fault token) vs P(non-fault token) per
  model, with the ratio (higher ratio = better fault localisation).

Usage
-----
  python -m coder.analysis.locator_scoring \\
      --remask_dir outputs/base_tuteng \\
      --dataset humaneval \\
      --device cuda
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
)


# ── Data loading ─────────────────────────────────────────────────────────────

@dataclass
class FaultPair:
    task_id:       str
    prompt:        str
    draft:         str    # AR completion that failed eval
    remasked:      str    # dLLM-refined completion that passed eval
    threshold:     float
    ar_model_tag:  str    # e.g. "deepseek", "qwen", "llama31"


def _extract_threshold(stem: str) -> float | None:
    """Parse the τ threshold from a filename stem like *_t0.9* ."""
    m = re.search(r"_t(\d+(?:\.\d+)?)", stem)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _ar_model_tag(stem: str) -> str:
    """Extract the AR-model prefix from a remask filename stem."""
    # Stem format: {ar_model}_dream_remask_{dataset}_t{thresh}[_*]
    if stem.startswith("remask_"):
        # Historical DeepSeek+Dream sweep files live under outputs/remask_kodai
        # and are named remask_{dataset}_t{thresh}.jsonl.
        return "deepseek"
    m = re.match(r"^(.+?)_dream_remask_", stem)
    return m.group(1) if m else "unknown"


def load_fault_pairs(
    remask_dir: str,
    dataset: str,
    max_diff_chars: int = 10,
) -> list[FaultPair]:
    """
    Load surgical (few-character) diff pairs where the draft failed and the
    remasked completion passed eval.

    Scans `remask_dir` for files matching:
        *_dream_remask_{dataset}_t*.jsonl
    (excludes _timed variants to avoid double-counting; uses the base file
    if available, otherwise any non-_timed match).
    """
    # Collect candidate remask JSONL files, deduplicated by (ar_model, threshold).
    remask_root = Path(remask_dir)
    all_files = sorted(glob.glob(str(remask_root / f"*_dream_remask_{dataset}_t*.jsonl")))

    # DeepSeek+Dream's original sweep is stored in outputs/remask_kodai with a
    # shorter historical filename.  Include it when the caller points at the
    # base outputs dir (the command documented in spec_locator_ablation.md).
    if remask_root.name == "base_tuteng":
        kodai_dir = remask_root.parent / "remask_kodai"
        all_files.extend(sorted(glob.glob(str(kodai_dir / f"remask_{dataset}_t*.jsonl"))))
    elif remask_root.name == "remask_kodai":
        all_files.extend(sorted(glob.glob(str(remask_root / f"remask_{dataset}_t*.jsonl"))))

    # Exclude .lock, .timing_summary, backups; also prefer non-_timed/_dedup files.
    def _priority(p: str) -> int:
        stem = Path(p).stem
        if "_timed" in stem or "_dedup" in stem:
            return 1
        return 0

    all_files = [
        f for f in all_files
        if not any(f.endswith(s) for s in (".lock", ".timing_summary.json", ".bak", "backup"))
        and "-sanitized" not in Path(f).stem
    ]
    all_files.sort(key=_priority)

    # Keep only one file per (ar_model_tag, threshold) — the lower-priority one wins
    # (i.e. the clean base file over _timed / _dedup).
    seen_keys: set[tuple[str, float]] = set()
    selected: list[str] = []
    for f in all_files:
        stem = Path(f).stem
        tag = _ar_model_tag(stem)
        thresh = _extract_threshold(stem)
        if thresh is None:
            continue
        key = (tag, thresh)
        if key not in seen_keys:
            seen_keys.add(key)
            selected.append(f)

    pairs: list[FaultPair] = []

    for remask_path in selected:
        stem = Path(remask_path).stem
        thresh = _extract_threshold(stem)
        ar_tag = _ar_model_tag(stem)

        rm_dir = Path(remask_path).parent

        # AR baseline eval: {ar_model}_{dataset}-sanitized_eval_results.json.
        # Prefer remask_dir, then its sibling base_tuteng for remask_kodai.
        ar_eval_candidates = [
            remask_root / f"{ar_tag}_{dataset}-sanitized_eval_results.json",
            remask_root.parent / "base_tuteng" / f"{ar_tag}_{dataset}-sanitized_eval_results.json",
            rm_dir.parent / "base_tuteng" / f"{ar_tag}_{dataset}-sanitized_eval_results.json",
        ]
        ar_eval: dict = {}
        ar_eval_path = next((p for p in ar_eval_candidates if p.exists()), None)
        if ar_eval_path is not None:
            with open(ar_eval_path) as f:
                ar_eval = json.load(f).get("eval", {})

        # Remask eval: try exact stem first, then _dedup fallback.
        rm_eval: dict = {}
        for candidate_stem in [stem, stem + "_dedup"]:
            rm_eval_path = rm_dir / f"{candidate_stem}-sanitized_eval_results.json"
            if rm_eval_path.exists():
                with open(rm_eval_path) as f:
                    rm_eval = json.load(f).get("eval", {})
                break

        with open(remask_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                tid = rec.get("task_id", "")
                draft    = rec.get("draft_completion", "")
                remasked = rec.get("raw_completion", "")

                if not draft or not remasked or draft == remasked:
                    continue

                # Filter to surgical diffs (few characters changed).
                len_diff  = abs(len(draft) - len(remasked))
                minlen    = min(len(draft), len(remasked))
                char_diffs = sum(1 for i in range(minlen) if draft[i] != remasked[i]) + len_diff
                if char_diffs > max_diff_chars:
                    continue

                # Keep only pairs where draft failed and remasked passed.
                def _status(eval_dict: dict, task_id: str) -> str:
                    entry = eval_dict.get(task_id)
                    if not entry:
                        return "unknown"
                    return entry[0].get("plus_status", "unknown")

                if _status(ar_eval, tid) != "fail":
                    continue
                if _status(rm_eval, tid) != "pass":
                    continue

                pairs.append(FaultPair(
                    task_id=tid,
                    prompt=rec.get("prompt", ""),
                    draft=draft,
                    remasked=remasked,
                    threshold=thresh,
                    ar_model_tag=ar_tag,
                ))

    return pairs


def find_diff_char_spans(draft: str, remasked: str) -> list[tuple[int, int]]:
    """Return contiguous character spans in `draft` that differ from `remasked`."""
    spans: list[tuple[int, int]] = []
    minlen = min(len(draft), len(remasked))
    i = 0
    while i < minlen:
        if draft[i] != remasked[i]:
            start = i
            while i < minlen and draft[i] != remasked[i]:
                i += 1
            spans.append((start, i))
        else:
            i += 1
    if len(draft) > minlen:
        spans.append((minlen, len(draft)))
    return spans


# ── Model-specific scoring ────────────────────────────────────────────────────

@torch.inference_mode()
def score_dream_loo(
    model, tokenizer, code: str, prompt: str, device: str,
    batch_size: int = 32, **_kw,
) -> np.ndarray:
    """
    DreamCoder leave-one-out masking.
    For each completion token, mask only that token; read P(token) from logits.
    """
    mask_token_id = model.config.mask_token_id

    messages = [{"role": "user", "content": prompt}]
    prompt_enc = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True,
    )
    prompt_ids = prompt_enc.input_ids.to(device)
    P = prompt_ids.shape[1]

    comp_ids = tokenizer(code, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    M = comp_ids.shape[1]
    if M == 0:
        return np.array([])

    base_ids = torch.cat([prompt_ids, comp_ids], dim=1)  # [1, P+M]
    confidence = torch.zeros(M, device=device)

    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)
        bs = end - start
        batch = base_ids.expand(bs, -1).clone()
        for j in range(bs):
            batch[j, P + start + j] = mask_token_id
        logits = model(batch).logits
        # Left-shift: logits[:, t, :] predicts token at t (DreamCoder convention)
        logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
        for j in range(bs):
            pos = P + start + j
            probs = torch.softmax(logits[j, pos, :].float(), dim=-1)
            confidence[start + j] = probs[comp_ids[0, start + j]]

    return confidence.cpu().numpy()


@torch.inference_mode()
def score_bert_loo(
    model, tokenizer, code: str, prompt: str, device: str,
    batch_size: int = 32, **_kw,
) -> np.ndarray:
    """
    CodeBERT leave-one-out masking.
    Prompt + code concatenated; only code tokens are scored.
    """
    full_text = prompt + code
    enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=True,
                    truncation=True, max_length=512)
    input_ids = enc.input_ids.to(device)
    N = input_ids.shape[1]

    # Locate where code tokens start: after [CLS] + prompt tokens + [SEP]
    prompt_enc = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=512)
    code_start = len(prompt_enc["input_ids"]) - 1   # exclude last [SEP]
    code_end   = N - 1                               # exclude final [SEP]
    M = code_end - code_start
    if M <= 0:
        return np.array([])

    mask_id = tokenizer.mask_token_id
    probs_out = torch.zeros(M, device=device)

    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)
        bs = end - start
        batch = input_ids.expand(bs, -1).clone()
        for j in range(bs):
            batch[j, code_start + start + j] = mask_id
        logits = model(batch).logits
        for j in range(bs):
            pos = code_start + start + j
            probs = torch.softmax(logits[j, pos, :].float(), dim=-1)
            probs_out[start + j] = probs[input_ids[0, pos]]

    return probs_out.cpu().numpy()


@torch.inference_mode()
def score_ar(
    model, tokenizer, code: str, prompt: str, device: str, **_kw,
) -> np.ndarray:
    """
    AR teacher-forced scoring.
    Prompt is formatted with the chat template (matching draft-generation conditions).
    """
    try:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
    except Exception:
        formatted = prompt

    prompt_ids = tokenizer(formatted, return_tensors="pt",
                           add_special_tokens=False).input_ids.to(device)
    comp_ids   = tokenizer(code, return_tensors="pt",
                           add_special_tokens=False).input_ids.to(device)
    P = prompt_ids.shape[1]
    M = comp_ids.shape[1]
    if M == 0:
        return np.array([])

    full_ids = torch.cat([prompt_ids, comp_ids], dim=1)
    logits   = model(full_ids).logits.float()           # [1, P+M, V]

    # logits[:, P+i-1, :] predicts token at position P+i
    comp_logits = logits[0, P - 1 : P + M - 1, :]      # [M, V]
    probs       = torch.softmax(comp_logits, dim=-1)
    confidence  = probs[torch.arange(M, device=device), comp_ids[0]]

    return confidence.cpu().numpy()


SCORERS = {
    "DLLM": score_dream_loo,
    "MLM":  score_bert_loo,
    "AR":   score_ar,
}


# ── Token → character-span mapping ───────────────────────────────────────────

def token_char_spans(tokenizer, code: str, model_key: str, prompt: str = "") -> list[tuple[int, int]]:
    """
    Return (start, end) char offsets in `code` for each scored token.
    Uses offset_mapping where available (fast tokenizers).
    """
    if model_key == "MLM":
        # BERT was scored on prompt+code; code tokens start after prompt tokens.
        full_text = prompt + code
        prompt_len = len(prompt)
        try:
            enc = tokenizer(full_text, add_special_tokens=True,
                            truncation=True, max_length=512,
                            return_offsets_mapping=True)
            spans = []
            for s, e in enc["offset_mapping"]:
                if s == 0 and e == 0:
                    continue  # special token
                if s >= prompt_len:
                    spans.append((s - prompt_len, e - prompt_len))
            return spans
        except Exception:
            pass
        # Fallback: tokenize code standalone
        enc = tokenizer(code, add_special_tokens=False)
        code_toks = enc["input_ids"]
    else:
        try:
            enc = tokenizer(code, add_special_tokens=False, return_offsets_mapping=True)
            return [(int(s), int(e)) for s, e in enc["offset_mapping"]]
        except Exception:
            pass
        enc = tokenizer(code, add_special_tokens=False)
        code_toks = enc["input_ids"]

    # Greedy fallback
    spans: list[tuple[int, int]] = []
    pos = 0
    for tid in code_toks:
        tok_str = tokenizer.decode([tid], skip_special_tokens=False)
        clean = tok_str.replace("\u2581", " ").replace("##", "").replace("\u0120", " ")
        idx = code.find(clean, pos)
        if idx == -1:
            spans.append((pos, pos + max(len(clean), 1)))
            pos += max(len(clean), 1)
        else:
            spans.append((idx, idx + len(clean)))
            pos = idx + len(clean)
    return spans


def fault_token_indices(
    spans: list[tuple[int, int]],
    diff_spans: list[tuple[int, int]],
) -> list[int]:
    """Token indices that overlap with any diff character span."""
    indices = []
    for i, (s, e) in enumerate(spans):
        if any(e > ds and s < de for ds, de in diff_spans):
            indices.append(i)
    return indices


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(
    fault_probs:    dict[str, list[float]],
    nonfault_probs: dict[str, list[float]],
    model_names:    list[str],
    n_pairs:        int,
) -> None:
    print("\n" + "=" * 120)
    print(f"  Fault vs Non-Fault Token Probabilities  ({n_pairs} surgical fault pairs)")
    print("  " + "-" * 110)
    print(f"  {'Model':<10} {'P(fault)':>12} {'±std':>8}  "
          f"{'P(non-fault)':>14} {'±std':>8}  {'ratio':>8}  "
          f"{'n_fault':>8}  {'n_nonfault':>10}")
    print("  " + "-" * 110)
    for m in model_names:
        fp  = fault_probs[m]
        nfp = nonfault_probs[m]
        mf  = float(np.mean(fp))  if fp  else float("nan")
        sf  = float(np.std(fp))   if fp  else float("nan")
        mnf = float(np.mean(nfp)) if nfp else float("nan")
        snf = float(np.std(nfp))  if nfp else float("nan")
        ratio = mnf / mf if mf > 0 and not math.isnan(mf) else float("inf")
        print(f"  {m:<10} {mf:>12.6f} {sf:>8.6f}  "
              f"{mnf:>14.6f} {snf:>8.6f}  {ratio:>8.2f}x "
              f"{len(fp):>8}  {len(nfp):>10}")
    print("=" * 120)


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--remask_dir", default="outputs/base_tuteng",
                   help="Directory with remask JSONL and eval-results files.")
    p.add_argument("--dataset", default="humaneval",
                   choices=["humaneval", "mbpp"])
    p.add_argument("--max_diff_chars", type=int, default=10)
    p.add_argument("--dream_model",    default="Dream-org/Dream-Coder-v0-Instruct-7B")
    p.add_argument("--bert_model",     default="microsoft/codebert-base-mlm")
    p.add_argument("--ar_model",       default="deepseek-ai/deepseek-coder-6.7b-instruct",
                   help="AR model used to generate the drafts (for teacher-forced scoring).")
    p.add_argument("--device",         default="cuda")
    p.add_argument("--threshold", type=float, default=None,
                   help="Only keep remask pairs from this threshold, e.g. 0.9.")
    p.add_argument("--ar_tag", default=None,
                   help="Only keep pairs from this AR draft tag, e.g. deepseek.")
    p.add_argument("--dedupe_task", action="store_true",
                   help="Keep at most one fault pair per task_id after filtering.")
    p.add_argument("--no_dream",  action="store_true")
    p.add_argument("--no_bert",   action="store_true")
    p.add_argument("--no_ar",     action="store_true")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Batch size for LOO masking (DLLM and MLM).")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    pairs = load_fault_pairs(args.remask_dir, args.dataset, args.max_diff_chars)
    if args.threshold is not None:
        pairs = [p for p in pairs if abs(p.threshold - args.threshold) < 1e-9]
    if args.ar_tag is not None:
        pairs = [p for p in pairs if p.ar_model_tag == args.ar_tag]
    if args.dedupe_task:
        deduped: list[FaultPair] = []
        seen: set[str] = set()
        for pair in pairs:
            if pair.task_id in seen:
                continue
            seen.add(pair.task_id)
            deduped.append(pair)
        pairs = deduped

    if not pairs:
        print(f"[warn] no surgical fault pairs found in {args.remask_dir} for {args.dataset}")
        sys.exit(1)

    filters = []
    if args.threshold is not None:
        filters.append(f"threshold={args.threshold}")
    if args.ar_tag is not None:
        filters.append(f"ar_tag={args.ar_tag}")
    if args.dedupe_task:
        filters.append("dedupe_task=True")
    filter_msg = f" ({', '.join(filters)})" if filters else ""
    print(f"Loaded {len(pairs)} surgical fault pairs{filter_msg} "
          f"(max_diff_chars={args.max_diff_chars})")
    for pair in pairs:
        diff_spans = find_diff_char_spans(pair.draft, pair.remasked)
        total_diff = sum(e - s for s, e in diff_spans)
        print(f"  [{pair.ar_model_tag}] {pair.task_id}  τ={pair.threshold}  "
              f"diff={total_diff}c")

    # Load models
    models: dict[str, tuple] = {}
    if not args.no_dream:
        print("\n[model] loading DreamCoder …")
        tok = AutoTokenizer.from_pretrained(args.dream_model, trust_remote_code=True)
        mdl = AutoModel.from_pretrained(
            args.dream_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to(args.device).eval()
        models["DLLM"] = (mdl, tok)

    if not args.no_bert:
        print("[model] loading CodeBERT …")
        tok = AutoTokenizer.from_pretrained(args.bert_model)
        mdl = AutoModelForMaskedLM.from_pretrained(args.bert_model).to(args.device).eval()
        models["MLM"] = (mdl, tok)

    if not args.no_ar:
        print(f"[model] loading AR model: {args.ar_model} …")
        tok = AutoTokenizer.from_pretrained(args.ar_model, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            args.ar_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to(args.device).eval()
        models["AR"] = (mdl, tok)

    if not models:
        print("[error] no models enabled.")
        sys.exit(1)

    model_names = list(models.keys())
    fault_probs:    dict[str, list[float]] = {m: [] for m in model_names}
    nonfault_probs: dict[str, list[float]] = {m: [] for m in model_names}

    print(f"\nScoring {len(pairs)} pairs with {model_names} …\n")
    for pi, pair in enumerate(pairs):
        diff_spans = find_diff_char_spans(pair.draft, pair.remasked)
        total_diff = sum(e - s for s, e in diff_spans)
        print(f"[{pi+1}/{len(pairs)}] {pair.task_id}  τ={pair.threshold}  "
              f"diff={total_diff}c")

        for mname, (model, tokenizer) in models.items():
            scorer = SCORERS[mname]
            conf = scorer(model, tokenizer, pair.draft, pair.prompt,
                          args.device, batch_size=args.batch_size)
            if len(conf) == 0:
                continue

            spans = token_char_spans(tokenizer, pair.draft, mname, pair.prompt)
            f_idx = set(fault_token_indices(spans, diff_spans))

            fp  = [float(conf[i]) for i in f_idx if i < len(conf)]
            nfp = [float(conf[i]) for i in range(len(conf)) if i not in f_idx]

            fault_probs[mname].extend(fp)
            nonfault_probs[mname].extend(nfp)

            mf  = np.mean(fp)  if fp  else float("nan")
            mnf = np.mean(nfp) if nfp else float("nan")
            print(f"  {mname:>5}: P(fault)={mf:.4f} ({len(fp)} tok)  "
                  f"P(non-fault)={mnf:.4f} ({len(nfp)} tok)")

    print_summary(fault_probs, nonfault_probs, model_names, len(pairs))


if __name__ == "__main__":
    main()

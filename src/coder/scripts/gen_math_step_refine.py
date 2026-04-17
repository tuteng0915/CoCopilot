#!/usr/bin/env python3
"""
gen_math_step_refine.py — Step-level truncation + AR (or dLLM) continuation for math.

Pipeline:
  1. Load AR math JSONL (from gen_math.py).
  2. Use a locator (LLaDA by default) to score per-token confidence.
  3. Segment the completion into line-level "steps"; aggregate confidence per step.
  4. Find the worst step (lowest mean confidence, if below --min_step_conf).
  5. Truncate the completion at that step boundary.
  6. Ask the rewriter (AR or LLaDA) to continue from the truncation point.
  7. Combine locked prefix + new continuation and write output JSONL.

The key insight: truncating at a step boundary and letting an AR rewriter
continue ensures causal consistency across CoT steps, unlike token-level
diffusion filling.

Step-selection strategies (--step_strategy):
  min    — always truncate at the step with lowest mean confidence (default)
  first  — truncate at the first step whose mean confidence < --min_step_conf
  last   — truncate at the last step whose mean confidence < --min_step_conf

Output format:
  Compatible with eval_math.py (has id, answer_ref, raw_completion, dataset).
  Also preserves draft_completion for comparison.

Usage:
  # LLaDA locator + Llama-3.1 AR rewriter (Direction B)
  python -m coder.scripts.gen_math_step_refine \\
      --input  outputs/base_tuteng/llama31_gsm8k.jsonl \\
      --out    outputs/math/llama31_llada_step_gsm8k.jsonl \\
      --locator_model llada \\
      --rewriter_model llama31

  # LLaDA as both locator and rewriter (Direction A, step-level)
  python -m coder.scripts.gen_math_step_refine \\
      --input  outputs/base_tuteng/llama31_gsm8k.jsonl \\
      --out    outputs/math/llama31_llada_step_gsm8k_rewrite_llada.jsonl \\
      --locator_model llada \\
      --rewriter_model llada \\
      --rewriter_mode diffusion
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from coder.models import (
    LLaDACoder,
    DreamCoder,
    Llama31Coder,
    QwenCoder,
    MistralCoder,
    DeepSeekCoder,
)
from coder.utils.schema import ModelRequest
from coder.utils.sharding import take_shard, validate_shard_args


# ── Model builders ────────────────────────────────────────────────────────────

_AR_MODELS = {
    "llama31":  (Llama31Coder,   "meta-llama/Llama-3.1-8B-Instruct"),
    "qwen":     (QwenCoder,      "Qwen/Qwen2.5-Coder-7B-Instruct"),
    "mistral":  (MistralCoder,   "mistralai/Mistral-7B-Instruct-v0.3"),
    "deepseek": (DeepSeekCoder,  "deepseek-ai/deepseek-coder-6.7b-instruct"),
}
_DLLM_MODELS = {
    "llada": (LLaDACoder,  "GSAI-ML/LLaDA-8B-Instruct"),
    "dream": (DreamCoder,  "Dream-org/Dream-Coder-v0-Instruct-7B"),
}


def build_ar_model(name: str, model_id: Optional[str], device: str):
    cls, default_id = _AR_MODELS[name]
    return cls(model_id=model_id or default_id, device=device)


def build_dllm_model(name: str, model_id: Optional[str], device: str):
    cls, default_id = _DLLM_MODELS[name]
    return cls(model_id=model_id or default_id, device=device)


# ── Locator: LLaDA single forward pass ───────────────────────────────────────

def _apply_chat_template(tokenizer, prompt_text: str) -> str:
    try:
        messages = [{"role": "user", "content": prompt_text}]
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
    except Exception:
        return prompt_text


@torch.inference_mode()
def score_completion_by_line(
    locator_model,
    locator_tok,
    prompt_text: str,
    completion: str,
    device: str,
) -> list[tuple[int, str, float]]:
    """
    Score tokens with a single dLLM forward pass; return per-line mean confidence.

    Returns list of (line_idx, line_text, mean_conf) for non-empty lines only.
    """
    formatted = _apply_chat_template(locator_tok, prompt_text)
    prompt_ids = locator_tok(
        formatted, add_special_tokens=False, return_tensors="pt",
    ).input_ids.to(device)

    comp_enc = locator_tok(
        completion, add_special_tokens=False,
        return_offsets_mapping=True, return_tensors="pt",
    )
    comp_ids = comp_enc.input_ids.to(device)
    M = comp_ids.shape[1]
    if M == 0:
        return []

    full_ids = torch.cat([prompt_ids, comp_ids], dim=1)
    logits = locator_model(full_ids).logits
    comp_logits = logits[0, prompt_ids.shape[1]:, :].float()
    probs = torch.softmax(comp_logits, dim=-1)
    conf = probs[torch.arange(M, device=device), comp_ids[0]].cpu().numpy()

    # Map tokens to lines via char offsets
    offsets = comp_enc.get("offset_mapping")
    if offsets is not None:
        char_spans = [(int(s), int(e)) for s, e in offsets[0].tolist()]
    else:
        char_spans = [(i, i + 1) for i in range(M)]

    lines = completion.split("\n")
    line_end_char: list[int] = []
    pos = 0
    for line in lines:
        pos += len(line) + 1
        line_end_char.append(pos - 1)

    def char_to_line(c: int) -> int:
        for li, le in enumerate(line_end_char):
            if c <= le:
                return li
        return len(line_end_char) - 1

    line_confs: dict[int, list[float]] = {}
    for tok_i, (cs, _) in enumerate(char_spans):
        li = char_to_line(cs)
        line_confs.setdefault(li, []).append(float(conf[tok_i]))

    result = []
    for li, line_text in enumerate(lines):
        if not line_text.strip():
            continue
        lc = line_confs.get(li, [])
        if not lc:
            continue
        result.append((li, line_text, float(np.mean(lc))))

    return result


def find_truncation_line(
    step_scores: list[tuple[int, str, float]],
    strategy: str,
    min_step_conf: float,
) -> Optional[int]:
    """
    Find the line_idx to truncate at (i.e. lines before this index are kept).

    Returns None if no truncation should happen (all steps are confident).
    """
    if not step_scores:
        return None

    if strategy == "min":
        # Always truncate at lowest-confidence step
        worst_pos = int(np.argmin([c for _, _, c in step_scores]))
        return step_scores[worst_pos][0]

    elif strategy == "first":
        for li, _, conf in step_scores:
            if conf < min_step_conf:
                return li
        return None

    elif strategy == "last":
        last = None
        for li, _, conf in step_scores:
            if conf < min_step_conf:
                last = li
        return last

    else:
        raise ValueError(f"Unknown step_strategy: {strategy}")


def truncate_and_get_prefix_text(completion: str, truncation_line_idx: int) -> str:
    """Return the completion text up to (not including) truncation_line_idx."""
    lines = completion.split("\n")
    kept = lines[:truncation_line_idx]
    prefix = "\n".join(kept)
    if prefix and not prefix.endswith("\n"):
        prefix += "\n"
    return prefix


def build_continuation_prompt(original_prompt: str, locked_prefix: str) -> str:
    """
    Combine the original prompt with the locked steps to form the continuation context.

    The resulting prompt is passed to the AR model's generate() method, which
    applies the chat template and asks the model to produce the remainder of the
    solution.
    """
    if locked_prefix.strip():
        return original_prompt.rstrip() + "\n" + locked_prefix
    return original_prompt


# ── AR continuation ───────────────────────────────────────────────────────────

def ar_continue(ar_model, prompt: str, max_new_tokens: int, seed: int) -> str:
    req = ModelRequest(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        seed=seed,
    )
    return ar_model.generate(req)


def dllm_continue(
    dllm_model,
    prompt: str,
    locked_prefix: str,
    max_new_tokens: int,
    seed: int,
) -> str:
    """
    Use the dLLM (LLaDA/Dream) to continue from the locked prefix.
    The dLLM generates fresh (no masking), conditioned on the full context.
    """
    req = ModelRequest(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        seed=seed,
    )
    return dllm_model.generate(req)


# ── JSONL helpers ─────────────────────────────────────────────────────────────

def read_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input",   required=True, help="AR math JSONL from gen_math.py")
    ap.add_argument("--out",     required=True, help="Output JSONL path")

    ap.add_argument(
        "--locator_model",
        choices=list(_DLLM_MODELS.keys()),
        default="llada",
        help="dLLM used to score steps (default: llada)",
    )
    ap.add_argument("--locator_model_id", default=None,
                    help="Override HuggingFace ID for locator.")

    ap.add_argument(
        "--rewriter_model",
        choices=list(_AR_MODELS.keys()) + list(_DLLM_MODELS.keys()),
        default="llama31",
        help="Model that continues from truncation point (default: llama31).",
    )
    ap.add_argument("--rewriter_model_id", default=None,
                    help="Override HuggingFace ID for rewriter.")
    ap.add_argument(
        "--rewriter_mode",
        choices=["ar", "diffusion"],
        default="ar",
        help="Use AR autoregressive continuation (default) or dLLM generation.",
    )

    ap.add_argument(
        "--step_strategy",
        choices=["min", "first", "last"],
        default="min",
        help="How to pick the truncation step: min (lowest-conf), "
             "first/last (first/last step below --min_step_conf).",
    )
    ap.add_argument("--min_step_conf", type=float, default=0.5,
                    help="Confidence threshold for first/last strategies.")
    ap.add_argument("--skip_if_no_low_conf", action="store_true",
                    help="For first/last strategies: keep draft if all steps "
                         "are above --min_step_conf (no truncation).")

    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--seed",           type=int, default=3407)
    ap.add_argument("--device",         default="cuda")
    ap.add_argument("--num_shards",     type=int, default=1)
    ap.add_argument("--shard_idx",      type=int, default=0)
    ap.add_argument("--resume",         action="store_true")
    args = ap.parse_args()

    try:
        validate_shard_args(num_shards=args.num_shards, shard_idx=args.shard_idx)
    except ValueError as e:
        ap.error(str(e))

    # Validate rewriter_mode consistency
    if args.rewriter_mode == "ar" and args.rewriter_model not in _AR_MODELS:
        ap.error(f"--rewriter_model {args.rewriter_model} is a dLLM; "
                 f"set --rewriter_mode diffusion")
    if args.rewriter_mode == "diffusion" and args.rewriter_model not in _DLLM_MODELS:
        ap.error(f"--rewriter_model {args.rewriter_model} is an AR model; "
                 f"set --rewriter_mode ar")

    # Load records
    records = list(read_jsonl(args.input))
    records = take_shard(records, args.num_shards, args.shard_idx)
    print(f"[data] {len(records)} records")

    # Resume
    done_ids: set[str] = set()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and out_path.exists():
        for rec in read_jsonl(str(out_path)):
            done_ids.add(rec.get("id", ""))
        print(f"[resume] skipping {len(done_ids)} done records")

    # Load locator (always a dLLM)
    print(f"\n[locator] loading {args.locator_model} …")
    loc_cls, loc_default = _DLLM_MODELS[args.locator_model]
    loc_model_id = args.locator_model_id or loc_default
    loc_tok = AutoTokenizer.from_pretrained(loc_model_id, trust_remote_code=True)
    loc_model = AutoModel.from_pretrained(
        loc_model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(args.device).eval()
    print(f"[locator] {loc_model_id} ready")

    # Load rewriter
    print(f"[rewriter] loading {args.rewriter_model} (mode={args.rewriter_mode}) …")
    if args.rewriter_mode == "ar":
        rewriter = build_ar_model(args.rewriter_model, args.rewriter_model_id, args.device)
    else:
        rewriter = build_dllm_model(args.rewriter_model, args.rewriter_model_id, args.device)
    print(f"[rewriter] ready")

    timing_total: list[float] = []
    n_truncated = 0
    n_written = 0

    with open(out_path, "a" if args.resume else "w", encoding="utf-8") as fout:
        for rec in tqdm(records, desc="step_refine"):
            rec_id = rec.get("id", "")
            if rec_id in done_ids:
                continue

            prompt_text = rec.get("prompt", "")
            draft = rec.get("raw_completion", "")
            if not draft:
                continue

            t0 = time.perf_counter()

            # Score steps with locator
            step_scores = score_completion_by_line(
                loc_model, loc_tok, prompt_text, draft, args.device,
            )

            # Find truncation point
            trunc_line = find_truncation_line(
                step_scores, args.step_strategy, args.min_step_conf,
            )

            if trunc_line is None or (args.skip_if_no_low_conf and trunc_line is None):
                # No confident low step found — keep draft unchanged
                refined = draft
                locked_prefix = draft
                truncated = False
            elif trunc_line == 0:
                # Worst step is the first line — re-generate everything
                locked_prefix = ""
                cont_prompt = prompt_text
                if args.rewriter_mode == "ar":
                    refined = ar_continue(rewriter, cont_prompt, args.max_new_tokens, args.seed)
                else:
                    refined = dllm_continue(rewriter, cont_prompt, locked_prefix,
                                            args.max_new_tokens, args.seed)
                truncated = True
            else:
                locked_prefix = truncate_and_get_prefix_text(draft, trunc_line)
                cont_prompt = build_continuation_prompt(prompt_text, locked_prefix)
                if args.rewriter_mode == "ar":
                    cont = ar_continue(rewriter, cont_prompt, args.max_new_tokens, args.seed)
                else:
                    cont = dllm_continue(rewriter, cont_prompt, locked_prefix,
                                         args.max_new_tokens, args.seed)
                # Remove the locked prefix echo if the model repeated it
                cont_stripped = cont
                if cont.startswith(locked_prefix):
                    cont_stripped = cont[len(locked_prefix):]
                refined = locked_prefix + cont_stripped
                truncated = True

            t1 = time.perf_counter()
            timing_total.append(t1 - t0)
            if truncated:
                n_truncated += 1

            out_rec = dict(rec)
            out_rec.update({
                "draft_completion": draft,
                "raw_completion":   refined,
                "locked_prefix":    locked_prefix if truncated else "",
                "truncation_line":  trunc_line if truncated else -1,
                "truncated":        truncated,
                "gen": {
                    "locator":         args.locator_model,
                    "locator_id":      loc_model_id,
                    "rewriter":        args.rewriter_model,
                    "rewriter_mode":   args.rewriter_mode,
                    "step_strategy":   args.step_strategy,
                    "min_step_conf":   args.min_step_conf,
                    "timing": {"total_s": t1 - t0},
                },
            })
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_written += 1

    avg_t = float(np.mean(timing_total)) if timing_total else 0.0
    print(f"\n[done] wrote {out_path}  ({n_written} records)")
    print(f"[stats] truncated {n_truncated}/{n_written} "
          f"({100*n_truncated/max(n_written,1):.1f}%),  "
          f"avg {avg_t:.1f}s/sample")

    # Write timing summary
    timing_path = out_path.with_suffix(out_path.suffix + ".timing_summary.json")
    timing_path.write_text(json.dumps({
        "script": "gen_math_step_refine",
        "out": str(out_path.resolve()),
        "n_records_written": n_written,
        "n_truncated": n_truncated,
        "locator": args.locator_model,
        "rewriter": args.rewriter_model,
        "rewriter_mode": args.rewriter_mode,
        "timing": {
            "total_s": float(sum(timing_total)),
            "generate_s_avg": avg_t,
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[timing] wrote {timing_path}")


if __name__ == "__main__":
    main()

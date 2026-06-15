#!/usr/bin/env python3
"""
gen_locate_ar_rewrite.py — Ablation: diffusion locator + AR targeted rewrite.

Idea:
  1) Take an AR draft (completion string).
  2) Use DreamCoder forward pass to score per-token confidence for the draft.
  3) Replace low-confidence token spans with a single `<MASK>` marker.
  4) Ask an AR model to rewrite the code by filling the `<MASK>` parts,
     while keeping the unmasked parts unchanged.

This is NOT hard-constrained editing; it's a lightweight ablation baseline that
uses diffusion only as a "localization" signal.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from coder.models import (
    ApiCoder,
    CodeLlamaCoder,
    DeepSeekCoder,
    QwenCoder,
    Qwen35Coder,
    LLaDACoder,
    StarCoder2Coder,
    MistralCoder,
    Llama31Coder,
    DiffuLLaMACoder,
    SeedDiffCoder,
    SeedCoder,
    DreamCoder,
    CoderModel,
)
from coder.locators import get_token_char_spans
from coder.utils.code_cleaning import build_evalplus_solution
from coder.utils.schema import ModelRequest


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_records_by_task_id(path: str) -> dict[str, Dict[str, Any]]:
    records: dict[str, Dict[str, Any]] = {}
    for rec in read_jsonl(path):
        task_id = rec.get("task_id")
        if isinstance(task_id, str) and task_id:
            records[task_id] = rec
    return records


def build_ar_model(name: str, device: str, model_id: Optional[str]) -> CoderModel:
    name = name.lower()
    if name in ["deepseek", "deepseek_coder", "ds"]:
        return DeepSeekCoder(model_id=model_id or "deepseek-ai/deepseek-coder-6.7b-instruct", device=device)
    if name in ["qwen", "qwen_coder"]:
        return QwenCoder(model_id=model_id or "Qwen/Qwen2.5-Coder-7B-Instruct", device=device)
    if name in ["qwen35", "qwen35_coder", "qwen3.5"]:
        return Qwen35Coder(model_id=model_id or "Qwen/Qwen3.5-4B", device=device)
    if name in ["llada", "llada_coder"]:
        return LLaDACoder(model_id=model_id or "GSAI-ML/LLaDA-8B-Instruct", device=device)
    if name in ["starcoder2", "starcoder2_coder", "sc2"]:
        return StarCoder2Coder(model_id=model_id or "bigcode/starcoder2-7b", device=device)
    if name in ["mistral", "mistral_coder"]:
        return MistralCoder(model_id=model_id or "mistralai/Mistral-7B-Instruct-v0.3", device=device)
    if name in ["codellama", "codellama_coder"]:
        return CodeLlamaCoder(model_id=model_id or "codellama/CodeLlama-7b-Instruct-hf", device=device)
    if name in ["llama31", "llama31_coder", "llama3.1"]:
        return Llama31Coder(model_id=model_id or "meta-llama/Llama-3.1-8B-Instruct", device=device)
    if name in ["diffullama", "diffullama_coder", "dflm"]:
        return DiffuLLaMACoder(model_id=model_id, device=device)
    if name in ["seed-diffcoder", "seed_diffcoder", "seeddiffcoder"]:
        return SeedDiffCoder(model_id=model_id, device=device)
    if name in ["seed-coder", "seed_coder", "seedcoder"]:
        return SeedCoder(model_id=model_id, device=device)
    if name in ["api", "api_coder", "closed_api"]:
        return ApiCoder(model_id=model_id, device="api")
    raise ValueError(f"Unknown --ar_model: {name}")


def mask_low_confidence_spans(
    draft: str,
    tokenizer,
    mask_pos: torch.Tensor,
    mask_token: str = "<MASK>",
) -> str:
    """
    Build a masked string from original character spans, collapsing consecutive
    masked tokens into a single `<MASK>` marker.

    Do not reconstruct text by decoding individual tokenizer IDs. Some
    tokenizers expose byte-level marker characters (for example "Ġ"/"Ċ") when
    tokens are decoded one-by-one, which pollutes the AR rewrite prompt.
    """
    spans = get_token_char_spans(tokenizer, draft)
    mp = mask_pos.detach().cpu().bool().tolist()
    out_parts: list[str] = []
    in_mask = False
    mask_had_newline = False
    cursor = 0

    for (start, end), m in zip(spans, mp):
        start = max(0, min(int(start), len(draft)))
        end = max(start, min(int(end), len(draft)))
        if m:
            if not in_mask:
                out_parts.append(draft[cursor:start])
                out_parts.append(mask_token)
                in_mask = True
                mask_had_newline = False
            if "\n" in draft[start:end]:
                mask_had_newline = True
            cursor = max(cursor, end)
            continue
        if in_mask and mask_had_newline:
            out_parts.append("\n")
        in_mask = False
        out_parts.append(draft[cursor:end])
        cursor = max(cursor, end)

    if in_mask and mask_had_newline:
        out_parts.append("\n")
    out_parts.append(draft[cursor:])

    return "".join(out_parts)


def expand_mask_span_level(mask_pos: torch.Tensor, span_merge_gap: int) -> torch.Tensor:
    gap = max(int(span_merge_gap), 0)
    if gap <= 0:
        return mask_pos
    mp = mask_pos.clone()
    idx = torch.nonzero(mp, as_tuple=False).view(-1)
    if idx.numel() == 0:
        return mp
    for a, b in zip(idx[:-1], idx[1:]):
        d = int(b.item() - a.item())
        if 1 < d <= gap + 1:
            mp[a + 1 : b] = True
    return mp


def expand_mask_line_level(draft: str, tokenizer, mask_pos: torch.Tensor) -> torch.Tensor:
    spans = get_token_char_spans(tokenizer, draft)
    mask_flags = mask_pos.detach().cpu().bool().tolist()
    line_ranges: list[tuple[int, int]] = []
    for (start, end), m in zip(spans, mask_flags):
        if not m:
            continue
        start = max(0, min(int(start), len(draft)))
        end = max(start, min(int(end), len(draft)))
        line_start = draft.rfind("\n", 0, start) + 1
        next_newline = draft.find("\n", end)
        line_end = len(draft) if next_newline == -1 else next_newline + 1
        line_ranges.append((line_start, line_end))

    if not line_ranges:
        return mask_pos

    expanded = []
    for start, end in spans:
        start = max(0, min(int(start), len(draft)))
        end = max(start, min(int(end), len(draft)))
        expanded.append(any(line_end > start and line_start < end for line_start, line_end in line_ranges))
    return torch.tensor(expanded, device=mask_pos.device, dtype=torch.bool)


def infer_mask_pos_from_masked_draft(
    draft: str,
    masked_draft: str,
    tokenizer,
    mask_token: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover token mask positions from a precomputed masked draft string."""
    spans = get_token_char_spans(tokenizer, draft)
    intervals: list[tuple[int, int]] = []
    cursor = 0
    parts = masked_draft.split(mask_token)

    for idx, literal in enumerate(parts):
        if idx == 0:
            if literal:
                found = draft.find(literal, cursor)
                if found > cursor:
                    intervals.append((cursor, found))
                cursor = (found + len(literal)) if found >= 0 else cursor
            continue

        if literal:
            found = draft.find(literal, cursor)
            if found < 0:
                found = cursor
            if found > cursor:
                intervals.append((cursor, found))
            cursor = found + len(literal)

    if masked_draft.endswith(mask_token) and cursor < len(draft):
        intervals.append((cursor, len(draft)))

    mask_flags = []
    for start, end in spans:
        start = max(0, min(int(start), len(draft)))
        end = max(start, min(int(end), len(draft)))
        mask_flags.append(any(mask_end > start and mask_start < end for mask_start, mask_end in intervals))

    comp_ids = tokenizer(draft, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    mask_pos = torch.tensor(mask_flags, device=device, dtype=torch.bool)
    return comp_ids, mask_pos


def compute_mask_positions(
    locator: DreamCoder,
    prompt: str,
    draft: str,
    confidence_threshold: float,
    mask_ratio: float | None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (comp_ids, confidence, mask_pos) on locator device.
    """
    messages = [{"role": "user", "content": prompt}]
    enc = locator.tok.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    )
    prompt_ids = enc.input_ids.to(locator.device)

    comp_enc = locator.tok(draft, return_tensors="pt", add_special_tokens=False)
    comp_ids = comp_enc.input_ids.to(locator.device)
    M = comp_ids.shape[1]
    if M == 0:
        conf = torch.zeros(0, device=locator.device)
        mask_pos = torch.zeros(0, device=locator.device, dtype=torch.bool)
        return comp_ids, conf, mask_pos

    confidence = locator.score_tokens(prompt_ids, comp_ids)

    if mask_ratio is not None:
        k = max(1, int(M * mask_ratio))
        threshold_val = torch.kthvalue(confidence, k).values.item()
        mask_pos = confidence <= threshold_val
    else:
        mask_pos = confidence < confidence_threshold

    return comp_ids, confidence, mask_pos


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ablation: use DreamCoder to locate low-confidence spans, then ask an AR model to rewrite masked spans."
    )
    ap.add_argument("--input", required=True, help="Input JSONL from AR generation (EvalPlus or LiveBench).")
    ap.add_argument("--out", required=True, help="Output JSONL path for rewritten samples.")

    ap.add_argument("--locator_model_id", default="Dream-org/Dream-Coder-v0-Instruct-7B")
    ap.add_argument("--locator_device", default="cuda")
    ap.add_argument(
        "--mask_source",
        default=None,
        help=(
            "Optional JSONL from a previous locate+rewrite run. When provided, "
            "reuse its token-level masked_draft decisions instead of rescoring "
            "with the locator model; useful for expanding token masks to span/line."
        ),
    )

    ap.add_argument("--ar_model", required=True, help="AR backbone to do rewrite (e.g., deepseek|qwen|api|llama31).")
    ap.add_argument(
        "--ar_model_id",
        "--model_id",
        dest="ar_model_id",
        default=None,
        help="Override AR HuggingFace model id / API model id.",
    )
    ap.add_argument("--ar_device", default="cuda")

    # Masking policy (use exactly one)
    ap.add_argument("--confidence_threshold", type=float, default=None)
    ap.add_argument("--mask_ratio", type=float, default=None)

    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of locate+rewrite rounds. Default 1 keeps old behavior.",
    )

    ap.add_argument("--mask_token", default="<MASK>")
    ap.add_argument(
        "--mask_granularity",
        choices=["token", "span", "line"],
        default="span",
        help="How to convert token-level low-confidence flags into masks in text: token|span|line. "
             "Default span (collapses consecutive masks).",
    )
    ap.add_argument(
        "--span_merge_gap",
        type=int,
        default=0,
        help="For --mask_granularity span: also mask gaps of <= this many tokens between masked tokens.",
    )
    ap.add_argument("--resume", action="store_true", help="Skip task_ids already present in --out (append mode).")
    args = ap.parse_args()

    if args.rounds < 1:
        ap.error("--rounds must be >= 1")

    if args.confidence_threshold is None and args.mask_ratio is None:
        args.confidence_threshold = 0.5
    if args.confidence_threshold is not None and args.mask_ratio is not None:
        ap.error("Specify at most one of --confidence_threshold and --mask_ratio.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if args.resume and out_path.exists():
        for rec in read_jsonl(str(out_path)):
            tid = rec.get("task_id")
            if isinstance(tid, str) and tid:
                done_ids.add(tid)
        print(f"[resume] skipping {len(done_ids)} already-done tasks.")

    mask_source_by_id = load_records_by_task_id(args.mask_source) if args.mask_source else {}
    if args.mask_source and args.rounds != 1:
        ap.error("--mask_source currently supports --rounds 1 only.")

    if args.mask_source:
        locator = None
        locator_tok = AutoTokenizer.from_pretrained(args.locator_model_id, trust_remote_code=True)
        if locator_tok.pad_token is None and locator_tok.eos_token is not None:
            locator_tok.pad_token = locator_tok.eos_token
        print(f"[mask_source] loaded {len(mask_source_by_id)} records from {args.mask_source}")
    else:
        locator = DreamCoder(model_id=args.locator_model_id, device=args.locator_device)
        locator_tok = locator.tok
    ar_model = build_ar_model(args.ar_model, device=args.ar_device, model_id=args.ar_model_id)

    records = list(read_jsonl(args.input))
    print(f"[info] {len(records)} records loaded from {args.input}")

    t_total0 = time.perf_counter()
    timing_locate_s: list[float] = []
    timing_maskbuild_s: list[float] = []
    timing_rewrite_s: list[float] = []
    n_records_written = 0

    with out_path.open("a", encoding="utf-8") as fout:
        for rec in tqdm(records, desc=f"locate+rewrite(locator={args.locator_model_id}, ar={ar_model.name})"):
            task_id = rec.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                continue
            if task_id in done_ids:
                continue

            is_livebench = "question_id" in rec or str(task_id).startswith("LiveBench/") or str(task_id).startswith("LiveCodeBench/")
            prompt_text: str = rec.get("prompt", "")
            cur = rec.get("raw_completion", "") or rec.get("solution", "")
            gran = (args.mask_granularity or "span").lower().strip()
            if gran not in ("token", "span", "line"):
                raise ValueError(f"Unsupported --mask_granularity: {args.mask_granularity}")

            rounds_trace: list[Dict[str, Any]] = []
            last_comp_ids = None
            last_confidence = None
            last_mask_pos = None
            last_masked_draft = None

            for r in range(int(args.rounds)):
                t_loc0 = time.perf_counter()
                if args.mask_source:
                    src_rec = mask_source_by_id.get(task_id)
                    if not src_rec:
                        print(f"[warn] {task_id}: missing --mask_source record; leaving draft unmasked.")
                        comp_ids = locator_tok(cur, return_tensors="pt", add_special_tokens=False).input_ids.to(args.ar_device)
                        mask_pos = torch.zeros(comp_ids.shape[1], device=args.ar_device, dtype=torch.bool)
                    else:
                        source_draft = (
                            src_rec.get("draft_completion")
                            or ((src_rec.get("rounds_trace") or [{}])[0].get("input_draft"))
                            or cur
                        )
                        if source_draft != cur:
                            cur = source_draft
                        comp_ids, mask_pos = infer_mask_pos_from_masked_draft(
                            draft=cur,
                            masked_draft=src_rec.get("masked_draft", cur),
                            tokenizer=locator_tok,
                            mask_token=args.mask_token,
                            device=args.ar_device,
                        )
                    confidence = torch.ones(mask_pos.numel(), device=args.ar_device)
                else:
                    comp_ids, confidence, mask_pos = compute_mask_positions(
                        locator=locator,
                        prompt=prompt_text,
                        draft=cur,
                        confidence_threshold=float(args.confidence_threshold),
                        mask_ratio=args.mask_ratio,
                    )
                t_loc1 = time.perf_counter()

                t_mb0 = time.perf_counter()
                masked_draft = cur
                mp = mask_pos
                if comp_ids.shape[1] > 0 and mp.numel() > 0 and bool(mp.any().item()):
                    if gran == "span":
                        mp = expand_mask_span_level(mask_pos=mp, span_merge_gap=args.span_merge_gap)
                    elif gran == "line":
                        mp = expand_mask_line_level(draft=cur, tokenizer=locator_tok, mask_pos=mp)

                    masked_draft = mask_low_confidence_spans(
                        draft=cur,
                        tokenizer=locator_tok,
                        mask_pos=mp,
                        mask_token=args.mask_token,
                    )
                t_mb1 = time.perf_counter()

                rewrite_prompt = (
                    "You are given a Python code draft with some spans replaced by the token "
                    f"{args.mask_token!r}.\n"
                    "Rewrite the code by filling in ONLY the masked spans. "
                    "Keep all unmasked text unchanged as much as possible.\n"
                    "Output ONLY valid Python code (no explanations).\n\n"
                    "[Problem]\n"
                    f"{prompt_text}\n\n"
                    "[Draft with masks]\n"
                    f"{masked_draft}\n"
                )

                ar_req = ModelRequest(
                    prompt=rewrite_prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=args.seed,
                )

                try:
                    t_rw0 = time.perf_counter()
                    nxt = ar_model.generate(ar_req)
                    t_rw1 = time.perf_counter()
                except Exception as e:
                    print(f"[warn] {task_id}: round={r} AR rewrite failed ({e}); stopping further rounds.")
                    nxt = cur
                    t_rw1 = time.perf_counter()
                    t_rw0 = t_rw1

                timing_locate_s.append(t_loc1 - t_loc0)
                timing_maskbuild_s.append(t_mb1 - t_mb0)
                timing_rewrite_s.append(t_rw1 - t_rw0)

                rounds_trace.append(
                    {
                        "round": r,
                        "input_draft": cur,
                        "masked_draft": masked_draft,
                        "n_masked_tokens": int(mp.sum().item()) if mp.numel() > 0 else 0,
                        "n_total_tokens": int(confidence.numel()),
                        "output": nxt,
                        "timing": {
                            "locate_s": t_loc1 - t_loc0,
                            "mask_build_s": t_mb1 - t_mb0,
                            "rewrite_s": t_rw1 - t_rw0,
                            "total_s": (t_loc1 - t_loc0) + (t_mb1 - t_mb0) + (t_rw1 - t_rw0),
                        },
                    }
                )

                last_comp_ids = comp_ids
                last_confidence = confidence
                last_mask_pos = mp
                last_masked_draft = masked_draft
                cur = nxt

            rewritten = cur

            if is_livebench:
                solution = rewritten
            else:
                solution = build_evalplus_solution(prompt_text, rewritten)

            out_rec: Dict[str, Any] = {
                "task_id": task_id,
                "prompt": prompt_text,
                "draft_completion": rec.get("raw_completion", "") or rec.get("solution", ""),
                "masked_draft": last_masked_draft if last_masked_draft is not None else (rec.get("raw_completion", "") or rec.get("solution", "")),
                "raw_completion": rewritten,
                "solution": solution,
                "model": f"locate_ar_rewrite::locator={args.locator_model_id}::ar={ar_model.name}",
                "gen": {
                    "source_model": rec.get("model", "unknown"),
                    "locator_model": args.locator_model_id,
                    "mask_source": args.mask_source,
                    "ar_model": ar_model.name,
                    "confidence_threshold": args.confidence_threshold,
                    "mask_ratio": args.mask_ratio,
                    "mask_token": args.mask_token,
                    "mask_granularity": args.mask_granularity,
                    "span_merge_gap": args.span_merge_gap,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "seed": args.seed,
                    "rounds": args.rounds,
                    "n_masked_tokens": int(last_mask_pos.sum().item()) if (last_mask_pos is not None and last_mask_pos.numel() > 0) else 0,
                    "n_total_tokens": int(last_confidence.numel()) if last_confidence is not None else 0,
                    "timing": {
                        "locate_s_total": float(sum([x.get("timing", {}).get("locate_s", 0.0) for x in rounds_trace])),
                        "mask_build_s_total": float(sum([x.get("timing", {}).get("mask_build_s", 0.0) for x in rounds_trace])),
                        "rewrite_s_total": float(sum([x.get("timing", {}).get("rewrite_s", 0.0) for x in rounds_trace])),
                        "total_s": float(sum([x.get("timing", {}).get("total_s", 0.0) for x in rounds_trace])),
                    },
                },
                "rounds_trace": rounds_trace,
            }

            if is_livebench:
                if "question_id" in rec:
                    out_rec["question_id"] = rec["question_id"]
                if "meta" in rec:
                    out_rec["meta"] = rec["meta"]
                if "benchmark" in rec:
                    out_rec["benchmark"] = rec["benchmark"]

            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_records_written += 1

    t_total1 = time.perf_counter()
    timing_path = str(out_path) + ".timing_summary.json"
    summary = {
        "script": "gen_locate_ar_rewrite",
        "out": str(out_path.resolve()),
        "locator_model": args.locator_model_id,
        "ar_model": ar_model.name,
        "rounds": args.rounds,
        "n_records_written": n_records_written,
        "timing": {
            "total_s": t_total1 - t_total0,
            "locate_s_total": float(sum(timing_locate_s)),
            "mask_build_s_total": float(sum(timing_maskbuild_s)),
            "rewrite_s_total": float(sum(timing_rewrite_s)),
            "locate_s_avg": (float(sum(timing_locate_s)) / len(timing_locate_s)) if timing_locate_s else None,
            "rewrite_s_avg": (float(sum(timing_rewrite_s)) / len(timing_rewrite_s)) if timing_rewrite_s else None,
        },
    }
    out_path.with_suffix(out_path.suffix + ".timing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[timing] wrote {timing_path}")

    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()

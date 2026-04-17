#!/usr/bin/env python3
"""
gen_remask.py — Refine AR drafts using DreamCoder/LLaDA token remasking.

Pipeline:
  1. Load AR JSONL outputs (prompt + raw_completion per task).
     - EvalPlus: from scripts/gen_evalplus.py
       fields: task_id, prompt, raw_completion, solution, model, gen
     - LiveBench-Coding: from scripts/gen_livebench.py
       fields: task_id, question_id, prompt, raw_completion, solution, model, gen, meta
     - Math (GSM8K/MATH-500): from scripts/gen_math.py
       fields: id, sample_id, question, prompt, answer_ref, raw_completion, dataset, model, gen
  2. For each task, score each completion token with DreamCoder's forward pass.
  3. Mask low-confidence tokens (by threshold or by ratio).
  4. Regenerate only the masked positions via DreamCoder diffusion.
  5. Write refined JSONL (format mirrors the input schema).

Output JSONL (EvalPlus):
  task_id, prompt, draft_completion, raw_completion, solution, model, gen

Output JSONL (LiveBench):
  task_id, question_id, prompt, draft_completion, raw_completion, solution, model, gen, meta

Output JSONL (Math):
  id, sample_id, question, prompt, answer_ref, dataset, draft_completion, raw_completion, model, gen
  (subject, level preserved for MATH-500; compatible with eval_math.py)
"""
from __future__ import annotations

import argparse
import fcntl
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from coder.locators import (
    align_confidence_to_spans,
    apply_masking_policy,
    build_locator,
    get_token_char_spans,
)
from coder.models import DreamCoder, LLaDACoder, DreamGeneral
from coder.utils.code_cleaning import (
    build_prompt_scaffold_solution,
    clean_model_completion,
    indent_as_body,
)
from coder.utils.schema import ModelRequest
from coder.utils.sharding import take_shard, validate_shard_args


def build_evalplus_solution(prompt_text: str, gen: str) -> str:
    """Build a full EvalPlus-compatible solution string from a prompt + completion."""
    g = clean_model_completion(gen, prompt_text).lstrip()
    prompt = prompt_text.rstrip()

    def extract_prompt_imports(p: str) -> str:
        imports = []
        for line in p.splitlines():
            s = line.strip()
            if s.startswith("from ") or s.startswith("import "):
                imports.append(line.rstrip())
                continue
            if s.startswith("def ") or s.startswith("class "):
                break
        return "\n".join(imports).strip()

    def infer_target_func_name(p: str) -> str | None:
        m = re.search(r"(?m)^\s*def\s+([A-Za-z_]\w*)\s*\(", p)
        return m.group(1) if m else None

    def extract_single_function(src: str, func_name: str) -> str | None:
        pattern = (
            rf"(?ms)^(?P<decor>(?:@\w[^\n]*\n)*)"
            rf"(?P<def>def\s+{re.escape(func_name)}\s*\(.*?)(?=^\s*(?:def|class)\s+|\Z)"
        )
        m = re.search(pattern, src)
        if not m:
            return None
        return (m.group("decor") + m.group("def")).strip()

    if re.search(r"(?m)^(def|class|import|from)\s+", g):
        target_name = infer_target_func_name(prompt)
        if target_name:
            extracted = extract_single_function(g, target_name)
            g2 = extracted if extracted else g.rstrip()
        else:
            g2 = g.rstrip()

        imports = extract_prompt_imports(prompt)
        if imports and imports not in g2:
            g2 = (imports + "\n\n" + g2).rstrip()
        return g2.rstrip()

    return (prompt + "\n" + indent_as_body(g.lstrip())).rstrip()


def read_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def aggregate_output_timing(path: Path) -> tuple[int, float, float, float]:
    n_records = 0
    remask_total = 0.0
    pack_total = 0.0
    for rec in read_jsonl(str(path)):
        timing = ((rec.get("gen") or {}).get("timing") or {})
        remask_total += float(timing.get("remask_generate_s") or 0.0)
        pack_total += float(timing.get("pack_solution_s") or 0.0)
        n_records += 1
    return n_records, remask_total + pack_total, remask_total, pack_total


def cleanup_lock(lock_f, lock_path: Path) -> None:
    try:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
    finally:
        lock_f.close()
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def infer_refiner_name(model_id: str) -> str:
    mid = (model_id or "").lower()
    if "llada" in mid:
        return "llada"
    return "dream"


_DEFAULT_REFINER_MODELS = {
    "dream": "Dream-org/Dream-Coder-v0-Instruct-7B",
    "dream_general": "Dream-org/Dream-v0-Instruct-7B",
    "llada": "GSAI-ML/LLaDA-8B-Instruct",
}


def build_refiner(name: str, model_id: str | None, device: str):
    resolved_id = model_id or _DEFAULT_REFINER_MODELS.get(name)
    if name == "dream":
        return DreamCoder(model_id=resolved_id, device=device)
    if name == "dream_general":
        return DreamGeneral(model_id=resolved_id, device=device)
    if name == "llada":
        return LLaDACoder(model_id=resolved_id, device=device)
    raise ValueError(f"Unsupported refiner: {name}")


def is_id_record(rec: dict) -> bool:
    """Non-code records use 'id' instead of 'task_id' and should not be code-wrapped."""
    return "id" in rec and "task_id" not in rec


def infer_benchmark(rec: dict, task_id: str) -> str | None:
    benchmark = rec.get("benchmark")
    if isinstance(benchmark, str) and benchmark.strip():
        return benchmark.strip()
    if task_id.startswith("LiveBench/"):
        return "livebench-coding"
    if task_id.startswith("LiveCodeBench/"):
        return "livecodebench"
    if task_id.startswith("BigCodeBench/"):
        return "bigcodebench"
    return None


@torch.inference_mode()
def compute_refiner_mask_plan(
    model,
    refiner_name: str,
    req: ModelRequest,
    draft: str,
    confidence_threshold: float | None,
    mask_ratio: float | None,
    mask_granularity: str,
    span_merge_gap: int,
    external_confidence: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.BoolTensor | None, dict]:
    """Score the draft once and compute the planned remask positions."""
    comp_enc = model.tok(draft, return_tensors="pt", add_special_tokens=False)
    comp_ids = comp_enc.input_ids.to(model.device)
    n_tokens = int(comp_ids.shape[1])
    if n_tokens == 0:
        return external_confidence, None, {
            "draft_tokens": 0,
            "mask_tokens": 0,
            "mask_fraction": 0.0,
            "confidence_mean": None,
            "confidence_min": None,
            "confidence_max": None,
        }

    if external_confidence is not None:
        confidence = external_confidence.to(model.device)
    elif refiner_name in ("dream", "dream_general"):
        messages = [{"role": "user", "content": req.prompt}]
        enc = model.tok.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        prompt_ids = enc.input_ids.to(model.device)
        confidence = model.score_tokens(prompt_ids, comp_ids)
    elif refiner_name == "llada":
        messages = [{"role": "user", "content": req.prompt}]
        prompt_text = model.tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        enc = model.tok(
            prompt_text,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        prompt_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)
        confidence = model.score_tokens(prompt_ids, comp_ids, attention_mask)
    else:
        raise ValueError(f"Unsupported refiner for mask planning: {refiner_name}")

    mask_pos = apply_masking_policy(
        confidence,
        confidence_threshold,
        mask_ratio,
        mask_granularity,
        span_merge_gap,
        comp_ids,
        model.tok,
    )
    n_mask = int(mask_pos.sum().item())
    stats = {
        "draft_tokens": n_tokens,
        "mask_tokens": n_mask,
        "mask_fraction": (float(n_mask) / n_tokens) if n_tokens else 0.0,
        "confidence_mean": float(confidence.float().mean().item()) if n_tokens else None,
        "confidence_min": float(confidence.float().min().item()) if n_tokens else None,
        "confidence_max": float(confidence.float().max().item()) if n_tokens else None,
    }
    return confidence, mask_pos, stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Refine AR drafts with a diffusion refiner via confidence-based remasking."
    )
    ap.add_argument("--input", required=True,
                    help="Input JSONL file from AR model generation.")
    ap.add_argument("--out", required=True,
                    help="Output JSONL path for refined samples.")
    ap.add_argument("--refiner", choices=["dream", "dream_general", "llada"], default=None,
                    help="Refiner family. Default: infer from --model_id.")
    ap.add_argument("--model_id", default=None,
                    help="Refiner HuggingFace model ID. "
                         "Defaults to Dream-org/Dream-Coder-v0-Instruct-7B for dream, "
                         "GSAI-ML/LLaDA-8B-Instruct for llada.")
    ap.add_argument("--device", default="cuda")

    # Locator — which model decides which tokens to remask
    ap.add_argument(
        "--locator",
        choices=["dream", "ar", "bert"],
        default="dream",
        help=(
            "Model used to score draft tokens and decide what to remask.\n"
            "  dream (default): the refiner itself scores via a single forward pass\n"
            "                   (original CoCoder behaviour).\n"
            "  ar:              an AR model's teacher-forced log-probabilities\n"
            "                   (causal, left-context only).\n"
            "  bert:            CodeBERT single-pass MLM confidence\n"
            "                   (bidirectional, lightweight)."
        ),
    )
    ap.add_argument(
        "--locator_model_id",
        default=None,
        help=(
            "HuggingFace model ID for the locator when --locator is 'ar' or 'bert'.\n"
            "Defaults: ar → deepseek-ai/deepseek-coder-6.7b-instruct,\n"
            "          bert → microsoft/codebert-base-mlm."
        ),
    )

    # Masking policy (use exactly one)
    ap.add_argument("--confidence_threshold", type=float, default=None,
                    help="Mask tokens where P(token|context) < threshold. Default 0.5.")
    ap.add_argument("--mask_ratio", type=float, default=None,
                    help="Mask the bottom K%% least-confident tokens (0.0–1.0).")
    ap.add_argument(
        "--mask_granularity",
        choices=["token", "span", "line"],
        default="token",
        help="Masking granularity: token (default), span (merge nearby masks), line (mask whole lines).",
    )
    ap.add_argument(
        "--span_merge_gap",
        type=int,
        default=0,
        help="For --mask_granularity span: also mask gaps of <= this many tokens between masked tokens.",
    )
    ap.add_argument(
        "--gate_min_mask_fraction",
        type=float,
        default=None,
        help=(
            "Skip refinement when the planned mask fraction is below this value. "
            "Uses only pre-refinement locator scores."
        ),
    )
    ap.add_argument(
        "--gate_max_mask_fraction",
        type=float,
        default=None,
        help=(
            "Skip refinement when the planned mask fraction is above this value. "
            "Uses only pre-refinement locator scores."
        ),
    )
    ap.add_argument(
        "--record_mask_stats",
        action="store_true",
        help="Record planned mask counts, mask fraction, and confidence stats in gen metadata.",
    )

    # Diffusion generation params for the refinement step
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--max_new_tokens", type=int, default=512,
                    help="Fallback budget; actual max_new_tokens is set to draft length.")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_idx", type=int, default=0)

    ap.add_argument("--resume", action="store_true",
                    help="Skip task_ids already present in --out (append mode).")
    args = ap.parse_args()

    # Resolve masking policy
    if args.confidence_threshold is None and args.mask_ratio is None:
        args.confidence_threshold = 0.5
    if args.confidence_threshold is not None and args.mask_ratio is not None:
        ap.error("Specify at most one of --confidence_threshold and --mask_ratio.")
    if args.gate_min_mask_fraction is not None and args.gate_min_mask_fraction < 0:
        ap.error("--gate_min_mask_fraction must be >= 0.")
    if args.gate_max_mask_fraction is not None and args.gate_max_mask_fraction < 0:
        ap.error("--gate_max_mask_fraction must be >= 0.")
    if (
        args.gate_min_mask_fraction is not None
        and args.gate_max_mask_fraction is not None
        and args.gate_min_mask_fraction > args.gate_max_mask_fraction
    ):
        ap.error("--gate_min_mask_fraction must be <= --gate_max_mask_fraction.")
    try:
        validate_shard_args(num_shards=args.num_shards, shard_idx=args.shard_idx)
    except ValueError as e:
        ap.error(str(e))

    refiner_name = args.refiner or infer_refiner_name(args.model_id or "")
    # Resolve default model_id per refiner type
    if args.model_id is None:
        _DEFAULT_MODEL_IDS = {
            "dream": "Dream-org/Dream-Coder-v0-Instruct-7B",
            "dream_general": "Dream-org/Dream-v0-Instruct-7B",
            "llada": "GSAI-ML/LLaDA-8B-Instruct",
        }
        args.model_id = _DEFAULT_MODEL_IDS.get(refiner_name, "Dream-org/Dream-Coder-v0-Instruct-7B")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = out_path.with_suffix(out_path.suffix + ".lock")
    lock_f = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as e:
        raise RuntimeError(f"Another gen_remask process is already writing to {out_path}") from e

    # Resume: collect already-done ids (support both task_id and math id)
    done_ids: set[str] = set()
    if args.resume and out_path.exists():
        for rec in read_jsonl(str(out_path)):
            done_ids.add(rec.get("task_id") or rec.get("id") or "")
        print(f"[resume] skipping {len(done_ids)} already-done tasks.")

    records = list(read_jsonl(args.input))
    records = take_shard(records, num_shards=args.num_shards, shard_idx=args.shard_idx)
    print(f"[info] {len(records)} records loaded from {args.input}")
    if not records:
        out_path.touch(exist_ok=True)
        summary = {
            "script": "gen_remask",
            "out": str(out_path.resolve()),
            "model": None,
            "refiner": refiner_name,
            "num_shards": args.num_shards,
            "shard_idx": args.shard_idx,
            "n_records_written": 0,
            "timing": {
                "total_s": 0.0,
                "remask_generate_s_total": 0.0,
                "pack_solution_s_total": 0.0,
                "remask_generate_s_avg": None,
            },
        }
        out_path.with_suffix(out_path.suffix + ".timing_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[timing] wrote {out_path}.timing_summary.json")
        print(f"[done] no records selected for shard {args.shard_idx}/{args.num_shards}")
        cleanup_lock(lock_f, lock_path)
        return

    # Load refiner once
    model = build_refiner(refiner_name, args.model_id, args.device)

    # Load locator (None means the refiner scores itself — original behaviour)
    locator = build_locator(
        name=args.locator,
        model_id=args.locator_model_id,
        device=args.device,
    )
    if locator is not None:
        args.locator_model_id = getattr(locator, "model_id", args.locator_model_id)
        print(f"[locator] using external locator: {args.locator} ({args.locator_model_id})")
    else:
        print(f"[locator] using {refiner_name} refiner as locator")

    t_total0 = time.perf_counter()
    timing_remask_s: list[float] = []
    timing_pack_s: list[float] = []
    n_records_written = 0

    with out_path.open("a", encoding="utf-8") as fout:
        for rec in tqdm(records, desc="remask"):
            id_mode = is_id_record(rec)
            rec_id: str = rec.get("id") if id_mode else rec.get("task_id", "")
            if rec_id in done_ids:
                continue

            prompt_text: str = rec.get("prompt", "")
            draft: str = (
                rec.get("raw_completion")
                or rec.get("solution")
                or rec.get("raw_solution")
                or ""
            )

            if not id_mode:
                draft = clean_model_completion(draft, prompt_text)

            req = ModelRequest(
                prompt=prompt_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
            )

            try:
                t0 = time.perf_counter()
                mask_stats: dict = {}
                skip_refine = False
                skip_reason: str | None = None

                # Compute external confidence when a non-dream locator is used.
                ext_conf: torch.Tensor | None = None
                if locator is not None and draft:
                    loc_conf, loc_spans = locator.score(prompt_text, draft)
                    if len(loc_conf) > 0:
                        # Align locator scores to the refiner's token space.
                        refiner_spans = get_token_char_spans(model.tok, draft)
                        aligned = align_confidence_to_spans(
                            loc_conf, loc_spans, refiner_spans,
                        )
                        ext_conf = torch.tensor(
                            aligned, dtype=torch.float32, device=args.device,
                        )

                should_plan_masks = (
                    args.record_mask_stats
                    or args.gate_min_mask_fraction is not None
                    or args.gate_max_mask_fraction is not None
                )
                if should_plan_masks and draft:
                    ext_conf, _mask_pos, mask_stats = compute_refiner_mask_plan(
                        model=model,
                        refiner_name=refiner_name,
                        req=req,
                        draft=draft,
                        confidence_threshold=args.confidence_threshold,
                        mask_ratio=args.mask_ratio,
                        mask_granularity=args.mask_granularity,
                        span_merge_gap=args.span_merge_gap,
                        external_confidence=ext_conf,
                    )
                    mask_fraction = float(mask_stats.get("mask_fraction") or 0.0)
                    if (
                        args.gate_min_mask_fraction is not None
                        and mask_fraction < args.gate_min_mask_fraction
                    ):
                        skip_refine = True
                        skip_reason = "mask_fraction_below_min"
                    if (
                        args.gate_max_mask_fraction is not None
                        and mask_fraction > args.gate_max_mask_fraction
                    ):
                        skip_refine = True
                        skip_reason = "mask_fraction_above_max"

                if skip_refine:
                    refined = draft
                else:
                    refined = model.generate_with_remask(
                        req=req,
                        draft=draft,
                        confidence_threshold=args.confidence_threshold,
                        mask_ratio=args.mask_ratio,
                        mask_granularity=args.mask_granularity,
                        span_merge_gap=args.span_merge_gap,
                        external_confidence=ext_conf,
                    )
                t1 = time.perf_counter()
            except Exception as e:
                print(f"[warn] {rec_id}: generate_with_remask failed ({e}); keeping draft.")
                refined = draft
                t1 = time.perf_counter()
                t0 = t1
                mask_stats = {}
                skip_refine = True
                skip_reason = "generate_with_remask_failed"

            t_pack0 = time.perf_counter()
            if id_mode:
                # Math/research/writing: no code solution wrapping; evaluators read raw_completion.
                pass
            else:
                benchmark = infer_benchmark(rec, rec_id) or "evalplus"
                is_livebench = benchmark in ("livebench-coding", "livecodebench")
                refined = clean_model_completion(refined, prompt_text)
                if is_livebench:
                    solution = refined
                elif benchmark == "bigcodebench":
                    solution = build_prompt_scaffold_solution(prompt_text, refined)
                elif skip_refine and rec.get("solution"):
                    solution = rec["solution"]
                else:
                    solution = build_evalplus_solution(prompt_text, refined)
            t_pack1 = time.perf_counter()

            timing_remask_s.append(t1 - t0)
            timing_pack_s.append(t_pack1 - t_pack0)

            gen_meta = {
                "source_model":         rec.get("model", "unknown"),
                "refiner":              refiner_name,
                "locator":              args.locator,
                "locator_model_id":     args.locator_model_id,
                "confidence_threshold": args.confidence_threshold,
                "mask_ratio":           args.mask_ratio,
                "mask_granularity":     args.mask_granularity,
                "span_merge_gap":       args.span_merge_gap,
                "gate_min_mask_fraction": args.gate_min_mask_fraction,
                "gate_max_mask_fraction": args.gate_max_mask_fraction,
                "skip_refine":          skip_refine,
                "skip_reason":          skip_reason,
                "temperature":          args.temperature,
                "top_p":                args.top_p,
                "seed":                 args.seed,
                "timing": {
                    "remask_generate_s": t1 - t0,
                    "pack_solution_s":   t_pack1 - t_pack0,
                    "total_s":           (t1 - t0) + (t_pack1 - t_pack0),
                },
            }
            if args.record_mask_stats or mask_stats:
                gen_meta.update(mask_stats)

            if id_mode:
                out_rec = dict(rec)
                out_rec.update({
                    "id":               rec_id,
                    "sample_id":        rec.get("sample_id", 0),
                    "prompt":           prompt_text,
                    "draft_completion": draft,
                    "raw_completion":   refined,
                    "model":            model.name,
                    "gen":              gen_meta,
                })
            else:
                out_rec = {
                    "task_id":           rec_id,
                    "benchmark":         benchmark,
                    "prompt":            prompt_text,
                    "draft_completion":  draft,
                    "raw_completion":    refined,
                    "solution":          solution,
                    "model":             model.name,
                    "gen":               gen_meta,
                }
                # LiveBench: preserve question_id / meta
                if is_livebench:
                    for key in ("question_id", "meta"):
                        if key in rec:
                            out_rec[key] = rec[key]
                if benchmark == "bigcodebench":
                    for key in ("split", "subset", "revision", "source_prompt"):
                        if key in rec:
                            out_rec[key] = rec[key]

            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_records_written += 1

    t_total1 = time.perf_counter()
    summary_n_records = n_records_written
    summary_total_s = t_total1 - t_total0
    summary_remask_s = float(sum(timing_remask_s))
    summary_pack_s = float(sum(timing_pack_s))
    if args.resume:
        try:
            summary_n_records, summary_total_s, summary_remask_s, summary_pack_s = aggregate_output_timing(out_path)
        except Exception as e:
            print(f"[warn] failed to aggregate resumed output timing ({e}); using this-run timing only.")

    timing_path = str(out_path) + ".timing_summary.json"
    summary = {
        "script": "gen_remask",
        "out": str(out_path.resolve()),
        "model": model.name,
        "refiner": refiner_name,
        "num_shards": args.num_shards,
        "shard_idx": args.shard_idx,
        "n_records_written": summary_n_records,
        "timing": {
            "total_s": summary_total_s,
            "remask_generate_s_total": summary_remask_s,
            "pack_solution_s_total": summary_pack_s,
            "remask_generate_s_avg": (summary_remask_s / summary_n_records) if summary_n_records else None,
        },
    }
    out_path.with_suffix(out_path.suffix + ".timing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[timing] wrote {timing_path}")

    print(f"[done] wrote {args.out}")
    cleanup_lock(lock_f, lock_path)


if __name__ == "__main__":
    main()

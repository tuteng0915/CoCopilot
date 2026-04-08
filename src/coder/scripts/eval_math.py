"""Evaluate math generation outputs from gen_math.py.

Reads a .jsonl file produced by gen_math.py, extracts the predicted answer
from each raw_completion, compares it to answer_ref, and reports accuracy.

Usage:
  python -m coder.scripts.eval_math --input outputs/gsm8k_dream.jsonl
  python -m coder.scripts.eval_math --input outputs/math500_qwen.jsonl --output results/math500_qwen_eval.json
  python -m coder.scripts.eval_math --input outputs/math500_qwen.jsonl --per_subject
"""
import argparse
import json
import os
import re
import unicodedata
from collections import defaultdict
from typing import Optional


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract the number after the last '####' marker in model output."""
    matches = re.findall(r"####\s*([^\n]+)", text)
    if matches:
        return normalize_number(matches[-1].strip())
    # Fallback: last standalone number in the text
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return normalize_number(nums[-1]) if nums else None


def extract_math500_answer(text: str) -> Optional[str]:
    """Extract content from the last \\boxed{} in model output."""
    # Handle nested braces with a simple brace-counting scan
    pattern = r"\\boxed\{"
    last_start = None
    for m in re.finditer(pattern, text):
        last_start = m.end()  # position right after the opening {

    if last_start is None:
        return None

    depth = 1
    i = last_start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth == 0:
        return normalize_latex(text[last_start : i - 1])
    return None


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize_number(s: str) -> str:
    """Strip commas, trailing zeros, whitespace from a numeric string."""
    s = s.strip().replace(",", "")
    try:
        # Normalize float representation: 72.0 -> 72, 3.50 -> 3.5
        f = float(s)
        if f == int(f):
            return str(int(f))
        return str(f)
    except ValueError:
        return s


def normalize_latex(s: str) -> str:
    """Light normalization for LaTeX math answers."""
    s = s.strip()
    # Remove outer $ if any
    s = s.strip("$").strip()
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    # Unicode normalization
    s = unicodedata.normalize("NFC", s)
    return s


# ---------------------------------------------------------------------------
# Answer comparison
# ---------------------------------------------------------------------------

def answers_match_gsm8k(pred: Optional[str], ref: str) -> bool:
    if pred is None:
        return False
    pred_n = normalize_number(pred)
    ref_n = normalize_number(ref)
    return pred_n == ref_n


def answers_match_math500(pred: Optional[str], ref: str) -> bool:
    if pred is None:
        return False
    pred_n = normalize_latex(pred)
    ref_n = normalize_latex(ref)
    if pred_n == ref_n:
        return True
    # Try numeric comparison as a fallback (handles "0.5" vs "\\frac{1}{2}" only if both parse)
    try:
        return abs(float(pred_n) - float(ref_n)) < 1e-6
    except (ValueError, TypeError):
        pass
    return False


def check_correct(dataset: str, pred: Optional[str], ref: str) -> bool:
    if dataset == "gsm8k":
        return answers_match_gsm8k(pred, ref)
    else:
        return answers_match_math500(pred, ref)


def extract_answer(dataset: str, text: str) -> Optional[str]:
    if dataset == "gsm8k":
        return extract_gsm8k_answer(text)
    else:
        return extract_math500_answer(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="Path to .jsonl file from gen_math.py")
    ap.add_argument("--out_summary", default=None, help="Path to write JSON evaluation summary (default: <samples>.eval.json)")
    ap.add_argument("--per_subject", action="store_true", help="(MATH-500 only) Break down accuracy by subject.")
    ap.add_argument("--per_level", action="store_true", help="(MATH-500 only) Break down accuracy by difficulty level.")
    ap.add_argument("--show_errors", type=int, default=0, help="Print first N wrong examples for debugging.")
    args = ap.parse_args()

    if not os.path.exists(args.samples):
        raise FileNotFoundError(args.samples)

    records = []
    with open(args.samples, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("[eval] no records found")
        return

    dataset = records[0].get("dataset")
    if dataset not in ("gsm8k", "math500"):
        # Try to infer from id field
        first_id = records[0].get("id", "")
        dataset = "gsm8k" if first_id.startswith("gsm8k/") else "math500"
    print(f"[eval] dataset={dataset}, records={len(records)}")

    # Group by problem id to support pass@n (multiple samples per problem)
    # A problem is correct if ANY sample is correct (pass@1 when num_samples=1)
    problems: dict[str, list] = defaultdict(list)
    for rec in records:
        problems[rec["id"]].append(rec)

    correct_problems = 0
    total_problems = len(problems)

    subject_stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    level_stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})

    errors = []

    for pid, samples in problems.items():
        ref = samples[0]["answer_ref"]
        any_correct = False
        for rec in samples:
            pred = extract_answer(dataset, rec["raw_completion"])
            if check_correct(dataset, pred, ref):
                any_correct = True
                break

        if any_correct:
            correct_problems += 1
        else:
            if len(errors) < args.show_errors:
                last_rec = samples[-1]
                pred = extract_answer(dataset, last_rec["raw_completion"])
                errors.append({
                    "id": pid,
                    "question": last_rec["question"][:200],
                    "ref": ref,
                    "pred": pred,
                    "completion_tail": last_rec["raw_completion"][-300:],
                })

        # Per-subject / per-level breakdown (MATH-500)
        subj = samples[0].get("subject", "unknown")
        lvl = str(samples[0].get("level", "unknown"))
        subject_stats[subj]["total"] += 1
        level_stats[lvl]["total"] += 1
        if any_correct:
            subject_stats[subj]["correct"] += 1
            level_stats[lvl]["correct"] += 1

    accuracy = correct_problems / total_problems if total_problems else 0.0
    print(f"\n[result] accuracy = {correct_problems}/{total_problems} = {accuracy:.4f} ({accuracy*100:.2f}%)")

    if args.per_subject and dataset == "math500":
        print("\n[subject breakdown]")
        for subj in sorted(subject_stats):
            s = subject_stats[subj]
            acc = s["correct"] / s["total"] if s["total"] else 0.0
            print(f"  {subj:<30s}  {s['correct']:3d}/{s['total']:3d}  {acc*100:.1f}%")

    if args.per_level and dataset == "math500":
        print("\n[level breakdown]")
        for lvl in sorted(level_stats):
            s = level_stats[lvl]
            acc = s["correct"] / s["total"] if s["total"] else 0.0
            print(f"  Level {lvl}  {s['correct']:3d}/{s['total']:3d}  {acc*100:.1f}%")

    if errors:
        print(f"\n[first {len(errors)} wrong examples]")
        for e in errors:
            print(f"  id={e['id']}")
            print(f"    ref : {e['ref']}")
            print(f"    pred: {e['pred']}")
            print(f"    tail: ...{e['completion_tail'][-200:]}")
            print()

    # Pull latency stats from companion timing_summary.json if available
    timing: Optional[dict] = None
    timing_path = args.samples + ".timing_summary.json"
    if os.path.exists(timing_path):
        with open(timing_path, encoding="utf-8") as _tf:
            timing = json.load(_tf).get("timing")

    # Write summary JSON
    out_path = args.out_summary or (args.samples + ".eval.json")
    summary = {
        "samples": os.path.abspath(args.samples),
        "dataset": dataset,
        "model": records[0].get("model", "unknown"),
        "n_problems": total_problems,
        "n_correct": correct_problems,
        "accuracy": accuracy,
        "timing": timing,
    }
    if args.per_subject and dataset == "math500":
        summary["subject_breakdown"] = {
            subj: {**s, "accuracy": s["correct"] / s["total"] if s["total"] else 0.0}
            for subj, s in subject_stats.items()
        }
    if args.per_level and dataset == "math500":
        summary["level_breakdown"] = {
            lvl: {**s, "accuracy": s["correct"] / s["total"] if s["total"] else 0.0}
            for lvl, s in level_stats.items()
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[eval]  wrote {out_path}")


if __name__ == "__main__":
    main()

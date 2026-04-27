"""Evaluate code-mode math generations by executing solution() outputs.

Supports JSONL samples produced by:
  - coder.scripts.gen_math_code
  - coder.scripts.gen_remask (when preserving math/id records in code mode)
"""
from __future__ import annotations

import argparse
import ast
import builtins
import contextlib
import io
import json
import math
import os
import re
import signal
import textwrap
import unicodedata
from collections import defaultdict
from fractions import Fraction
from typing import Any, Optional

try:
    import sympy
    from sympy.parsing.sympy_parser import (
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )
except Exception:  # pragma: no cover - optional dependency
    sympy = None
    parse_expr = None
    standard_transformations = ()
    implicit_multiplication_application = None


TIMEOUT_S = 5.0
_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_SOLUTION_DEF_RE = re.compile(r"(?m)^\s*def\s+solution\s*\(")
_BOXED_RE = re.compile(r"\\boxed\{")
_FRAC_RE = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
_SQRT_RE = re.compile(r"\\sqrt\s*\{([^{}]+)\}")


def read_jsonl(path: str) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def strip_code_fences(text: str) -> str:
    text = text or ""
    if not text.strip():
        return ""
    match = _CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip("\n")
    return text.strip("\n")


def ensure_solution_function(code: str) -> str:
    code = strip_code_fences(code)
    if not code:
        return "def solution():\n    return None\n"

    match = _SOLUTION_DEF_RE.search(code)
    if match:
        return code[match.start():].strip() + "\n"

    body = textwrap.dedent(code).strip("\n")
    if not body.strip():
        return "def solution():\n    return None\n"
    return f"def solution():\n{textwrap.indent(body, '    ')}\n"


@contextlib.contextmanager
def time_limit(seconds: float):
    def _handle_timeout(signum, frame):
        raise TimeoutError(f"Timed out after {seconds:.1f}s")

    prev_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    prev_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prev_handler)
        if prev_timer != (0.0, 0.0):
            signal.setitimer(signal.ITIMER_REAL, *prev_timer)


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    allowed = {"fractions", "math", "sympy"}
    root = name.split(".", 1)[0]
    if root not in allowed:
        raise ImportError(f"Import blocked in eval sandbox: {name}")
    return builtins.__import__(name, globals, locals, fromlist, level)


SAFE_BUILTINS = {
    "__import__": _safe_import,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "Exception": Exception,
    "float": float,
    "filter": filter,
    "Fraction": Fraction,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "TypeError": TypeError,
    "tuple": tuple,
    "ValueError": ValueError,
    "zip": zip,
}


def build_exec_namespace() -> dict[str, Any]:
    namespace = {
        "__builtins__": SAFE_BUILTINS,
        "__name__": "__math_solution__",
        "Fraction": Fraction,
        "math": math,
    }
    if sympy is not None:
        namespace["sympy"] = sympy
        namespace["sp"] = sympy
    return namespace


def exec_solution(code: str, timeout_s: float = TIMEOUT_S) -> Optional[str]:
    full_code = ensure_solution_function(code)
    try:
        tree = ast.parse(full_code, filename="<math_solution>", mode="exec")
    except SyntaxError:
        return None

    has_solution = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "solution"
        for node in tree.body
    )
    if not has_solution:
        return None

    namespace = build_exec_namespace()
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    try:
        with time_limit(timeout_s):
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                exec(compile(tree, "<math_solution>", "exec"), namespace)
                result = namespace["solution"]()
        return str(result).strip()
    except Exception:
        return None


def extract_boxed(text: str) -> str:
    text = text.strip()
    last_start = None
    for match in _BOXED_RE.finditer(text):
        last_start = match.end()
    if last_start is None:
        return text

    depth = 1
    idx = last_start
    while idx < len(text) and depth > 0:
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
        idx += 1
    if depth == 0:
        return text[last_start: idx - 1]
    return text


def normalize_number(text: str) -> str:
    text = unicodedata.normalize("NFC", (text or "").strip())
    text = text.replace(",", "")
    if not text:
        return ""
    try:
        value = float(text)
    except ValueError:
        return text
    if math.isfinite(value) and value == int(value):
        return str(int(value))
    return str(value)


def _replace_latex_constructs(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = _FRAC_RE.sub(r"(\1)/(\2)", text)
        text = _SQRT_RE.sub(r"sqrt(\1)", text)
    return text


def normalize_math_expr(text: str) -> str:
    text = unicodedata.normalize("NFC", (text or "").strip())
    text = extract_boxed(text)
    text = text.strip("$").strip()
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\cdot", "*").replace("\\times", "*")
    text = text.replace("\\pi", "pi")
    text = text.replace("^", "**")
    text = _replace_latex_constructs(text)
    text = text.replace("{", "(").replace("}", ")")
    text = re.sub(r"\(([-+]?\d+(?:\.\d+)?)\)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    numeric = normalize_number(text)
    return numeric if numeric else text


def maybe_parse_expr(text: str):
    if sympy is None or parse_expr is None:
        return None
    normalized = normalize_math_expr(text)
    if not normalized:
        return None
    try:
        transformations = standard_transformations
        if implicit_multiplication_application is not None:
            transformations = transformations + (implicit_multiplication_application,)
        return parse_expr(normalized, transformations=transformations, evaluate=True)
    except Exception:
        return None


def answers_match_gsm8k(pred: Optional[str], ref: str) -> bool:
    if pred is None:
        return False
    pred_n = normalize_number(pred)
    ref_n = normalize_number(ref)
    if pred_n == ref_n:
        return True
    try:
        return math.isclose(float(pred_n), float(ref_n), rel_tol=1e-6, abs_tol=1e-6)
    except (TypeError, ValueError):
        return False


def answers_match_math500(pred: Optional[str], ref: str) -> bool:
    if pred is None:
        return False
    pred_n = normalize_math_expr(pred)
    ref_n = normalize_math_expr(ref)
    if pred_n == ref_n:
        return True

    try:
        if math.isclose(float(pred_n), float(ref_n), rel_tol=1e-6, abs_tol=1e-6):
            return True
    except (TypeError, ValueError):
        pass

    pred_expr = maybe_parse_expr(pred_n)
    ref_expr = maybe_parse_expr(ref_n)
    if pred_expr is None or ref_expr is None or sympy is None:
        return False

    try:
        return bool(sympy.simplify(pred_expr - ref_expr) == 0)
    except Exception:
        return False


_INTEGER_ANSWER_DATASETS = {"gsm8k", "aime", "aime2025"}


def check_correct(dataset: str, pred: Optional[str], ref: str) -> bool:
    if dataset in _INTEGER_ANSWER_DATASETS:
        return answers_match_gsm8k(pred, ref)
    return answers_match_math500(pred, ref)


def infer_dataset(records: list[dict[str, Any]]) -> str:
    dataset = records[0].get("dataset")
    if dataset in ("gsm8k", "math500", "aime", "aime2025"):
        return dataset
    first_id = str(records[0].get("id", ""))
    if first_id.startswith("gsm8k/"):
        return "gsm8k"
    if first_id.startswith("aime2025/"):
        return "aime2025"
    if first_id.startswith("aime/"):
        return "aime"
    return "math500"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="Path to JSONL samples from gen_math_code.py")
    ap.add_argument("--samples", default=None, help="Alias for --input")
    ap.add_argument("--out", default=None, help="Path to write JSON summary")
    ap.add_argument("--out_summary", default=None, help="Alias for --out")
    ap.add_argument("--completion_field", default="raw_completion", help="Field to execute. Default: raw_completion")
    ap.add_argument("--per_subject", action="store_true", help="(MATH-500 only) Break down accuracy by subject.")
    ap.add_argument("--per_level", action="store_true", help="(MATH-500 only) Break down accuracy by difficulty level.")
    ap.add_argument("--show_errors", type=int, default=0, help="Include first N wrong examples in the summary.")
    ap.add_argument("--timeout_s", type=float, default=TIMEOUT_S, help="Per-sample execution timeout in seconds.")
    args = ap.parse_args()

    samples_path = args.input or args.samples
    if not samples_path:
        ap.error("One of --input or --samples is required.")
    if not os.path.exists(samples_path):
        raise FileNotFoundError(samples_path)

    records = read_jsonl(samples_path)
    if not records:
        print("[eval] no records found")
        return

    dataset = infer_dataset(records)
    print(f"[eval] dataset={dataset}, records={len(records)}, completion_field={args.completion_field}")

    problems: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        problems[rec["id"]].append(rec)

    correct_problems = 0
    total_problems = len(problems)
    exec_failures = 0
    subject_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    level_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    problem_results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for pid, samples in problems.items():
        ref = samples[0]["answer_ref"]
        any_correct = False
        chosen_pred = None
        chosen_sample_id = None

        for rec in samples:
            completion = rec.get(args.completion_field)
            if completion is None and args.completion_field != "raw_completion":
                completion = rec.get("raw_completion")
            pred = exec_solution(completion or "", timeout_s=args.timeout_s)
            if pred is None:
                exec_failures += 1
            if check_correct(dataset, pred, ref):
                any_correct = True
                chosen_pred = pred
                chosen_sample_id = rec.get("sample_id", 0)
                break
            chosen_pred = pred
            chosen_sample_id = rec.get("sample_id", 0)

        if any_correct:
            correct_problems += 1
        elif len(errors) < args.show_errors:
            last_rec = samples[-1]
            errors.append({
                "id": pid,
                "question": last_rec.get("question", "")[:200],
                "ref": ref,
                "pred": chosen_pred,
                "sample_id": chosen_sample_id,
                "completion_tail": (
                    last_rec.get(args.completion_field)
                    or last_rec.get("raw_completion")
                    or ""
                )[-300:],
            })

        subj = samples[0].get("subject", "unknown")
        lvl = str(samples[0].get("level", "unknown"))
        subject_stats[subj]["total"] += 1
        level_stats[lvl]["total"] += 1
        if any_correct:
            subject_stats[subj]["correct"] += 1
            level_stats[lvl]["correct"] += 1

        problem_results.append({
            "id": pid,
            "correct": any_correct,
            "pred": chosen_pred,
            "ref": ref,
            "sample_id": chosen_sample_id,
            "num_samples": len(samples),
        })

    accuracy = correct_problems / total_problems if total_problems else 0.0
    print(f"[result] accuracy = {correct_problems}/{total_problems} = {accuracy:.4f} ({accuracy * 100:.2f}%)")

    timing = None
    timing_path = samples_path + ".timing_summary.json"
    if os.path.exists(timing_path):
        with open(timing_path, encoding="utf-8") as tf:
            timing = json.load(tf).get("timing")

    out_path = args.out or args.out_summary or (samples_path + ".eval.json")
    summary = {
        "samples": os.path.abspath(samples_path),
        "dataset": dataset,
        "model": records[0].get("model", "unknown"),
        "completion_field": args.completion_field,
        "timeout_s": args.timeout_s,
        "n_records": len(records),
        "n_problems": total_problems,
        "n_correct": correct_problems,
        "accuracy": accuracy,
        "exec_failures": exec_failures,
        "timing": timing,
        "results": problem_results,
    }
    if args.per_subject and dataset == "math500":
        summary["subject_breakdown"] = {
            subj: {**stats, "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0.0}
            for subj, stats in subject_stats.items()
        }
    if args.per_level and dataset == "math500":
        summary["level_breakdown"] = {
            lvl: {**stats, "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0.0}
            for lvl, stats in level_stats.items()
        }
    if errors:
        summary["errors"] = errors

    with open(out_path, "w", encoding="utf-8") as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()

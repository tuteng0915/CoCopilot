# scripts/eval_evalplus.py
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional


def _candidate_evalplus_result_paths(samples: str) -> list[str]:
    """Possible evalplus output paths across versions."""
    if os.path.isdir(samples):
        return [os.path.join(samples, "eval_results.json")]

    if samples.endswith(".jsonl"):
        # Newer / common style
        p1 = samples.replace(".jsonl", ".eval_results.json")
        # Older style seen in some setups
        p2 = samples.replace(".jsonl", "_eval_results.json")
        # Another occasional variant
        p3 = samples.replace(".jsonl", "-eval_results.json")
        # Dedup while preserving order
        out = []
        for p in (p1, p2, p3):
            if p not in out:
                out.append(p)
        return out

    return [samples + ".eval_results.json"]


def _default_evalplus_result_path(samples: str) -> str:
    """Best-guess default path (used before file exists)."""
    return _candidate_evalplus_result_paths(samples)[0]


def _resolve_actual_evalplus_result_path(samples: str) -> str:
    """
    Resolve actual evalplus result file after evaluation.
    Tries known naming conventions and returns the first existing one.
    """
    candidates = _candidate_evalplus_result_paths(samples)
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError(
        "Could not find EvalPlus result file after evaluation.\n"
        f"Tried: {candidates}\n"
        "Please check your evalplus version output naming."
    )


def _infer_model_name(samples_path: str) -> str:
    """
    Infer model name from sample filename, e.g.
      outputs/dream_humaneval-sanitized.jsonl -> dream
      outputs/deepseek_mbpp.jsonl -> deepseek
    """
    name = Path(samples_path).name
    name = re.sub(r"\.jsonl$", "", name)
    name = re.sub(r"-sanitized$", "", name)
    if "_" in name:
        return name.split("_", 1)[0]
    return name


def _safe_pass(x: Optional[str]) -> bool:
    return isinstance(x, str) and x.lower() == "pass"


def _build_summary(
    data: Dict[str, Any],
    model: str,
    dataset: str,
    source_eval_file: str,
    include_per_task: bool,
) -> Dict[str, Any]:
    eval_map = data.get("eval", {}) or {}

    base_status_counts = Counter()
    plus_status_counts = Counter()

    n_tasks = 0
    n_samples_total = 0
    n_base_pass = 0
    n_plus_pass = 0
    n_both_pass = 0
    plus_seen = False

    per_task_summary = {}

    for task_id, rows in eval_map.items():
        n_tasks += 1
        rows = rows or []
        n_samples_total += len(rows)

        task_base_counts = Counter()
        task_plus_counts = Counter()
        task_base_pass = 0
        task_plus_pass = 0
        task_both_pass = 0

        for r in rows:
            b = r.get("base_status")
            p = r.get("plus_status")

            task_base_counts[str(b)] += 1
            base_status_counts[str(b)] += 1

            if p is not None:
                plus_seen = True
                task_plus_counts[str(p)] += 1
                plus_status_counts[str(p)] += 1

            bp = _safe_pass(b)
            pp = _safe_pass(p) if p is not None else False

            task_base_pass += int(bp)
            n_base_pass += int(bp)

            if p is not None:
                task_plus_pass += int(pp)
                n_plus_pass += int(pp)

            both = bp and (p is None or pp)
            task_both_pass += int(both)
            n_both_pass += int(both)

        if include_per_task:
            per_task_summary[task_id] = {
                "n_samples": len(rows),
                "base_pass": task_base_pass,
                "plus_pass": task_plus_pass if plus_seen else None,
                "base_status_counts": dict(task_base_counts),
                "plus_status_counts": dict(task_plus_counts) if task_plus_counts else {},
            }

    out = {
        "model": model,
        "dataset": dataset,
        "source_eval_file": source_eval_file,
        "date": data.get("date"),
        "hash": data.get("hash"),
        "pass_at_k": data.get("pass_at_k", {}),
        "summary": {
            "n_tasks": n_tasks,
            "n_samples_total": n_samples_total,
            "n_base_pass": n_base_pass,
            "n_plus_pass": n_plus_pass if plus_seen else None,
            "n_both_pass": n_both_pass,
            "base_status_counts": dict(base_status_counts),
            "plus_status_counts": dict(plus_status_counts) if plus_seen else {},
        },
    }

    if include_per_task:
        out["per_task_summary"] = per_task_summary

    return out


def _write_summary(
    eval_result_path: str,
    samples_path: str,
    dataset: str,
    summary_out: Optional[str] = None,
    model_name: Optional[str] = None,
    include_per_task: bool = False,
) -> str:
    with open(eval_result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model = model_name or _infer_model_name(samples_path)

    if summary_out is None:
        out_dir = Path(eval_result_path).parent
        summary_out = str(out_dir / f"{model}_{dataset}_summary.json")

    summary = _build_summary(
        data=data,
        model=model,
        dataset=dataset,
        source_eval_file=os.path.abspath(eval_result_path),
        include_per_task=include_per_task,
    )

    Path(summary_out).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    ap.add_argument("--samples", required=True, help="samples jsonl (recommend sanitized)")
    ap.add_argument("--repo_root", default=".", help="Mounted to /app in docker mode")
    ap.add_argument("--override_path", default=None, help="Subset override .jsonl.gz (optional)")
    ap.add_argument("--base_only", action="store_true", help="Only run base tests (faster)")
    ap.add_argument("--mini", action="store_true", help="Use EvalPlus mini benchmark (faster)")
    ap.add_argument(
        "--test_details",
        action="store_true",
        help="Run all tests + store detailed failures (slower, larger)",
    )
    ap.add_argument("--parallel", type=int, default=None, help="Pass through to evalplus.evaluate")
    ap.add_argument("--version", default=None, help="Pass through to evalplus.evaluate")
    ap.add_argument("--noextreme", action="store_true", help="Pass through to evalplus.evaluate")

    # IMPORTANT: this is wrapper-level output handling only.
    # We NEVER pass --output_file to evalplus.evaluate for compatibility with older evalplus versions.
    ap.add_argument(
        "--output_file",
        default=None,
        help="Final eval result json path (wrapper post-process rename/copy; NOT passed to evalplus.evaluate)",
    )

    ap.add_argument("--image", default="ganler/evalplus:latest")
    ap.add_argument("--pull_always", action="store_true")
    ap.add_argument(
        "--backend",
        choices=["docker", "local"],
        default="docker",
        help="Run evalplus.evaluate in docker or directly in current environment",
    )

    # Summary controls
    ap.add_argument("--no_summary", action="store_true", help="Do not generate {model}_{dataset}_summary.json")
    ap.add_argument("--summary_out", default=None, help="Explicit summary output path")
    ap.add_argument("--summary_model", default=None, help="Override model name used in summary filename/content")
    ap.add_argument("--summary_per_task", action="store_true", help="Include per-task summary in summary json")

    args = ap.parse_args()

    # Best guess before run (for logging only)
    guessed_eval_output_file = _default_evalplus_result_path(args.samples)

    # ---------- local mode ----------
    if args.backend == "local":
        env = os.environ.copy()

        # subset override support
        if args.override_path:
            override_abs = os.path.abspath(args.override_path)
            if args.dataset == "humaneval":
                env["HUMANEVAL_OVERRIDE_PATH"] = override_abs
            else:
                env["MBPP_OVERRIDE_PATH"] = override_abs

        cmd = ["evalplus.evaluate", "--dataset", args.dataset, "--samples", args.samples]
        if args.base_only:
            cmd += ["--base-only"]
        if args.mini:
            cmd += ["--mini"]
        if args.test_details:
            cmd += ["--test-details"]
        if args.noextreme:
            cmd += ["--noextreme"]
        if args.parallel is not None:
            cmd += ["--parallel", str(args.parallel)]
        if args.version is not None:
            cmd += ["--version", args.version]

        print("[cmd]", " ".join(cmd))
        print(f"[info] expecting EvalPlus result near: {guessed_eval_output_file}")
        subprocess.run(cmd, check=True, env=env)

    # ---------- docker mode ----------
    else:
        repo_root = os.path.abspath(args.repo_root)
        samples_abs = os.path.abspath(args.samples)

        cmd = ["docker", "run", "--rm"]
        if args.pull_always:
            cmd += ["--pull=always"]

        cmd += ["-v", f"{repo_root}:/app"]

        # subset override support
        if args.override_path:
            override_abs = os.path.abspath(args.override_path)
            if not override_abs.startswith(repo_root):
                raise ValueError("--override_path must be inside repo_root so docker can mount it")
            override_in_container = "/app/" + os.path.relpath(override_abs, repo_root)
            if args.dataset == "humaneval":
                cmd += ["-e", f"HUMANEVAL_OVERRIDE_PATH={override_in_container}"]
            else:
                cmd += ["-e", f"MBPP_OVERRIDE_PATH={override_in_container}"]

        if not samples_abs.startswith(repo_root):
            raise ValueError("--samples must be inside repo_root so docker can mount it")

        cmd += [args.image, "evalplus.evaluate", "--dataset", args.dataset]

        samples_in_container = "/app/" + os.path.relpath(samples_abs, repo_root)
        cmd += ["--samples", samples_in_container]

        if args.base_only:
            cmd += ["--base-only"]
        if args.mini:
            cmd += ["--mini"]
        if args.test_details:
            cmd += ["--test-details"]
        if args.noextreme:
            cmd += ["--noextreme"]
        if args.parallel is not None:
            cmd += ["--parallel", str(args.parallel)]
        if args.version is not None:
            cmd += ["--version", args.version]

        print("[cmd]", " ".join(cmd))
        print(f"[info] expecting EvalPlus result near: {guessed_eval_output_file}")
        subprocess.run(cmd, check=True)

    # ---------- locate actual evalplus result ----------
    actual_eval_result_path = _resolve_actual_evalplus_result_path(args.samples)
    print(f"[evalplus] result file: {actual_eval_result_path}")

    # ---------- optional wrapper-level rename/copy ----------
    if args.output_file:
        requested_eval_path = os.path.abspath(args.output_file)
        if requested_eval_path != actual_eval_result_path:
            Path(requested_eval_path).parent.mkdir(parents=True, exist_ok=True)
            try:
                os.replace(actual_eval_result_path, requested_eval_path)
                actual_eval_result_path = requested_eval_path
                print(f"[move] eval result moved to: {actual_eval_result_path}")
            except OSError:
                shutil.copy2(actual_eval_result_path, requested_eval_path)
                actual_eval_result_path = requested_eval_path
                print(f"[copy] eval result copied to: {actual_eval_result_path}")

    # ---------- summary step ----------
    if not args.no_summary:
        summary_path = _write_summary(
            eval_result_path=actual_eval_result_path,
            samples_path=args.samples,
            dataset=args.dataset,
            summary_out=args.summary_out,
            model_name=args.summary_model,
            include_per_task=args.summary_per_task,
        )
        print(f"[summary] wrote {summary_path}")


if __name__ == "__main__":
    main()
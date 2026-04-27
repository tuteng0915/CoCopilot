#!/usr/bin/env python3
"""Build statistical case-study tables for remask refinement runs.

The analysis intentionally separates the observed pipeline into:

  locator/gate signal: mask fraction, skipped/kept decision, confidence stats
  rewriter signal: edit size, syntax/signature/import changes, outcome transition

It does not try to prove semantic bug localization automatically. Instead it
creates deterministic candidate categories that can be audited in a case study.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PASS = "pass"


@dataclass(frozen=True)
class RunSpec:
    name: str
    jsonl_path: Path
    eval_path: Path


def load_jsonl_by_task(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open() as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            task_id = rec.get("task_id") or rec.get("id")
            if not task_id:
                raise ValueError(f"{path}:{lineno}: missing task_id/id")
            records[str(task_id)] = rec
    return records


def _row_pass(row: dict[str, Any], key: str) -> bool:
    return str(row.get(key) or "").lower() == PASS


def load_eval_passes(path: Path) -> dict[str, dict[str, bool]]:
    data = json.load(path.open())
    eval_map = data.get("eval")
    if not isinstance(eval_map, dict):
        raise ValueError(f"{path}: missing top-level eval map")

    out: dict[str, dict[str, bool]] = {}
    for task_id, rows in eval_map.items():
        if not isinstance(rows, list):
            rows = [rows]
        out[str(task_id)] = {
            "base_pass": any(_row_pass(r, "base_status") for r in rows),
            "plus_pass": any(_row_pass(r, "plus_status") for r in rows),
        }
    return out


def transition_name(before_pass: bool, after_pass: bool) -> str:
    if before_pass and after_pass:
        return "same_pass"
    if not before_pass and after_pass:
        return "wrong_to_correct"
    if before_pass and not after_pass:
        return "correct_to_wrong"
    return "same_fail"


def levenshtein(a: str, b: str) -> int:
    """Exact Levenshtein distance with prefix/suffix trimming."""
    if a == b:
        return 0
    start = 0
    min_len = min(len(a), len(b))
    while start < min_len and a[start] == b[start]:
        start += 1
    a = a[start:]
    b = b[start:]
    while a and b and a[-1] == b[-1]:
        a = a[:-1]
        b = b[:-1]
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (ca != cb)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def line_edit_stats(a: str, b: str) -> dict[str, int]:
    import difflib

    a_lines = a.splitlines()
    b_lines = b.splitlines()
    stats = {
        "line_equal": 0,
        "line_replace": 0,
        "line_insert": 0,
        "line_delete": 0,
        "line_changed": 0,
    }
    matcher = difflib.SequenceMatcher(a=a_lines, b=b_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            stats["line_equal"] += i2 - i1
            continue
        if tag == "replace":
            stats["line_replace"] += max(i2 - i1, j2 - j1)
        elif tag == "insert":
            stats["line_insert"] += j2 - j1
        elif tag == "delete":
            stats["line_delete"] += i2 - i1
        stats["line_changed"] += max(i2 - i1, j2 - j1)
    return stats


def parse_ok(src: str) -> bool:
    try:
        ast.parse(src or "")
        return True
    except SyntaxError:
        return False


def import_block(src: str) -> str:
    imports: list[str] = []
    for line in (src or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(stripped)
    return "\n".join(imports)


def first_function_signature(src: str) -> str | None:
    try:
        tree = ast.parse(src or "")
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
                return f"{node.name}({ast.unparse(node.args)}){ret}"

    for line in (src or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("async def "):
            return stripped.rstrip(":")
    return None


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def median(values: Iterable[float]) -> float | None:
    vals = list(values)
    return statistics.median(vals) if vals else None


def detect_gate_decision(gen: dict[str, Any]) -> str:
    if gen.get("skip_refine"):
        return "skipped"
    if gen.get("gate_min_mask_fraction") is not None or gen.get("gate_max_mask_fraction") is not None:
        return "kept"
    return "no_gate"


def build_long_rows(
    dataset: str,
    baseline_jsonl: dict[str, dict[str, Any]],
    baseline_eval: dict[str, dict[str, bool]],
    runs: list[RunSpec],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        run_jsonl = load_jsonl_by_task(run.jsonl_path)
        run_eval = load_eval_passes(run.eval_path)
        task_ids = sorted(set(baseline_jsonl) | set(baseline_eval) | set(run_jsonl) | set(run_eval))
        for task_id in task_ids:
            base_rec = baseline_jsonl.get(task_id, {})
            run_rec = run_jsonl.get(task_id, {})
            base_status = baseline_eval.get(task_id, {})
            run_status = run_eval.get(task_id, {})

            base_plus = bool(base_status.get("plus_pass"))
            run_plus = bool(run_status.get("plus_pass"))
            base_base = bool(base_status.get("base_pass"))
            run_base = bool(run_status.get("base_pass"))

            draft = run_rec.get("draft_completion")
            if not isinstance(draft, str):
                draft = base_rec.get("raw_completion") or base_rec.get("solution") or ""
            refined = run_rec.get("raw_completion")
            if not isinstance(refined, str):
                refined = run_rec.get("solution") or ""
            base_solution = base_rec.get("solution") or base_rec.get("raw_completion") or ""
            run_solution = run_rec.get("solution") or refined

            gen = run_rec.get("gen") or {}
            if not isinstance(gen, dict):
                gen = {}

            edit_distance = levenshtein(draft, refined)
            base_raw = base_rec.get("raw_completion") or ""
            raw_vs_baseline_distance = levenshtein(base_raw, refined)
            solution_vs_baseline_distance = levenshtein(base_solution, run_solution)
            line_stats = line_edit_stats(draft, refined)
            draft_sig = first_function_signature(draft)
            refined_sig = first_function_signature(refined)
            draft_imports = import_block(draft)
            refined_imports = import_block(refined)

            row = {
                "dataset": dataset,
                "task_id": task_id,
                "run_name": run.name,
                "baseline_plus_pass": int(base_plus),
                "run_plus_pass": int(run_plus),
                "plus_transition": transition_name(base_plus, run_plus),
                "baseline_base_pass": int(base_base),
                "run_base_pass": int(run_base),
                "base_transition": transition_name(base_base, run_base),
                "gate_decision": detect_gate_decision(gen),
                "skip_refine": int(bool(gen.get("skip_refine"))),
                "skip_reason": gen.get("skip_reason") or "",
                "mask_fraction": to_float(gen.get("mask_fraction")),
                "mask_tokens": gen.get("mask_tokens"),
                "draft_tokens": gen.get("draft_tokens"),
                "confidence_mean": to_float(gen.get("confidence_mean")),
                "confidence_min": to_float(gen.get("confidence_min")),
                "confidence_max": to_float(gen.get("confidence_max")),
                "gate_min_mask_fraction": to_float(gen.get("gate_min_mask_fraction")),
                "gate_max_mask_fraction": to_float(gen.get("gate_max_mask_fraction")),
                "mask_ratio": to_float(gen.get("mask_ratio")),
                "mask_granularity": gen.get("mask_granularity") or "",
                "span_merge_gap": gen.get("span_merge_gap"),
                "materialized_from": gen.get("materialized_from") or "",
                "char_edit_distance": edit_distance,
                "char_edit_ratio": edit_distance / max(len(draft), len(refined), 1),
                "raw_vs_baseline_char_edit_distance": raw_vs_baseline_distance,
                "raw_vs_baseline_char_edit_ratio": raw_vs_baseline_distance / max(len(base_raw), len(refined), 1),
                "solution_vs_baseline_char_edit_distance": solution_vs_baseline_distance,
                "solution_changed_without_raw_change": int(base_raw == refined and base_solution != run_solution),
                "line_changed": line_stats["line_changed"],
                "line_replace": line_stats["line_replace"],
                "line_insert": line_stats["line_insert"],
                "line_delete": line_stats["line_delete"],
                "changed_signature": int(draft_sig != refined_sig),
                "draft_signature": draft_sig or "",
                "refined_signature": refined_sig or "",
                "changed_imports": int(draft_imports != refined_imports),
                "draft_parse_ok": int(parse_ok(draft)),
                "refined_parse_ok": int(parse_ok(refined)),
                "baseline_solution_parse_ok": int(parse_ok(base_solution)),
                "run_solution_parse_ok": int(parse_ok(run_solution)),
                "empty_refined": int(not bool(refined.strip())),
                "no_raw_change": int(draft == refined),
                "same_raw_as_baseline": int(base_raw == refined),
                "raw_same_outcome_changed": int(base_raw == refined and transition_name(base_plus, run_plus) in ("wrong_to_correct", "correct_to_wrong")),
            }
            row["row_mechanism_label"] = row_mechanism_label(row)
            rows.append(row)
    return rows


def row_mechanism_label(row: dict[str, Any]) -> str:
    transition = row["plus_transition"]
    gate = row["gate_decision"]
    no_change = bool(row["no_raw_change"])
    if row.get("raw_same_outcome_changed"):
        return "packaging_or_eval_artifact"
    if gate == "skipped":
        if transition == "same_pass":
            return "gate_safe_skip"
        if transition == "same_fail":
            return "gate_skipped_failed"
        return "gate_changed_outcome_candidate"
    if no_change:
        return f"no_raw_change_{transition}"
    if transition == "wrong_to_correct":
        return "rewrite_helped"
    if transition == "correct_to_wrong":
        return "rewrite_hurt"
    if transition == "same_fail":
        return "rewrite_ineffective"
    return "rewrite_safe"


def role_map(role_args: list[list[str]] | None) -> dict[str, set[str]]:
    roles: dict[str, set[str]] = defaultdict(set)
    for role, name in role_args or []:
        roles[role].add(name)
    return roles


def select_rows(rows: list[dict[str, Any]], names: set[str]) -> list[dict[str, Any]]:
    if not names:
        return []
    return [r for r in rows if r["run_name"] in names]


def infer_role_rows(task_rows: list[dict[str, Any]], roles: dict[str, set[str]]) -> dict[str, list[dict[str, Any]]]:
    out = {role: select_rows(task_rows, names) for role, names in roles.items()}
    if "fresh" not in out:
        out["fresh"] = [r for r in task_rows if "fresh" in r["run_name"]]
    if "offpolicy" not in out:
        out["offpolicy"] = [
            r for r in task_rows
            if "gate" in r["run_name"] and "fresh" not in r["run_name"] and r["materialized_from"]
        ]
    if "gate" not in out:
        out["gate"] = [r for r in task_rows if r["gate_decision"] in ("skipped", "kept")]
    if "nogate" not in out:
        out["nogate"] = [r for r in task_rows if r["gate_decision"] == "no_gate"]
    return out


def add_category(categories: list[str], name: str) -> None:
    if name not in categories:
        categories.append(name)


def build_task_rows(long_rows: list[dict[str, Any]], roles: dict[str, set[str]]) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in long_rows:
        by_task[row["task_id"]].append(row)

    out: list[dict[str, Any]] = []
    for task_id, task_rows in sorted(by_task.items()):
        dataset = task_rows[0]["dataset"]
        baseline_plus = bool(task_rows[0]["baseline_plus_pass"])
        role_rows = infer_role_rows(task_rows, roles)
        transitions = Counter(r["plus_transition"] for r in task_rows)
        categories: list[str] = []

        def artifact(row: dict[str, Any]) -> bool:
            return bool(row.get("raw_same_outcome_changed"))

        any_artifact = any(artifact(r) for r in task_rows)
        any_nogate_hurt = any(
            r["plus_transition"] == "correct_to_wrong" and not artifact(r)
            for r in role_rows["nogate"]
        )
        any_nogate_packaging_hurt = any(
            r["plus_transition"] == "correct_to_wrong" and artifact(r)
            for r in role_rows["nogate"]
        )
        any_gate_safe_skip = any(
            r["gate_decision"] == "skipped" and r["plus_transition"] == "same_pass"
            for r in role_rows["gate"]
        )
        if any_nogate_hurt and any_gate_safe_skip:
            add_category(categories, "gate_prevented_harm_candidate")
        if any_nogate_packaging_hurt and any_gate_safe_skip:
            add_category(categories, "gate_prevented_packaging_harm_candidate")

        if any(r["plus_transition"] == "wrong_to_correct" and not artifact(r) for r in role_rows["offpolicy"]) and any(
            r["plus_transition"] == "same_fail" for r in role_rows["fresh"]
        ):
            add_category(categories, "offpolicy_fix_not_reproduced_candidate")
        if any(r["plus_transition"] == "wrong_to_correct" and artifact(r) for r in role_rows["offpolicy"]) and any(
            r["plus_transition"] == "same_fail" for r in role_rows["fresh"]
        ):
            add_category(categories, "offpolicy_packaging_fix_not_reproduced_candidate")

        if any(
            r["gate_decision"] == "kept" and r["plus_transition"] == "correct_to_wrong"
            and not artifact(r)
            for r in role_rows["gate"]
        ):
            add_category(categories, "low_disagreement_hurt_candidate")
        if any(
            r["gate_decision"] == "kept" and r["plus_transition"] == "correct_to_wrong"
            and artifact(r)
            for r in role_rows["gate"]
        ):
            add_category(categories, "low_disagreement_packaging_hurt_candidate")

        if any(r["plus_transition"] == "wrong_to_correct" and not artifact(r) for r in role_rows["nogate"]) and any(
            r["gate_decision"] == "skipped" and r["plus_transition"] == "same_fail"
            for r in role_rows["gate"]
        ):
            add_category(categories, "gate_overconservative_candidate")
        if any(r["plus_transition"] == "wrong_to_correct" and artifact(r) for r in role_rows["nogate"]) and any(
            r["gate_decision"] == "skipped" and r["plus_transition"] == "same_fail"
            for r in role_rows["gate"]
        ):
            add_category(categories, "gate_overconservative_packaging_candidate")

        if not baseline_plus and not any(r["plus_transition"] == "wrong_to_correct" and not artifact(r) for r in task_rows):
            if any(r["gate_decision"] != "skipped" for r in task_rows):
                add_category(categories, "baseline_wrong_rewrite_ineffective")

        if baseline_plus and not any(r["plus_transition"] == "correct_to_wrong" and not artifact(r) for r in task_rows):
            add_category(categories, "baseline_correct_no_observed_harm")

        if any_artifact:
            add_category(categories, "packaging_or_eval_artifact_candidate")

        if any(r["plus_transition"] == "wrong_to_correct" and not artifact(r) for r in task_rows) and any(
            r["plus_transition"] == "same_fail" for r in task_rows
        ):
            add_category(categories, "rewrite_stochastic_candidate")

        if not categories:
            add_category(categories, "uncategorized")

        mask_values = [r["mask_fraction"] for r in task_rows if r["mask_fraction"] is not None]
        edit_ratios = [r["char_edit_ratio"] for r in task_rows if r["char_edit_ratio"] is not None]
        evidence = "; ".join(
            f"{r['run_name']}:{r['plus_transition']}/{r['gate_decision']}"
            for r in sorted(task_rows, key=lambda x: x["run_name"])
        )
        out.append({
            "dataset": dataset,
            "task_id": task_id,
            "baseline_plus_pass": int(baseline_plus),
            "categories": "|".join(categories),
            "n_categories": len(categories),
            "n_runs": len(task_rows),
            "n_wrong_to_correct": transitions["wrong_to_correct"],
            "n_correct_to_wrong": transitions["correct_to_wrong"],
            "n_same_pass": transitions["same_pass"],
            "n_same_fail": transitions["same_fail"],
            "min_mask_fraction": min(mask_values) if mask_values else None,
            "median_mask_fraction": median(mask_values),
            "max_mask_fraction": max(mask_values) if mask_values else None,
            "median_char_edit_ratio": median(edit_ratios),
            "evidence": evidence,
        })
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row.get(k) is None else row.get(k) for k in fieldnames})


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        if value is None:
            return ""
        return str(value).replace("\n", " ")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    return "\n".join(lines)


def run_summary_rows(long_rows: list[dict[str, Any]]) -> list[list[Any]]:
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in long_rows:
        by_run[row["run_name"]].append(row)
    rows = []
    for run_name, rows_for_run in sorted(by_run.items()):
        n = len(rows_for_run)
        plus = sum(int(r["run_plus_pass"]) for r in rows_for_run)
        baseline_plus = sum(int(r["baseline_plus_pass"]) for r in rows_for_run)
        transitions = Counter(r["plus_transition"] for r in rows_for_run)
        skipped = sum(int(r["gate_decision"] == "skipped") for r in rows_for_run)
        kept = sum(int(r["gate_decision"] == "kept") for r in rows_for_run)
        masks = [r["mask_fraction"] for r in rows_for_run if r["mask_fraction"] is not None]
        edit = [r["char_edit_ratio"] for r in rows_for_run]
        rows.append([
            run_name,
            n,
            f"{plus / n * 100:.1f}" if n else "",
            f"{(plus - baseline_plus) / n * 100:+.1f}" if n else "",
            transitions["wrong_to_correct"],
            transitions["correct_to_wrong"],
            transitions["same_pass"],
            transitions["same_fail"],
            skipped,
            kept,
            fmt_float(statistics.mean(masks) if masks else None),
            fmt_float(median(edit)),
        ])
    return rows


MASK_BINS = [
    (0.0, 0.008),
    (0.008, 0.010),
    (0.010, 0.012),
    (0.012, 0.015),
    (0.015, 0.020),
    (0.020, 0.030),
    (0.030, 0.050),
    (0.050, math.inf),
]


def bin_label(lo: float, hi: float) -> str:
    if hi == math.inf:
        return f"[{lo:.3f}, inf)"
    return f"[{lo:.3f}, {hi:.3f})"


def mask_bin_rows(long_rows: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in long_rows:
        if row["mask_fraction"] is not None:
            by_run[row["run_name"]].append(row)
    for run_name, run_rows in sorted(by_run.items()):
        for lo, hi in MASK_BINS:
            members = [
                r for r in run_rows
                if r["mask_fraction"] is not None and lo <= r["mask_fraction"] < hi
            ]
            if not members:
                continue
            transitions = Counter(r["plus_transition"] for r in members)
            rows.append([
                run_name,
                bin_label(lo, hi),
                len(members),
                transitions["wrong_to_correct"],
                transitions["correct_to_wrong"],
                transitions["same_pass"],
                transitions["same_fail"],
                f"{transitions['wrong_to_correct'] - transitions['correct_to_wrong']:+d}",
            ])
    return rows


def edit_summary_rows(long_rows: list[dict[str, Any]]) -> list[list[Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in long_rows:
        grouped[(row["run_name"], row["plus_transition"])].append(row)
    rows = []
    for (run_name, transition), members in sorted(grouped.items()):
        ratios = [float(r["char_edit_ratio"]) for r in members]
        line_changes = [float(r["line_changed"]) for r in members]
        rows.append([
            run_name,
            transition,
            len(members),
            fmt_float(median(ratios)),
            fmt_float(percentile(ratios, 0.9)),
            fmt_float(median(line_changes), digits=2),
            sum(int(r["changed_signature"]) for r in members),
            sum(int(r["changed_imports"]) for r in members),
            sum(1 - int(r["refined_parse_ok"]) for r in members),
        ])
    return rows


def category_summary_rows(task_rows: list[dict[str, Any]], max_examples: int) -> list[list[Any]]:
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in task_rows:
        for cat in str(row["categories"]).split("|"):
            by_cat[cat].append(row)
    rows = []
    for cat, members in sorted(by_cat.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        members = sorted(
            members,
            key=lambda r: (
                -(int(r["n_wrong_to_correct"]) + int(r["n_correct_to_wrong"])),
                r["task_id"],
            ),
        )
        examples = ", ".join(r["task_id"] for r in members[:max_examples])
        rows.append([cat, len(members), examples])
    return rows


def candidate_rows(task_rows: list[dict[str, Any]], max_per_category: int) -> list[list[Any]]:
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in task_rows:
        for cat in str(row["categories"]).split("|"):
            if cat == "uncategorized":
                continue
            by_cat[cat].append(row)
    rows = []
    for cat, members in sorted(by_cat.items()):
        members = sorted(
            members,
            key=lambda r: (
                -(int(r["n_wrong_to_correct"]) + int(r["n_correct_to_wrong"])),
                r["task_id"],
            ),
        )
        for row in members[:max_per_category]:
            rows.append([
                cat,
                row["task_id"],
                row["n_wrong_to_correct"],
                row["n_correct_to_wrong"],
                fmt_float(row["median_mask_fraction"]),
                fmt_float(row["median_char_edit_ratio"]),
                row["evidence"],
            ])
    return rows


def write_markdown_report(
    path: Path,
    dataset: str,
    baseline_jsonl_path: Path,
    baseline_eval_path: Path,
    runs: list[RunSpec],
    long_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    max_examples: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Remask Locator/Rewriter Case-Study Analysis",
        "",
        "This report is generated mechanically. Category labels are candidate labels for case-study selection, not final semantic judgments.",
        "",
        "## Inputs",
        "",
        f"- Dataset: `{dataset}`",
        f"- Baseline JSONL: `{baseline_jsonl_path}`",
        f"- Baseline eval: `{baseline_eval_path}`",
    ]
    for run in runs:
        lines.append(f"- Run `{run.name}`: `{run.jsonl_path}` / `{run.eval_path}`")

    lines.extend([
        "",
        "## Run-Level Outcomes",
        "",
        markdown_table(
            [
                "run", "n", "plus%", "delta plus pp", "wrong->correct",
                "correct->wrong", "same pass", "same fail", "skipped",
                "kept", "mean mask frac", "median edit ratio",
            ],
            run_summary_rows(long_rows),
        ),
        "",
        "## Candidate Category Counts",
        "",
        markdown_table(["category", "n tasks", "examples"], category_summary_rows(task_rows, max_examples)),
        "",
        "## Mask-Fraction Bins",
        "",
        markdown_table(
            ["run", "mask_fraction bin", "n", "wrong->correct", "correct->wrong", "same pass", "same fail", "net"],
            mask_bin_rows(long_rows),
        ),
        "",
        "## Edit Metrics By Transition",
        "",
        markdown_table(
            [
                "run", "transition", "n", "median char edit ratio",
                "p90 char edit ratio", "median changed lines",
                "changed signature", "changed imports", "refined parse fails",
            ],
            edit_summary_rows(long_rows),
        ),
        "",
        "## Deterministic Case Candidates",
        "",
        markdown_table(
            [
                "category", "task", "wrong->correct runs", "correct->wrong runs",
                "median mask frac", "median edit ratio", "evidence",
            ],
            candidate_rows(task_rows, max_examples),
        ),
        "",
        "## Notes",
        "",
        "- `gate_prevented_harm_candidate`: a no-gate run breaks a baseline-correct task and a gated run skips it while preserving pass.",
        "- `gate_prevented_packaging_harm_candidate`: same pattern, but raw code is unchanged, so the observed harm is likely packaging/eval rather than rewriting.",
        "- `offpolicy_fix_not_reproduced_candidate`: an off-policy/materialized gate run fixes a baseline-failing task but a fresh run does not.",
        "- `offpolicy_packaging_fix_not_reproduced_candidate`: same pattern, but raw code is unchanged, so it should be audited as packaging/eval first.",
        "- `low_disagreement_hurt_candidate`: a gated run keeps refinement and still causes `correct->wrong`.",
        "- `low_disagreement_packaging_hurt_candidate`: same pattern, but raw code is unchanged.",
        "- `gate_overconservative_candidate`: a no-gate run fixes a baseline-failing task but a gated run skips it.",
        "- `gate_overconservative_packaging_candidate`: same pattern, but the apparent no-gate fix is likely packaging/eval.",
        "- `packaging_or_eval_artifact_candidate`: raw output is unchanged from baseline but eval outcome changes.",
        "- Locator span overlap cannot be judged from older artifacts unless mask spans were recorded during generation.",
    ])
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--baseline_jsonl", required=True, type=Path)
    ap.add_argument("--baseline_eval", required=True, type=Path)
    ap.add_argument(
        "--run",
        action="append",
        nargs=3,
        metavar=("NAME", "JSONL", "EVAL"),
        default=[],
        help="Run spec. Repeat as needed.",
    )
    ap.add_argument(
        "--role",
        action="append",
        nargs=2,
        metavar=("ROLE", "RUN_NAME"),
        default=[],
        help="Optional role binding, e.g. --role offpolicy dream_gate012 --role fresh dream_gate012_fresh.",
    )
    ap.add_argument("--out_csv", required=True, type=Path)
    ap.add_argument("--out_task_csv", required=True, type=Path)
    ap.add_argument("--out_md", required=True, type=Path)
    ap.add_argument("--max_examples", type=int, default=8)
    args = ap.parse_args()
    if not args.run:
        ap.error("At least one --run NAME JSONL EVAL is required.")
    return args


def main() -> None:
    args = parse_args()
    runs = [RunSpec(name=r[0], jsonl_path=Path(r[1]), eval_path=Path(r[2])) for r in args.run]
    baseline_jsonl = load_jsonl_by_task(args.baseline_jsonl)
    baseline_eval = load_eval_passes(args.baseline_eval)
    long_rows = build_long_rows(args.dataset, baseline_jsonl, baseline_eval, runs)
    task_rows = build_task_rows(long_rows, role_map(args.role))

    write_csv(args.out_csv, long_rows)
    write_csv(args.out_task_csv, task_rows)
    write_markdown_report(
        path=args.out_md,
        dataset=args.dataset,
        baseline_jsonl_path=args.baseline_jsonl,
        baseline_eval_path=args.baseline_eval,
        runs=runs,
        long_rows=long_rows,
        task_rows=task_rows,
        max_examples=args.max_examples,
    )
    print(f"Wrote {len(long_rows)} run-task rows to {args.out_csv}")
    print(f"Wrote {len(task_rows)} task category rows to {args.out_task_csv}")
    print(f"Wrote report to {args.out_md}")


if __name__ == "__main__":
    main()

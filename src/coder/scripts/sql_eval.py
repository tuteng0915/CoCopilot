#!/usr/bin/env python3
"""
Minimal Spider execution-accuracy evaluator.

Examples:
  python -m coder.scripts.sql_eval \
    --pred outputs/sql_feasibility/deepseek_spider_dev.jsonl \
    --spider_dir outputs/sql_feasibility/spider \
    > outputs/sql_feasibility/deepseek_spider_dev_eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sqlite3
import sys
from typing import Any


PRED_SQL_FIELDS = ("pred_sql", "raw_completion", "raw_sql", "sql", "completion")
GOLD_SQL_FIELDS = ("gold_sql", "query")


def _jsonl_records(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"[warn] skipping non-JSON line {line_no} in {path}: {exc}",
                    file=sys.stderr,
                )


def _normalize_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bytes):
        return value.hex().lower()
    return str(value).lower()


def execute_sql(db_path: str, sql: str) -> set[tuple[str, ...]] | None:
    """Execute SQL and return a set of normalized rows, or None on error."""
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
        return {tuple(_normalize_value(v) for v in row) for row in rows}
    except Exception:
        return None


def schema_to_text(tables_info: dict) -> str:
    """Convert one Spider tables.json entry to compact SQLite DDL text."""
    lines: list[str] = []
    col_names = tables_info["column_names_original"]
    col_types = tables_info["column_types"]
    table_names = tables_info["table_names_original"]
    primary_keys = set(tables_info.get("primary_keys") or [])

    for table_idx, table_name in enumerate(table_names):
        cols: list[str] = []
        for col_idx, ((col_table_idx, col_name), col_type) in enumerate(
            zip(col_names, col_types)
        ):
            if col_table_idx != table_idx:
                continue
            pk = " PRIMARY KEY" if col_idx in primary_keys else ""
            cols.append(f"{col_name} {col_type}{pk}")
        lines.append(f"CREATE TABLE {table_name} ({', '.join(cols)});")

    for c1_idx, c2_idx in tables_info.get("foreign_keys", []):
        t1_idx, c1 = col_names[c1_idx]
        t2_idx, c2 = col_names[c2_idx]
        if t1_idx < 0 or t2_idx < 0:
            continue
        lines.append(f"-- FK: {table_names[t1_idx]}.{c1} -> {table_names[t2_idx]}.{c2}")

    return "\n".join(lines)


def make_prompt(schema_text: str, question: str) -> str:
    return (
        "Given the following SQLite database schema:\n"
        f"{schema_text}\n\n"
        f"Write a single SQL query to answer: {question}\n"
        "Output only the SQL query, no explanation.\n"
    )


def extract_sql(raw: str) -> str:
    """Strip common markdown/prose wrappers around a generated SQL query."""
    text = (raw or "")
    # Some code tokenizers expose byte-level whitespace markers in decoded text.
    text = text.replace("\u0120", " ").replace("\u010a", "\n").replace("\u0109", "\t")
    text = text.strip()
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.I | re.S)
    if fence:
        text = fence.group(1).strip()
    else:
        text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text).strip()

    text = re.sub(r"^(?:sql\s*(?:query)?|query)\s*:\s*", "", text, flags=re.I).strip()

    # Keep the first plausible SQL statement when the model adds prose after it.
    m = re.search(r"\b(select|with)\b", text, flags=re.I)
    if m:
        text = text[m.start():].strip()
    if ";" in text:
        text = text.split(";", 1)[0].strip() + ";"
    text = _repair_sql_spacing(text)
    return text


def _repair_sql_spacing(sql: str) -> str:
    """Repair common tokenizer-induced SQL keyword/identifier spacing loss."""
    text = re.sub(r"\s+", " ", sql).strip()

    # SELECTCOUNT(*) -> SELECT COUNT(*), FROMtable -> FROM table.
    # Keep this case-sensitive to avoid breaking identifiers like Country.
    leading_keywords = "SELECT|FROM|WHERE|HAVING|LIMIT|JOIN|ORDER|GROUP"
    text = re.sub(
        rf"(^|[\s,(])({leading_keywords})(?=[A-Za-z_(])",
        lambda m: f"{m.group(1)}{m.group(2)} ",
        text,
    )

    # colFROMtable -> col FROM table, valueWHEREcol -> value WHERE col.
    trailing_keywords = "FROM|WHERE|HAVING|LIMIT|JOIN|ORDER|GROUP|DESC|ASC"
    text = re.sub(
        rf"([A-Za-z0-9_)])(?=({trailing_keywords})(?:[A-Za-z_(]|\s|$))",
        lambda m: f"{m.group(1)} ",
        text,
    )
    text = re.sub(
        rf"(^|[\s,(])({leading_keywords})(?=[A-Za-z_(])",
        lambda m: f"{m.group(1)}{m.group(2)} ",
        text,
    )

    # SQLite accepts aliases with AS or whitespace; fix )asAlias glue.
    text = re.sub(r"(?<=[A-Za-z0-9_)])as(?=[A-Z_])", " AS ", text)
    text = re.sub(r"(?<=[A-Za-z0-9_)])AS(?=[A-Za-z_])", " AS ", text)

    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" ;", ";").replace("( ", "(").replace(" )", ")")
    return text


def _load_spider_dev(spider_dir: pathlib.Path) -> list[dict]:
    dev_path = spider_dir / "dev.json"
    if not dev_path.exists():
        return []
    return json.loads(dev_path.read_text(encoding="utf-8"))


def _lookup_from_task_id(rec: dict, dev_data: list[dict]) -> dict | None:
    task_id = str(rec.get("task_id") or "")
    match = re.match(r"^spider/dev/(\d+)$", task_id)
    if not match:
        return None
    idx = int(match.group(1))
    if idx < 0 or idx >= len(dev_data):
        return None
    return dev_data[idx]


def _first_present(rec: dict, fields: tuple[str, ...]) -> str | None:
    for field in fields:
        value = rec.get(field)
        if value is not None:
            return str(value)
    return None


def _resolve_sql_record(rec: dict, dev_data: list[dict]) -> tuple[str, str, str]:
    spider_item = _lookup_from_task_id(rec, dev_data)

    db_id = rec.get("db_id")
    if db_id is None and spider_item is not None:
        db_id = spider_item.get("db_id")

    gold_sql = _first_present(rec, GOLD_SQL_FIELDS)
    if gold_sql is None and spider_item is not None:
        gold_sql = spider_item.get("query")

    pred_sql = _first_present(rec, PRED_SQL_FIELDS)

    missing = [
        name
        for name, value in (("db_id", db_id), ("gold_sql", gold_sql), ("pred_sql", pred_sql))
        if value is None
    ]
    if missing:
        task_id = rec.get("task_id", "<unknown>")
        raise ValueError(f"{task_id}: missing required field(s): {', '.join(missing)}")

    return str(db_id), str(gold_sql), extract_sql(str(pred_sql))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True, help="Input prediction JSONL.")
    ap.add_argument("--spider_dir", required=True, help="Path to Spider directory.")
    ap.add_argument(
        "--out",
        default=None,
        help="Optional output JSONL. Defaults to stdout for shell redirection.",
    )
    ap.add_argument(
        "--summary_out",
        default=None,
        help="Optional JSON summary path for docs/results generation.",
    )
    args = ap.parse_args()

    pred_path = pathlib.Path(args.pred)
    spider_dir = pathlib.Path(args.spider_dir)
    dev_data = _load_spider_dev(spider_dir)

    out_f = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    close_out = out_f is not sys.stdout

    n_pass = 0
    n_total = 0
    n_pred_errors = 0
    n_gold_errors = 0

    try:
        for rec in _jsonl_records(pred_path):
            db_id, gold_sql, pred_sql = _resolve_sql_record(rec, dev_data)
            db_path = spider_dir / "database" / db_id / f"{db_id}.sqlite"

            pred_result = execute_sql(str(db_path), pred_sql)
            gold_result = execute_sql(str(db_path), gold_sql)
            passed = pred_result is not None and pred_result == gold_result

            out_rec = dict(rec)
            out_rec.update({
                "db_id": db_id,
                "gold_sql": gold_sql,
                "pred_sql": pred_sql,
                "exec_pass": passed,
                "pred_exec_error": pred_result is None,
                "gold_exec_error": gold_result is None,
                "pred_result_n_rows": len(pred_result) if pred_result is not None else None,
                "gold_result_n_rows": len(gold_result) if gold_result is not None else None,
            })
            print(json.dumps(out_rec, ensure_ascii=False), file=out_f)

            n_pass += int(passed)
            n_pred_errors += int(pred_result is None)
            n_gold_errors += int(gold_result is None)
            n_total += 1
    finally:
        if close_out:
            out_f.close()

    accuracy = (n_pass / n_total) if n_total else 0.0
    summary = {
        "script": "sql_eval",
        "pred": str(pred_path),
        "spider_dir": str(spider_dir),
        "n_pass": n_pass,
        "n_total": n_total,
        "execution_accuracy": accuracy,
        "n_pred_exec_errors": n_pred_errors,
        "n_gold_exec_errors": n_gold_errors,
    }

    print(
        f"=== Execution Accuracy: {n_pass}/{n_total} = {accuracy * 100:.1f}%",
        file=sys.stderr,
        flush=True,
    )

    if args.summary_out:
        summary_path = pathlib.Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

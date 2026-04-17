"""Evaluate WildBench creative writing outputs with checklist LLM judging."""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from tqdm import tqdm

from coder.models import ApiCoder
from coder.utils.schema import ModelRequest


class Judge(Protocol):
    def ask(self, prompt: str) -> str:
        ...


class AnthropicJudge:
    def __init__(self, model_id: str, timeout_sec: int = 120):
        self.model_id = model_id
        self.timeout_sec = timeout_sec
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise SystemExit("ANTHROPIC_API_KEY is required for eval_writing Claude judging.")

    def ask(self, prompt: str) -> str:
        payload = {
            "model": self.model_id,
            "max_tokens": 8,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
        req = urllib.request.Request(
            url="https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Anthropic judge request failed: HTTP {e.code} {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Anthropic judge request failed: {e.reason}") from e

        parts = data.get("content") or []
        texts = []
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(str(part.get("text", "")))
        return "".join(texts).strip()


class ApiCoderJudge:
    def __init__(self):
        if not os.getenv("CODER_API_KEY") and os.getenv("ANTHROPIC_API_KEY"):
            os.environ["CODER_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]
        os.environ.setdefault("CODER_API_SYSTEM_PROMPT", "Answer only YES or NO.")
        self.model = ApiCoder(model_id=None, device="api")

    def ask(self, prompt: str) -> str:
        return self.model.generate(ModelRequest(
            prompt=prompt,
            max_new_tokens=8,
            temperature=0.0,
            top_p=1.0,
            seed=None,
        ))


def read_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize_checklist(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        criteria: list[str] = []
        for item in value:
            criteria.extend(normalize_checklist(item))
        return [c for c in criteria if c.strip()]
    if isinstance(value, dict):
        for key in ("criterion", "criteria", "text", "description", "content"):
            if key in value:
                return normalize_checklist(value[key])
        return [json.dumps(value, ensure_ascii=False)]
    return [str(value)]


def build_judge_prompt(criterion: str, response: str) -> str:
    return (
        "Given this creative writing response, does it satisfy the following "
        "criterion? Answer only YES or NO.\n"
        f"Criterion: {criterion}\n"
        f"Response: {response[:2000]}"
    )


def parse_yes_no(text: str) -> int:
    answer = (text or "").strip().upper()
    if answer.startswith("YES"):
        return 1
    if answer.startswith("NO"):
        return 0
    return 0


def build_judge(judge_model: str) -> Judge:
    if judge_model == "api":
        return ApiCoderJudge()
    return AnthropicJudge(model_id=judge_model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL file from gen_writing.py or gen_remask.py")
    ap.add_argument("--out", required=True, help="Path to write evaluation JSON")
    ap.add_argument("--judge_model", default="claude-sonnet-4-6")
    ap.add_argument("--limit", type=int, default=0, help="Only evaluate first N records. 0 = all.")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    records = read_jsonl(args.input)
    if args.limit > 0:
        records = records[: args.limit]

    judge = build_judge(args.judge_model)

    per_item: list[dict[str, Any]] = []
    total_score = 0.0

    for idx, rec in enumerate(tqdm(records, desc="judge:writing")):
        criteria = normalize_checklist(rec.get("checklist", []))
        response = str(rec.get("raw_completion", ""))
        n_pass = 0
        for criterion in criteria:
            judge_prompt = build_judge_prompt(criterion, response)
            verdict = judge.ask(judge_prompt)
            n_pass += parse_yes_no(verdict)

        n_criteria = len(criteria)
        score = (n_pass / n_criteria) if n_criteria else 0.0
        total_score += score
        per_item.append({
            "id": rec.get("id", f"wildbench_writing/{idx}"),
            "score": score,
            "n_criteria": n_criteria,
            "n_pass": n_pass,
        })

    n_total = len(records)
    summary = {
        "dataset": "wildbench_writing",
        "model": records[0].get("model", "unknown") if records else "unknown",
        "judge_model": args.judge_model,
        "n_total": n_total,
        "checklist_pass_rate": (total_score / n_total) if n_total else 0.0,
        "per_item": per_item,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[eval] records={n_total}")
    print(f"[eval] checklist_pass_rate={summary['checklist_pass_rate']:.4f}")
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()

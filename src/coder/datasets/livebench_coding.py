# src/coder/datasets/livebench_coding.py
from __future__ import annotations

import base64
import json
import pickle
import random
import zlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset


def translate_private_test_cases(encoded_data: str) -> Any:
    """
    Decode private_test_cases used by LiveBench/LiveCodeBench style datasets:
    base64 -> zlib -> pickle -> json
    (Same idea as lighteval implementation.)
    """
    decoded_data = base64.b64decode(encoded_data)
    decompressed_data = zlib.decompress(decoded_data)
    original_data = pickle.loads(decompressed_data)
    return json.loads(original_data)


def _maybe_json_loads(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return x
    return x


def _parse_meta_func_name(row: Dict[str, Any]) -> Optional[str]:
    """
    Try to find the function name for 'functional' (LeetCode-style) tasks.
    Commonly stored in meta_data as {"func_name": "..."}.
    """
    meta = row.get("meta_data", None)
    meta = _maybe_json_loads(meta)
    if isinstance(meta, dict):
        fn = meta.get("func_name") or meta.get("fn_name")
        if isinstance(fn, str) and fn.strip():
            return fn.strip()
    return None


def _safe_parse_test_cases(x: Any, is_private: bool) -> List[Dict[str, Any]]:
    """
    public_test_cases: usually a JSON string/list of {"input","output","testtype"}
    private_test_cases: usually encoded string, decode to list
    """
    if x is None:
        return []

    if isinstance(x, list):
        return x

    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []

        # If already JSON text
        if s.startswith("[") or s.startswith("{"):
            try:
                obj = json.loads(s)
                return obj if isinstance(obj, list) else [obj]
            except Exception:
                return []

        # Else: private encoded blob
        if is_private:
            try:
                obj = translate_private_test_cases(s)
                return obj if isinstance(obj, list) else [obj]
            except Exception:
                return []

    return []


@dataclass(frozen=True)
class LiveBenchProblem:
    task_id: str
    title: str
    content: str
    platform: str
    difficulty: str
    func_name: Optional[str]
    starter_code: str
    public_tests: List[Dict[str, Any]]
    private_tests: List[Dict[str, Any]]

    @property
    def has_functional(self) -> bool:
        for t in (self.public_tests + self.private_tests):
            if (t.get("testtype") or "").lower() == "functional":
                return True
        return False

    @property
    def has_stdin(self) -> bool:
        for t in (self.public_tests + self.private_tests):
            if (t.get("testtype") or "").lower() == "stdin":
                return True
        return False


def build_prompt(problem: LiveBenchProblem) -> str:
    """
    Unified prompt template for both models.
    - If functional: ask for `class Solution` with method `func_name` (or starter_code).
    - If stdin: ask for a complete program reading stdin/writing stdout.
    """
    header = (
        "### Instructions:\n"
        "You are an expert Python programmer.\n"
        "Write a correct and efficient solution in Python 3.\n"
        "Return ONLY the code (no explanation, no markdown).\n\n"
    )

    task = (
        "### Problem:\n"
        f"Title: {problem.title}\n"
        f"Platform: {problem.platform}\n"
        f"Difficulty: {problem.difficulty}\n\n"
        f"{problem.content}\n\n"
    )

    guidance_lines = ["### Output Requirements:"]
    if problem.has_functional:
        if problem.func_name:
            guidance_lines.append(
                f"- Implement LeetCode-style solution: class Solution with method `{problem.func_name}`."
            )
        else:
            guidance_lines.append(
                "- Implement LeetCode-style solution: class Solution with the correct method name."
            )
        guidance_lines.append("- Do NOT read from stdin in functional mode unless the statement requires it.")
    if problem.has_stdin:
        guidance_lines.append("- Implement a full program that reads from stdin and writes to stdout.")
    guidance_lines.append("- You may use standard libraries.")
    guidance = "\n".join(guidance_lines) + "\n\n"

    starter = ""
    if problem.starter_code and problem.starter_code.strip():
        starter = "### Starter Code (use/complete if helpful):\n" + problem.starter_code.strip() + "\n\n"

    return header + task + guidance + starter


def load_livebench_coding(
    split: str = "test",
    limit: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 0,
) -> List[LiveBenchProblem]:
    ds = load_dataset("livebench/coding", split=split)

    rows = list(ds)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)

    if limit is not None:
        rows = rows[: int(limit)]

    problems: List[LiveBenchProblem] = []
    for idx, row in enumerate(rows):
        title = str(row.get("question_title", "")).strip()
        content = str(row.get("question_content", "")).strip()
        platform = str(row.get("platform", "unknown")).strip()
        difficulty = str(row.get("difficulty", "unknown")).strip()
        starter_code = str(row.get("starter_code", "") or "").rstrip()

        public_tests = _safe_parse_test_cases(row.get("public_test_cases"), is_private=False)
        private_tests = _safe_parse_test_cases(row.get("private_test_cases"), is_private=True)

        # task_id: prefer stable hash/id if exists; else fallback to idx
        task_id = (
            str(row.get("question_id") or row.get("task_id") or row.get("id") or "").strip()
            or f"livebench_coding_{idx}"
        )

        func_name = _parse_meta_func_name(row)

        problems.append(
            LiveBenchProblem(
                task_id=task_id,
                title=title,
                content=content,
                platform=platform,
                difficulty=difficulty,
                func_name=func_name,
                starter_code=starter_code,
                public_tests=public_tests,
                private_tests=private_tests,
            )
        )

    return problems

from __future__ import annotations

import re

CODE_START_RE = re.compile(r"(?m)^(def|class|import|from|@)\s+")
FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", flags=re.S | re.I)


def clean_model_completion(text: str, prompt: str | None = None) -> str:
    if not text:
        return ""

    s = text.strip()

    prompt_text = (prompt or "").strip()
    if prompt_text and s.startswith(prompt_text):
        s = s[len(prompt_text):].lstrip()

    fence_blocks = FENCE_RE.findall(s)
    if fence_blocks:
        s = fence_blocks[-1].strip()

    s = s.replace("```python", "").replace("```", "").strip()
    s = re.sub(r"^\s*(assistant|response)\s*:\s*", "", s, flags=re.I)

    match = CODE_START_RE.search(s)
    if match:
        s = s[match.start():].lstrip()

    return s.strip()


def starts_like_code(text: str) -> bool:
    return bool(re.match(r"^\s*(def|class|import|from|@)\s+", text or ""))


def indent_as_body(completion: str, spaces: int = 4) -> str:
    pad = " " * spaces
    out_lines = []
    for line in completion.splitlines():
        if line.strip():
            if line.startswith((" ", "\t")):
                out_lines.append(line.rstrip())
            else:
                out_lines.append(pad + line.rstrip())
        else:
            out_lines.append("")
    return "\n".join(out_lines).rstrip()


def extract_last_fenced_code(text: str) -> str | None:
    if not text:
        return None
    fence_blocks = FENCE_RE.findall(text)
    if not fence_blocks:
        return None
    return fence_blocks[-1].strip()


def build_prompt_scaffold_solution(prompt: str, completion: str) -> str:
    gen = clean_model_completion(completion, prompt=prompt)
    if not gen:
        return ""
    if starts_like_code(gen):
        return gen.rstrip()

    scaffold = extract_last_fenced_code(prompt)
    if not scaffold:
        return gen.rstrip()

    scaffold = scaffold.rstrip()
    if scaffold.endswith(":"):
        return (scaffold + "\n" + indent_as_body(gen.lstrip())).rstrip()
    return (scaffold + "\n" + gen.lstrip()).rstrip()

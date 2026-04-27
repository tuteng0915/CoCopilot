from __future__ import annotations

import ast
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


def build_evalplus_solution(prompt_text: str, completion: str) -> str:
    """Build a full EvalPlus-compatible solution string from a prompt + completion."""
    gen = clean_model_completion(completion, prompt_text).lstrip()
    prompt = prompt_text.rstrip()

    def extract_prompt_imports(p: str) -> str:
        imports = []
        for line in p.splitlines():
            stripped = line.strip()
            if stripped.startswith("from ") or stripped.startswith("import "):
                imports.append(line.rstrip())
                continue
            if stripped.startswith("def ") or stripped.startswith("class "):
                break
        return "\n".join(imports).strip()

    def infer_target_func_name(p: str) -> str | None:
        match = re.search(r"(?m)^\s*def\s+([A-Za-z_]\w*)\s*\(", p)
        return match.group(1) if match else None

    def extract_single_function(src: str, func_name: str) -> str | None:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            tree = None
        if tree is not None:
            lines = src.splitlines()
            for node in tree.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if node.name != func_name or not getattr(node, "end_lineno", None):
                    continue
                start_lineno = node.lineno
                if node.decorator_list:
                    start_lineno = min(d.lineno for d in node.decorator_list)
                return "\n".join(lines[start_lineno - 1 : node.end_lineno]).strip()

        pattern = (
            rf"(?ms)^(?P<decor>(?:@\w[^\n]*\n)*)"
            rf"(?P<def>(?:async\s+def|def)\s+{re.escape(func_name)}\s*\(.*?)(?=^(?:async\s+def|def|class)\s+|\Z)"
        )
        match = re.search(pattern, src)
        if not match:
            return None
        return (match.group("decor") + match.group("def")).strip()

    if re.search(r"(?m)^(def|class|import|from)\s+", gen):
        target_name = infer_target_func_name(prompt)
        if target_name:
            extracted = extract_single_function(gen, target_name)
            body = extracted if extracted else gen.rstrip()
        else:
            body = gen.rstrip()

        imports = extract_prompt_imports(prompt)
        if imports and imports not in body:
            body = (imports + "\n\n" + body).rstrip()
        return body.rstrip()

    return (prompt + "\n" + indent_as_body(gen.lstrip())).rstrip()

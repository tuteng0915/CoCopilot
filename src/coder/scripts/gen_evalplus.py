import argparse
import gzip
import json
import os
import random
import time
from typing import Dict, Tuple, List

from tqdm import tqdm
from evalplus.data import get_human_eval_plus, get_mbpp_plus

from coder.utils.schema import ModelRequest, SampleRecord
from coder.utils.code_cleaning import clean_model_completion as _clean_model_completion
from coder.models import (
    CoderModel,
    DreamCoder,
    DeepSeekCoder,
    QwenCoder,
    Qwen35Coder,
    LLaDACoder,
    StarCoder2Coder,
    MistralCoder,
    Llama31Coder,
    DiffuLLaMACoder,
    SeedDiffCoder,
    SeedCoder,
    ApiCoder,
)
import re


def build_model(name: str, device: str, model_id: str | None) -> CoderModel:
    if name == "dream":
        return DreamCoder(
            model_id=model_id or "Dream-org/Dream-Coder-v0-Instruct-7B",
            device=device,
        )
    if name == "deepseek":
        return DeepSeekCoder(
            model_id=model_id or "deepseek-ai/deepseek-coder-6.7b-instruct",
            device=device,
        )
    if name == "qwen":
        return QwenCoder(
            model_id=model_id or "Qwen/Qwen2.5-Coder-7B-Instruct",
            device=device,
        )
    if name == "llada":
        return LLaDACoder(
            model_id=model_id or "GSAI-ML/LLaDA-8B-Instruct",
            device=device,
        )
    if name == "qwen35":
        return Qwen35Coder(
            model_id=model_id or "Qwen/Qwen3.5-4B",
            device=device,
        )
    if name == "starcoder2":
        return StarCoder2Coder(
            model_id=model_id or "bigcode/starcoder2-7b",
            device=device,
        )
    if name == "mistral":
        return MistralCoder(
            model_id=model_id or "mistralai/Mistral-7B-Instruct-v0.3",
            device=device,
        )
    if name == "llama31":
        return Llama31Coder(
            model_id=model_id or "meta-llama/Llama-3.1-8B-Instruct",
            device=device,
        )
    if name == "diffullama":
        return DiffuLLaMACoder(
            model_id=model_id,
            device=device,
        )
    if name == "seed-diffcoder":
        return SeedDiffCoder(
            model_id=model_id,
            device=device,
        )
    if name == "seed-coder":
        return SeedCoder(
            model_id=model_id,
            device=device,
        )
    if name == "api":
        return ApiCoder(
            model_id=model_id,
            device="api",
        )
    raise ValueError(f"Unknown model: {name}")


def select_tasks(
    problems: Dict[str, Dict],
    limit: int,
    task_ids: List[str] | None,
    shuffle: bool,
    seed: int,
) -> List[Tuple[str, Dict]]:
    items = list(problems.items())
    items.sort(key=lambda x: x[0])  # deterministic order

    if task_ids:
        wanted = set(task_ids)
        items = [(tid, p) for tid, p in items if tid in wanted]

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(items)

    if limit > 0:
        items = items[:limit]

    return items


def write_override_dataset_gz(out_path: str, selected: List[Tuple[str, Dict]]):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for _, problem in selected:
            f.write(json.dumps(problem, ensure_ascii=False) + "\n")

def clean_model_completion(text: str, prompt: str | None = None) -> str:
    return _clean_model_completion(text, prompt=prompt)


def build_evalplus_solution(prob: dict, gen: str) -> str:
    """
    EvalPlus accepts a full `solution` program.
    If gen already looks like complete code, use it directly.
    Otherwise, treat gen as completion and prepend prompt.
    """
    g = (gen or "").lstrip()
    prompt = prob["prompt"].rstrip()

    def extract_prompt_imports(p: str) -> str:
        imports = []
        for line in p.splitlines():
            s = line.strip()
            if s.startswith("from ") or s.startswith("import "):
                imports.append(line.rstrip())
                continue
            # Stop once we hit code beyond the import prelude.
            if s.startswith("def ") or s.startswith("class "):
                break
        return "\n".join(imports).strip()

    def infer_target_func_name(p: str) -> str | None:
        m = re.search(r"(?m)^\s*def\s+([A-Za-z_]\w*)\s*\(", p)
        return m.group(1) if m else None

    def extract_single_function(src: str, func_name: str) -> str | None:
        # Capture optional decorators right above the target def.
        # Then capture the def block until the next top-level def/class or EOF.
        pattern = (
            rf"(?ms)^(?P<decor>(?:@\w[^\n]*\n)*)"
            rf"(?P<def>def\s+{re.escape(func_name)}\s*\(.*?)(?=^\s*(?:def|class)\s+|\Z)"
        )
        m = re.search(pattern, src)
        if not m:
            return None
        return (m.group("decor") + m.group("def")).strip()

    def indent_as_body(completion: str, spaces: int = 4) -> str:
        # Body completions should be indented under a function definition.
        # Some models output a mix (first line not indented, later lines already indented).
        # We only indent lines that start at column 0, preserving existing indentation.
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

    # If the model already returned full code, don't prepend the prompt again.
    if re.search(r"(?m)^(def|class|import|from)\s+", g):
        # Heuristics for models that dump multiple defs / miss prompt imports.
        tgt = infer_target_func_name(prompt)
        if tgt:
            extracted = extract_single_function(g, tgt)
            if extracted:
                g2 = extracted
            else:
                g2 = g.rstrip()
        else:
            g2 = g.rstrip()

        imports = extract_prompt_imports(prompt)
        if imports and imports not in g2:
            # If the prompt contained imports, keep them so typing names resolve.
            g2 = (imports + "\n\n" + g2).rstrip()
        return g2.rstrip()

    # Otherwise assume completion (e.g., body-only for HumanEval)
    body = indent_as_body(gen.lstrip())
    return (prompt + "\n" + body).rstrip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        choices=[
            "dream",
            "deepseek",
            "qwen",
            "qwen35",
            "llada",
            "starcoder2",
            "mistral",
            "llama31",
            "diffullama",
            "seed-diffcoder",
            "seed-coder",
            "api",
        ],
        required=True,
    )
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)

    ap.add_argument("--out", required=True)
    ap.add_argument("--override_out", default=None, help="If set, also write override dataset (.jsonl.gz) for subset eval.")

    ap.add_argument("--limit", type=int, default=0, help="Only generate first N tasks after filtering/shuffle. 0 = all.")
    ap.add_argument("--task_ids", type=str, default=None, help="Comma-separated task_ids to run (overrides full set).")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle tasks before taking --limit.")

    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of AR samples per task (for pass@n). Default 1 keeps old behavior.",
    )

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--model_id", type=str, default=None, help="Override HuggingFace model id.")

    args = ap.parse_args()
    if args.num_samples < 1:
        ap.error("--num_samples must be >= 1")

    problems = get_human_eval_plus() if args.dataset == "humaneval" else get_mbpp_plus()
    task_ids = [x.strip() for x in args.task_ids.split(",")] if args.task_ids else None

    selected = select_tasks(
        problems=problems,
        limit=args.limit,
        task_ids=task_ids,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    # If user wants subset, strongly recommend writing override dataset so evalplus.evaluate won't assert.
    if (args.limit > 0 or task_ids) and args.override_out is None:
        base = os.path.splitext(os.path.basename(args.out))[0]
        args.override_out = f"outputs/{args.dataset}_override_{base}.jsonl.gz"

    if args.override_out:
        write_override_dataset_gz(args.override_out, selected)
        print(f"[override] wrote {args.override_out}")
        if args.dataset == "humaneval":
            print(f'  export HUMANEVAL_OVERRIDE_PATH="{os.path.abspath(args.override_out)}"')
        else:
            print(f'  export MBPP_OVERRIDE_PATH="{os.path.abspath(args.override_out)}"')

    model = build_model(args.model, device=args.device, model_id=args.model_id)

    t_total0 = time.perf_counter()
    timing_generate_s: list[float] = []
    timing_post_s: list[float] = []
    n_records_written = 0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for task_id, prob in tqdm(selected, desc=f"gen:{args.model}:{args.dataset}"):
            prompt = (
                "Complete the following Python function. "
                "Only output valid Python code (no explanation).\n\n"
                f"{prob['prompt']}\n"
            )
            base_req = ModelRequest(
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
            )

            for sample_idx in range(args.num_samples):
                # Keep old behavior when num_samples=1.
                req = base_req
                if args.num_samples > 1 and req.seed is not None:
                    req = ModelRequest(
                        prompt=base_req.prompt,
                        max_new_tokens=base_req.max_new_tokens,
                        temperature=base_req.temperature,
                        top_p=base_req.top_p,
                        seed=int(args.seed) + sample_idx,
                    )

                t0 = time.perf_counter()
                raw_gen = model.generate(req)
                t1 = time.perf_counter()
                gen = clean_model_completion(raw_gen, prompt=req.prompt)
                solution = build_evalplus_solution(prob, gen)
                t2 = time.perf_counter()

                timing_generate_s.append(t1 - t0)
                timing_post_s.append(t2 - t1)

                # EvalPlus pass@n: write multiple rows with the SAME task_id.
                rec = {
                    "task_id": task_id,
                    "sample_id": sample_idx,
                    "prompt": req.prompt,
                    "raw_completion": gen,
                    "solution": solution,
                    "model": model.name,
                    "gen": {
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "seed": req.seed,
                        "num_samples": args.num_samples,
                        "sample_id": sample_idx,
                        "timing": {
                            "generate_s": t1 - t0,
                            "postprocess_s": t2 - t1,
                            "total_s": t2 - t0,
                        },
                    },
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_records_written += 1

    t_total1 = time.perf_counter()

    summary = {
        "script": "gen_evalplus",
        "out": os.path.abspath(args.out),
        "model": model.name,
        "dataset": args.dataset,
        "num_samples": args.num_samples,
        "n_records_written": n_records_written,
        "timing": {
            "total_s": t_total1 - t_total0,
            "generate_s_total": float(sum(timing_generate_s)),
            "postprocess_s_total": float(sum(timing_post_s)),
            "generate_s_avg": (float(sum(timing_generate_s)) / len(timing_generate_s)) if timing_generate_s else None,
            "postprocess_s_avg": (float(sum(timing_post_s)) / len(timing_post_s)) if timing_post_s else None,
        },
    }
    timing_path = args.out + ".timing_summary.json"
    with open(timing_path, "w", encoding="utf-8") as tf:
        json.dump(summary, tf, ensure_ascii=False, indent=2)

    print(f"[samples] wrote {args.out}")
    print(f"[timing] wrote {timing_path}")


if __name__ == "__main__":
    main()

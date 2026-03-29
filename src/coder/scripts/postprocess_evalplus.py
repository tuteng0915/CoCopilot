# scripts/postprocess_evalplus.py
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str]):
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def guess_sanitized_path(samples: str) -> str:
    p = Path(samples)
    if p.suffix == ".jsonl":
        return str(p.with_name(p.stem + "-sanitized" + p.suffix))
    return samples + "-sanitized"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--skip_syncheck", action="store_true")
    args = ap.parse_args()

    # syncheck usually needs dataset
    if not args.skip_syncheck:
        run(["evalplus.syncheck", "--samples", args.samples, "--dataset", args.dataset])

    # sanitize in your evalplus version does NOT take --dataset
    run(["evalplus.sanitize", "--samples", args.samples])

    sanitized = guess_sanitized_path(args.samples)
    print("[info] sanitized samples:", sanitized)

    if not args.skip_syncheck:
        run(["evalplus.syncheck", "--samples", sanitized, "--dataset", args.dataset])


if __name__ == "__main__":
    main()

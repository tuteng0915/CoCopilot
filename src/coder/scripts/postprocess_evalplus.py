# scripts/postprocess_evalplus.py
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Resolve evalplus CLI tools relative to the current Python interpreter so they
# are found even when the conda env's bin directory isn't in PATH.
_EVALPLUS_BIN = Path(sys.executable).parent


def _evalplus_cmd(name: str) -> str:
    """Return the absolute path to an evalplus CLI tool."""
    candidate = _EVALPLUS_BIN / name
    if candidate.exists():
        return str(candidate)
    return name  # fall back to PATH lookup


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
        run([_evalplus_cmd("evalplus.syncheck"), "--samples", args.samples, "--dataset", args.dataset])

    # sanitize in your evalplus version does NOT take --dataset
    run([_evalplus_cmd("evalplus.sanitize"), "--samples", args.samples])

    sanitized = guess_sanitized_path(args.samples)
    print("[info] sanitized samples:", sanitized)

    if not args.skip_syncheck:
        run([_evalplus_cmd("evalplus.syncheck"), "--samples", sanitized, "--dataset", args.dataset])


if __name__ == "__main__":
    main()

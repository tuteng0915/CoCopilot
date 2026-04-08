#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def count_unique_ids(path: Path, key: str) -> int:
    if not path.exists():
        return 0
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            value = obj.get(key)
            if value:
                seen.add(str(value))
    return len(seen)


def expected_shard_count(total: int, num_shards: int, shard_idx: int) -> int:
    if shard_idx >= total:
        return 0
    return 1 + (total - 1 - shard_idx) // num_shards


def query_gpu_state() -> dict[int, dict[str, int]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True)
    state: dict[int, dict[str, int]] = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        idx_s, mem_s, util_s = [part.strip() for part in line.split(",")]
        state[int(idx_s)] = {"memory_used_mb": int(mem_s), "utilization_gpu": int(util_s)}
    return state


def summary_pass_pct(path: Path) -> float | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if "accuracy" in payload and payload["accuracy"] is not None:
        return float(payload["accuracy"]) * 100.0
    pass_at_k = payload.get("pass_at_k") or {}
    if "pass@1" in pass_at_k and pass_at_k["pass@1"] is not None:
        return float(pass_at_k["pass@1"]) * 100.0
    return None


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    total_tasks: int
    id_field: str
    shard_dir: Path
    merged_out: Path
    summary_out: Path
    eval_kind: str
    judgments_out: Path | None
    input_path: Path | None

    def shard_out(self, shard_idx: int) -> Path:
        return self.shard_dir / f"{self.name}_shard{shard_idx:02d}.jsonl"

    def shard_log(self, shard_idx: int, log_dir: Path) -> Path:
        return log_dir / f"{self.name}_shard{shard_idx:02d}.log"

    def eval_log(self, log_dir: Path) -> Path:
        return log_dir / f"{self.name}_eval.log"


@dataclass
class RunningJob:
    spec_name: str
    shard_idx: int
    gpu: int
    proc: subprocess.Popen[Any]
    log_path: Path
    log_file: Any


def build_experiments(repo_root: Path) -> list[ExperimentSpec]:
    outputs = repo_root / "outputs" / "base_tuteng"
    shard_root = outputs / "extended_shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    live_input = outputs / "deepseek_livecodebench_pass1_clean.jsonl"
    bcb_input = outputs / "deepseek_bigcodebench_instruct_full_pass1_clean.jsonl"
    total_live = count_jsonl(live_input)
    total_bcb = count_jsonl(bcb_input)
    if total_live <= 0 or total_bcb <= 0:
        raise RuntimeError("Expected cleaned DeepSeek inputs for LiveCodeBench and BigCodeBench.")

    return [
        ExperimentSpec(
            name="dream_livecodebench_pass1",
            total_tasks=total_live,
            id_field="task_id",
            shard_dir=shard_root / "dream_livecodebench_pass1",
            merged_out=outputs / "dream_livecodebench_pass1_sharded.jsonl",
            summary_out=outputs / "dream_livecodebench_pass1_sharded_summary.json",
            eval_kind="livecodebench",
            judgments_out=outputs / "dream_livecodebench_pass1_sharded_judgments.jsonl",
            input_path=None,
        ),
        ExperimentSpec(
            name="dream_bigcodebench_instruct_full_pass1",
            total_tasks=total_bcb,
            id_field="task_id",
            shard_dir=shard_root / "dream_bigcodebench_instruct_full_pass1",
            merged_out=outputs / "dream_bigcodebench_instruct_full_pass1_sharded.jsonl",
            summary_out=outputs / "dream_bigcodebench_instruct_full_pass1_sharded_summary.json",
            eval_kind="bigcodebench",
            judgments_out=None,
            input_path=None,
        ),
        ExperimentSpec(
            name="collab_t0.9_livecodebench",
            total_tasks=total_live,
            id_field="task_id",
            shard_dir=shard_root / "collab_t0.9_livecodebench",
            merged_out=outputs / "collab_t0.9_livecodebench_sharded.jsonl",
            summary_out=outputs / "collab_t0.9_livecodebench_sharded_summary.json",
            eval_kind="livecodebench",
            judgments_out=outputs / "collab_t0.9_livecodebench_sharded_judgments.jsonl",
            input_path=live_input,
        ),
        ExperimentSpec(
            name="collab_t0.9_bigcodebench_instruct_full",
            total_tasks=total_bcb,
            id_field="task_id",
            shard_dir=shard_root / "collab_t0.9_bigcodebench_instruct_full",
            merged_out=outputs / "collab_t0.9_bigcodebench_instruct_full_sharded.jsonl",
            summary_out=outputs / "collab_t0.9_bigcodebench_instruct_full_sharded_summary.json",
            eval_kind="bigcodebench",
            judgments_out=None,
            input_path=bcb_input,
        ),
    ]


def build_shard_cmd(spec: ExperimentSpec, shard_idx: int, num_shards: int) -> list[str]:
    if spec.name.startswith("dream_livecodebench"):
        return [
            sys.executable,
            "-m",
            "coder.scripts.gen_livebench",
            "--benchmark",
            "livecodebench",
            "--model",
            "dream",
            "--num_shards",
            str(num_shards),
            "--shard_idx",
            str(shard_idx),
            "--out",
            str(spec.shard_out(shard_idx)),
            "--resume",
        ]
    if spec.name.startswith("dream_bigcodebench"):
        return [
            sys.executable,
            "-m",
            "coder.scripts.gen_bigcodebench",
            "--model",
            "dream",
            "--split",
            "instruct",
            "--subset",
            "full",
            "--num_shards",
            str(num_shards),
            "--shard_idx",
            str(shard_idx),
            "--out",
            str(spec.shard_out(shard_idx)),
            "--resume",
        ]
    if spec.name.startswith("collab_t0.9_livecodebench"):
        return [
            sys.executable,
            "-m",
            "coder.scripts.gen_remask",
            "--input",
            str(spec.input_path),
            "--out",
            str(spec.shard_out(shard_idx)),
            "--refiner",
            "dream",
            "--confidence_threshold",
            "0.9",
            "--temperature",
            "0.1",
            "--top_p",
            "0.95",
            "--seed",
            "3407",
            "--num_shards",
            str(num_shards),
            "--shard_idx",
            str(shard_idx),
            "--resume",
        ]
    if spec.name.startswith("collab_t0.9_bigcodebench"):
        return [
            sys.executable,
            "-m",
            "coder.scripts.gen_remask",
            "--input",
            str(spec.input_path),
            "--out",
            str(spec.shard_out(shard_idx)),
            "--refiner",
            "dream",
            "--confidence_threshold",
            "0.9",
            "--temperature",
            "0.1",
            "--top_p",
            "0.95",
            "--seed",
            "3407",
            "--num_shards",
            str(num_shards),
            "--shard_idx",
            str(shard_idx),
            "--resume",
        ]
    raise ValueError(f"Unknown experiment: {spec.name}")


def build_eval_cmd(spec: ExperimentSpec) -> list[str]:
    if spec.eval_kind == "livecodebench":
        return [
            sys.executable,
            "-m",
            "coder.scripts.eval_livebench",
            "--benchmark",
            "livecodebench",
            "--samples",
            str(spec.merged_out),
            "--out_judgments",
            str(spec.judgments_out),
            "--out_summary",
            str(spec.summary_out),
        ]
    if spec.eval_kind == "bigcodebench":
        return [
            sys.executable,
            "-m",
            "coder.scripts.eval_bigcodebench",
            "--samples",
            str(spec.merged_out),
            "--split",
            "instruct",
            "--subset",
            "full",
            "--execution",
            "local",
            "--out_summary",
            str(spec.summary_out),
        ]
    raise ValueError(f"Unknown eval_kind: {spec.eval_kind}")


def shard_complete(spec: ExperimentSpec, shard_idx: int, num_shards: int) -> bool:
    expected = expected_shard_count(spec.total_tasks, num_shards, shard_idx)
    actual = count_unique_ids(spec.shard_out(shard_idx), spec.id_field)
    return actual == expected


def merge_shards(spec: ExperimentSpec, num_shards: int) -> None:
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for shard_idx in range(num_shards):
        for rec in read_jsonl(spec.shard_out(shard_idx)):
            key = str(rec[spec.id_field])
            if key not in merged:
                order.append(key)
            merged[key] = rec
    if len(merged) != spec.total_tasks:
        raise RuntimeError(
            f"{spec.name}: merged unique rows={len(merged)} expected={spec.total_tasks}"
        )
    spec.merged_out.parent.mkdir(parents=True, exist_ok=True)
    with spec.merged_out.open("w", encoding="utf-8") as f:
        for key in order:
            f.write(json.dumps(merged[key], ensure_ascii=False) + "\n")
    print(f"[merge] {spec.name}: wrote {spec.merged_out} ({len(merged)} rows)", flush=True)


def merged_is_fresh(spec: ExperimentSpec, num_shards: int) -> bool:
    if not spec.merged_out.exists():
        return False
    merged_mtime = spec.merged_out.stat().st_mtime
    for shard_idx in range(num_shards):
        shard_path = spec.shard_out(shard_idx)
        if not shard_path.exists() or merged_mtime < shard_path.stat().st_mtime:
            return False
    return count_unique_ids(spec.merged_out, spec.id_field) == spec.total_tasks


def summary_is_fresh(spec: ExperimentSpec) -> bool:
    if not spec.summary_out.exists() or not spec.merged_out.exists():
        return False
    return spec.summary_out.stat().st_mtime >= spec.merged_out.stat().st_mtime


def write_status_snapshot(specs: list[ExperimentSpec], num_shards: int, out_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        shard_rows = []
        for shard_idx in range(num_shards):
            shard_path = spec.shard_out(shard_idx)
            shard_rows.append(
                {
                    "shard_idx": shard_idx,
                    "expected_rows": expected_shard_count(spec.total_tasks, num_shards, shard_idx),
                    "actual_unique_rows": count_unique_ids(shard_path, spec.id_field),
                    "path": str(shard_path),
                }
            )
        rows.append(
            {
                "name": spec.name,
                "merged_out": str(spec.merged_out),
                "summary_out": str(spec.summary_out),
                "pass_at_1_pct": summary_pass_pct(spec.summary_out),
                "shards": shard_rows,
            }
        )
    payload = {
        "updated_at_epoch_s": time.time(),
        "num_shards": num_shards,
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_eval(spec: ExperimentSpec, log_dir: Path) -> None:
    cmd = build_eval_cmd(spec)
    log_path = spec.eval_log(log_dir)
    print(f"[eval] {spec.name}: {' '.join(cmd)}", flush=True)
    with log_path.open("a", encoding="utf-8") as log_f:
        log_f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log_f.write("[cmd] " + " ".join(cmd) + "\n")
        log_f.flush()
        subprocess.run(cmd, check=True, stdout=log_f, stderr=subprocess.STDOUT)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Queue and run the remaining tab:extended Dream/Collaborative benchmarks."
    )
    ap.add_argument("--gpus", default="2,0,3,4,5,6,7,1", help="Comma-separated GPU indices to use when idle.")
    ap.add_argument("--num_shards", type=int, default=4)
    ap.add_argument("--poll_interval_s", type=int, default=60)
    ap.add_argument("--mem_threshold_mb", type=int, default=8000)
    ap.add_argument("--util_threshold", type=int, default=10)
    ap.add_argument(
        "--status_out",
        default="outputs/base_tuteng/extended_table_t0.9_status.json",
    )
    ap.add_argument(
        "--log_dir",
        default="outputs/base_tuteng/extended_scheduler_logs",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    os.chdir(repo_root)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    gpus = [int(part.strip()) for part in args.gpus.split(",") if part.strip()]
    specs = build_experiments(repo_root)
    for spec in specs:
        spec.shard_dir.mkdir(parents=True, exist_ok=True)

    running: dict[int, RunningJob] = {}
    completed_evals: set[str] = set()

    while True:
        for gpu, job in list(running.items()):
            ret = job.proc.poll()
            if ret is None:
                continue
            job.log_file.write(f"\n[exit_code] {ret}\n")
            job.log_file.flush()
            job.log_file.close()
            del running[gpu]
            if ret != 0:
                raise RuntimeError(
                    f"{job.spec_name} shard {job.shard_idx} failed on GPU {gpu}. See {job.log_path}"
                )
            print(
                f"[done] {job.spec_name} shard {job.shard_idx} finished on GPU {gpu}",
                flush=True,
            )

        for spec in specs:
            all_shards_done = all(
                shard_complete(spec, shard_idx, args.num_shards)
                for shard_idx in range(args.num_shards)
            )
            if not all_shards_done:
                continue
            if not merged_is_fresh(spec, args.num_shards):
                merge_shards(spec, args.num_shards)
            if spec.name not in completed_evals and not summary_is_fresh(spec):
                run_eval(spec, log_dir=log_dir)
            if summary_is_fresh(spec):
                completed_evals.add(spec.name)

        write_status_snapshot(
            specs=specs,
            num_shards=args.num_shards,
            out_path=Path(args.status_out),
        )

        if len(completed_evals) == len(specs):
            print("[done] all extended benchmark jobs finished", flush=True)
            break

        gpu_state = query_gpu_state()
        running_specs = {(job.spec_name, job.shard_idx) for job in running.values()}

        for gpu in gpus:
            if gpu in running:
                continue
            state = gpu_state.get(gpu)
            if state is None:
                continue
            if state["memory_used_mb"] > args.mem_threshold_mb:
                continue
            if state["utilization_gpu"] > args.util_threshold:
                continue

            next_job: tuple[ExperimentSpec, int] | None = None
            for shard_idx in range(args.num_shards):
                for spec in specs:
                    if shard_complete(spec, shard_idx, args.num_shards):
                        continue
                    if (spec.name, shard_idx) in running_specs:
                        continue
                    next_job = (spec, shard_idx)
                    break
                if next_job is not None:
                    break
            if next_job is None:
                break

            spec, shard_idx = next_job
            cmd = build_shard_cmd(spec, shard_idx=shard_idx, num_shards=args.num_shards)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONPATH"] = f"{repo_root / 'src'}:{env.get('PYTHONPATH', '')}".rstrip(":")

            log_path = spec.shard_log(shard_idx=shard_idx, log_dir=log_dir)
            log_f = log_path.open("a", encoding="utf-8")
            log_f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            log_f.write("[cmd] " + " ".join(cmd) + "\n")
            log_f.flush()

            proc = subprocess.Popen(
                cmd,
                cwd=repo_root,
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
            running[gpu] = RunningJob(
                spec_name=spec.name,
                shard_idx=shard_idx,
                gpu=gpu,
                proc=proc,
                log_path=log_path,
                log_file=log_f,
            )
            running_specs.add((spec.name, shard_idx))
            print(
                f"[launch] {spec.name} shard {shard_idx} on GPU {gpu} -> {log_path}",
                flush=True,
            )

        time.sleep(args.poll_interval_s)


if __name__ == "__main__":
    main()

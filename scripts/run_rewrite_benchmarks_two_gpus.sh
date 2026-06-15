#!/usr/bin/env bash
set -euo pipefail

source /home/tteng/miniconda3/etc/profile.d/conda.sh
conda activate code
cd /model/tteng/CoCoder
export PYTHONPATH="/model/tteng/CoCoder/src:${PYTHONPATH:-}"

RUN_DIR="outputs/rewrite_run"
mkdir -p "$RUN_DIR" outputs/rewrite

repair_jsonl() {
  python3 - "$@" <<'PY'
import json
import pathlib
import sys

for raw in sys.argv[1:]:
    path = pathlib.Path(raw)
    if not path.exists():
        continue
    original = path.read_text(encoding="utf-8")
    good = []
    bad = 0
    for line in original.splitlines():
        if not line.strip():
            continue
        try:
            json.loads(line)
        except Exception:
            bad += 1
            continue
        good.append(line)
    if bad:
        backup = path.with_suffix(path.suffix + ".corrupt_tail_backup")
        backup.write_text(original, encoding="utf-8")
        path.write_text("\n".join(good) + ("\n" if good else ""), encoding="utf-8")
        print(f"[repair] {path}: removed {bad} invalid line(s), backup={backup}")
PY
}

count_jsonl() {
  python3 - "$@" <<'PY'
import pathlib
import sys

for raw in sys.argv[1:]:
    path = pathlib.Path(raw)
    n = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()) if path.exists() else 0
    print(f"{path}: {n}")
PY
}

repair_jsonl \
  outputs/rewrite/asset_llama31.jsonl \
  outputs/rewrite/asset_dream_general.jsonl \
  outputs/rewrite/asset_llama31_dream_general_t0.9.jsonl \
  outputs/rewrite/coedit_llama31.jsonl \
  outputs/rewrite/coedit_dream_general.jsonl \
  outputs/rewrite/coedit_llama31_dream_general_t0.9.jsonl

lane_asset() {
  export CUDA_VISIBLE_DEVICES=0
  python -m coder.scripts.gen_rewrite --model llama31 --dataset asset --out outputs/rewrite/asset_llama31.jsonl --max_new_tokens 128 --temperature 0.0 --seed 3407 --resume
  python -m coder.scripts.gen_rewrite --model dream_general --dataset asset --out outputs/rewrite/asset_dream_general.jsonl --max_new_tokens 128 --temperature 0.1 --seed 3407 --resume
  python -m coder.scripts.gen_remask --input outputs/rewrite/asset_llama31.jsonl --out outputs/rewrite/asset_llama31_dream_general_t0.9.jsonl --refiner dream_general --model_id Dream-org/Dream-v0-Instruct-7B --confidence_threshold 0.9 --temperature 0.1 --top_p 0.95 --seed 3407 --resume
  python -m coder.scripts.eval_rewrite --input outputs/rewrite/asset_llama31.jsonl --out outputs/rewrite/asset_llama31_eval.json
  python -m coder.scripts.eval_rewrite --input outputs/rewrite/asset_dream_general.jsonl --out outputs/rewrite/asset_dream_general_eval.json
  python -m coder.scripts.eval_rewrite --input outputs/rewrite/asset_llama31_dream_general_t0.9.jsonl --out outputs/rewrite/asset_cocoder_eval.json
}

lane_coedit() {
  export CUDA_VISIBLE_DEVICES=3
  python -m coder.scripts.gen_rewrite --model llama31 --dataset coedit --tasks_filter gec,paraphrase --out outputs/rewrite/coedit_llama31.jsonl --max_new_tokens 128 --temperature 0.0 --seed 3407 --resume
  python -m coder.scripts.gen_rewrite --model dream_general --dataset coedit --tasks_filter gec,paraphrase --out outputs/rewrite/coedit_dream_general.jsonl --max_new_tokens 128 --temperature 0.1 --seed 3407 --resume
  python -m coder.scripts.gen_remask --input outputs/rewrite/coedit_llama31.jsonl --out outputs/rewrite/coedit_llama31_dream_general_t0.9.jsonl --refiner dream_general --model_id Dream-org/Dream-v0-Instruct-7B --confidence_threshold 0.9 --temperature 0.1 --top_p 0.95 --seed 3407 --resume
  python -m coder.scripts.eval_rewrite --input outputs/rewrite/coedit_llama31.jsonl --out outputs/rewrite/coedit_llama31_eval.json --by_task
  python -m coder.scripts.eval_rewrite --input outputs/rewrite/coedit_dream_general.jsonl --out outputs/rewrite/coedit_dream_general_eval.json --by_task
  python -m coder.scripts.eval_rewrite --input outputs/rewrite/coedit_llama31_dream_general_t0.9.jsonl --out outputs/rewrite/coedit_cocoder_eval.json --by_task
}

( lane_asset ) >"$RUN_DIR/lane_asset_gpu0.log" 2>&1 &
PID_ASSET=$!
( lane_coedit ) >"$RUN_DIR/lane_coedit_gpu3.log" 2>&1 &
PID_COEDIT=$!

echo "[run] asset lane pid=$PID_ASSET log=$RUN_DIR/lane_asset_gpu0.log"
echo "[run] coedit lane pid=$PID_COEDIT log=$RUN_DIR/lane_coedit_gpu3.log"

wait "$PID_ASSET"
wait "$PID_COEDIT"

count_jsonl \
  outputs/rewrite/asset_llama31.jsonl \
  outputs/rewrite/asset_dream_general.jsonl \
  outputs/rewrite/asset_llama31_dream_general_t0.9.jsonl \
  outputs/rewrite/coedit_llama31.jsonl \
  outputs/rewrite/coedit_dream_general.jsonl \
  outputs/rewrite/coedit_llama31_dream_general_t0.9.jsonl

python3 - <<'PY'
import json
import pathlib

print("=== ASSET (SARI / BLEU-4) ===")
for label, path in [
    ("AR (Llama31)", "outputs/rewrite/asset_llama31_eval.json"),
    ("DreamGeneral", "outputs/rewrite/asset_dream_general_eval.json"),
    ("CoCoder t=0.9", "outputs/rewrite/asset_cocoder_eval.json"),
]:
    try:
        d = json.loads(pathlib.Path(path).read_text())
        print(f"  {label:<14}: SARI={d['sari']:.2f}  BLEU={d['bleu4']:.2f}")
    except FileNotFoundError:
        print(f"  {label:<14}: NOT FOUND")

print()
print("=== CoEdIT (SARI / BLEU-4) ===")
for label, path in [
    ("AR (Llama31)", "outputs/rewrite/coedit_llama31_eval.json"),
    ("DreamGeneral", "outputs/rewrite/coedit_dream_general_eval.json"),
    ("CoCoder t=0.9", "outputs/rewrite/coedit_cocoder_eval.json"),
]:
    try:
        d = json.loads(pathlib.Path(path).read_text())
    except FileNotFoundError:
        print(f"  {label:<14}: NOT FOUND")
        continue
    for task in ("gec", "paraphrase"):
        row = d.get("by_task", {}).get(task, {})
        print(f"  {label:<14} {task:<10}: SARI={row.get('sari', 0.0):.2f}  BLEU={row.get('bleu4', 0.0):.2f}")
PY

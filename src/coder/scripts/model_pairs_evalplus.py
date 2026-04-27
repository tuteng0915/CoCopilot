from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evalplus.data import get_human_eval_plus, get_mbpp_plus

from coder.scripts.eval_evalplus import _candidate_evalplus_result_paths, _write_summary
from coder.scripts.postprocess_evalplus import guess_sanitized_path


@dataclass(frozen=True)
class PairConfig:
    slug: str
    dataset: str
    tau_conf: float
    ar_label: str
    refiner_label: str
    ar_input: str
    ar_summary: str
    collab_output: str
    collab_summary: str
    refiner: str
    refiner_model_id: str
    extra_collab_inputs: tuple[str, ...] = ()


PAIR_CONFIGS: tuple[PairConfig, ...] = (
    # ── HumanEval ──────────────────────────────────────────────────────────
    PairConfig(
        slug="deepseek_dream_humaneval_t0.9",
        dataset="humaneval",
        tau_conf=0.9,
        ar_label="DeepSeek-Coder 6.7B",
        refiner_label="Dream-Coder 7B",
        ar_input="outputs/base_tuteng/deepseek_humaneval.jsonl",
        ar_summary="outputs/base_tuteng/deepseek_humaneval_summary.json",
        collab_output="outputs/remask_kodai/remask_humaneval_t0.9.jsonl",
        collab_summary="outputs/remask_kodai/remask_humaneval_t0.9_summary.json",
        refiner="dream",
        refiner_model_id="Dream-org/Dream-Coder-v0-Instruct-7B",
    ),
    PairConfig(
        slug="qwen_dream_humaneval_t0.9",
        dataset="humaneval",
        tau_conf=0.9,
        ar_label="Qwen2.5-Coder 7B",
        refiner_label="Dream-Coder 7B",
        ar_input="outputs/base_tuteng/qwen_humaneval.jsonl",
        ar_summary="outputs/base_tuteng/qwen_humaneval_summary.json",
        collab_output="outputs/base_tuteng/qwen_dream_remask_humaneval_t0.9.jsonl",
        collab_summary="outputs/base_tuteng/qwen_dream_remask_humaneval_t0.9_summary.json",
        refiner="dream",
        refiner_model_id="Dream-org/Dream-Coder-v0-Instruct-7B",
    ),
    PairConfig(
        slug="llama31_dream_humaneval_t0.9",
        dataset="humaneval",
        tau_conf=0.9,
        ar_label="Llama-3.1 8B",
        refiner_label="Dream-Coder 7B",
        ar_input="outputs/base_tuteng/llama31_humaneval.jsonl",
        ar_summary="outputs/base_tuteng/llama31_humaneval_summary.json",
        collab_output="outputs/base_tuteng/llama31_dream_remask_humaneval_t0.9.jsonl",
        collab_summary="outputs/base_tuteng/llama31_dream_remask_humaneval_t0.9_summary.json",
        refiner="dream",
        refiner_model_id="Dream-org/Dream-Coder-v0-Instruct-7B",
    ),
    PairConfig(
        slug="starcoder2_dream_humaneval_t0.9",
        dataset="humaneval",
        tau_conf=0.9,
        ar_label="StarCoder2 7B",
        refiner_label="Dream-Coder 7B",
        ar_input="outputs/base_tuteng/starcoder2_humaneval.jsonl",
        ar_summary="outputs/base_tuteng/starcoder2_humaneval_summary.json",
        collab_output="outputs/base_tuteng/starcoder2_dream_remask_humaneval_t0.9.jsonl",
        collab_summary="outputs/base_tuteng/starcoder2_dream_remask_humaneval_t0.9_summary.json",
        refiner="dream",
        refiner_model_id="Dream-org/Dream-Coder-v0-Instruct-7B",
        extra_collab_inputs=(
            "outputs/base_tuteng/starcoder2_shards/starcoder2_dream_remask_humaneval_t0.9_shard0.jsonl",
            "outputs/base_tuteng/starcoder2_shards/starcoder2_dream_remask_humaneval_t0.9_shard1.jsonl",
            "outputs/base_tuteng/starcoder2_shards/starcoder2_dream_remask_humaneval_t0.9_shard2.jsonl",
            "outputs/base_tuteng/starcoder2_shards/starcoder2_dream_remask_humaneval_t0.9_shard3.jsonl",
            "outputs/base_tuteng/starcoder2_reshard/starcoder2_dream_remask_humaneval_t0.9_reshard0.jsonl",
            "outputs/base_tuteng/starcoder2_reshard/starcoder2_dream_remask_humaneval_t0.9_reshard1.jsonl",
            "outputs/base_tuteng/starcoder2_reshard/starcoder2_dream_remask_humaneval_t0.9_reshard2.jsonl",
            "outputs/base_tuteng/starcoder2_reshard/starcoder2_dream_remask_humaneval_t0.9_reshard3.jsonl",
            "outputs/base_tuteng/starcoder2_reshard/starcoder2_dream_remask_humaneval_t0.9_reshard4.jsonl",
            "outputs/base_tuteng/starcoder2_reshard/starcoder2_dream_remask_humaneval_t0.9_reshard5.jsonl",
        ),
    ),
    PairConfig(
        slug="mistral_dream_humaneval_t0.9",
        dataset="humaneval",
        tau_conf=0.9,
        ar_label="Mistral 7B",
        refiner_label="Dream-Coder 7B",
        ar_input="outputs/base_tuteng/mistral_humaneval.jsonl",
        ar_summary="outputs/base_tuteng/mistral_humaneval_summary.json",
        collab_output="outputs/base_tuteng/mistral_dream_remask_humaneval_t0.9.jsonl",
        collab_summary="outputs/base_tuteng/mistral_dream_remask_humaneval_t0.9_summary.json",
        refiner="dream",
        refiner_model_id="Dream-org/Dream-Coder-v0-Instruct-7B",
    ),
    PairConfig(
        slug="codellama_dream_humaneval_t0.9",
        dataset="humaneval",
        tau_conf=0.9,
        ar_label="CodeLlama 7B",
        refiner_label="Dream-Coder 7B",
        ar_input="outputs/base_tuteng/codellama_humaneval.jsonl",
        ar_summary="outputs/base_tuteng/codellama_humaneval_summary.json",
        collab_output="outputs/base_tuteng/codellama_dream_remask_humaneval_t0.9.jsonl",
        collab_summary="outputs/base_tuteng/codellama_dream_remask_humaneval_t0.9_summary.json",
        refiner="dream",
        refiner_model_id="Dream-org/Dream-Coder-v0-Instruct-7B",
    ),
    PairConfig(
        slug="deepseek_llada_humaneval_t0.9",
        dataset="humaneval",
        tau_conf=0.9,
        ar_label="DeepSeek-Coder 6.7B",
        refiner_label="LLaDA 8B",
        ar_input="outputs/base_tuteng/deepseek_humaneval.jsonl",
        ar_summary="outputs/base_tuteng/deepseek_humaneval_summary.json",
        collab_output="outputs/base_tuteng/deepseek_llada_remask_humaneval_t0.9.jsonl",
        collab_summary="outputs/base_tuteng/deepseek_llada_remask_humaneval_t0.9_summary.json",
        refiner="llada",
        refiner_model_id="GSAI-ML/LLaDA-8B-Instruct",
    ),
    PairConfig(
        slug="qwen_llada_humaneval_t0.9",
        dataset="humaneval",
        tau_conf=0.9,
        ar_label="Qwen2.5-Coder 7B",
        refiner_label="LLaDA 8B",
        ar_input="outputs/base_tuteng/qwen_humaneval.jsonl",
        ar_summary="outputs/base_tuteng/qwen_humaneval_summary.json",
        collab_output="outputs/base_tuteng/qwen_llada_remask_humaneval_t0.9.jsonl",
        collab_summary="outputs/base_tuteng/qwen_llada_remask_humaneval_t0.9_summary.json",
        refiner="llada",
        refiner_model_id="GSAI-ML/LLaDA-8B-Instruct",
    ),
    PairConfig(
        slug="llama31_llada_humaneval_t0.9",
        dataset="humaneval",
        tau_conf=0.9,
        ar_label="Llama-3.1 8B",
        refiner_label="LLaDA 8B",
        ar_input="outputs/base_tuteng/llama31_humaneval.jsonl",
        ar_summary="outputs/base_tuteng/llama31_humaneval_summary.json",
        collab_output="outputs/base_tuteng/llama31_llada_remask_humaneval_t0.9.jsonl",
        collab_summary="outputs/base_tuteng/llama31_llada_remask_humaneval_t0.9_summary.json",
        refiner="llada",
        refiner_model_id="GSAI-ML/LLaDA-8B-Instruct",
    ),
    PairConfig(
        slug="starcoder2_llada_humaneval_t0.9",
        dataset="humaneval",
        tau_conf=0.9,
        ar_label="StarCoder2 7B",
        refiner_label="LLaDA 8B",
        ar_input="outputs/base_tuteng/starcoder2_humaneval.jsonl",
        ar_summary="outputs/base_tuteng/starcoder2_humaneval_summary.json",
        collab_output="outputs/base_tuteng/starcoder2_llada_remask_humaneval_t0.9.jsonl",
        collab_summary="outputs/base_tuteng/starcoder2_llada_remask_humaneval_t0.9_summary.json",
        refiner="llada",
        refiner_model_id="GSAI-ML/LLaDA-8B-Instruct",
    ),
    PairConfig(
        slug="mistral_llada_humaneval_t0.9",
        dataset="humaneval",
        tau_conf=0.9,
        ar_label="Mistral 7B",
        refiner_label="LLaDA 8B",
        ar_input="outputs/base_tuteng/mistral_humaneval.jsonl",
        ar_summary="outputs/base_tuteng/mistral_humaneval_summary.json",
        collab_output="outputs/base_tuteng/mistral_llada_remask_humaneval_t0.9.jsonl",
        collab_summary="outputs/base_tuteng/mistral_llada_remask_humaneval_t0.9_summary.json",
        refiner="llada",
        refiner_model_id="GSAI-ML/LLaDA-8B-Instruct",
    ),
    PairConfig(
        slug="codellama_llada_humaneval_t0.9",
        dataset="humaneval",
        tau_conf=0.9,
        ar_label="CodeLlama 7B",
        refiner_label="LLaDA 8B",
        ar_input="outputs/base_tuteng/codellama_humaneval.jsonl",
        ar_summary="outputs/base_tuteng/codellama_humaneval_summary.json",
        collab_output="outputs/base_tuteng/codellama_llada_remask_humaneval_t0.9.jsonl",
        collab_summary="outputs/base_tuteng/codellama_llada_remask_humaneval_t0.9_summary.json",
        refiner="llada",
        refiner_model_id="GSAI-ML/LLaDA-8B-Instruct",
    ),
    PairConfig(
        slug="seed_coder_instruct_dream_humaneval_t0.9",
        dataset="humaneval",
        tau_conf=0.9,
        ar_label="Seed-Coder-Instruct 8B",
        refiner_label="Dream-Coder 7B",
        ar_input="outputs/base_tuteng/packaging_v2/seed-coder-instruct_humaneval_pkgv2.jsonl",
        ar_summary="outputs/base_tuteng/packaging_v2/seed-coder-instruct_humaneval_pkgv2_summary.json",
        collab_output="outputs/base_tuteng/packaging_v2/seed-coder-instruct_dream_remask_humaneval_t0.9_pkgv2.jsonl",
        collab_summary="outputs/base_tuteng/packaging_v2/seed-coder-instruct_dream_remask_humaneval_t0.9_pkgv2_summary.json",
        refiner="dream",
        refiner_model_id="Dream-org/Dream-Coder-v0-Instruct-7B",
    ),
    PairConfig(
        slug="seed_coder_instruct_llada_humaneval_t0.9",
        dataset="humaneval",
        tau_conf=0.9,
        ar_label="Seed-Coder-Instruct 8B",
        refiner_label="LLaDA 8B",
        ar_input="outputs/base_tuteng/packaging_v2/seed-coder-instruct_humaneval_pkgv2.jsonl",
        ar_summary="outputs/base_tuteng/packaging_v2/seed-coder-instruct_humaneval_pkgv2_summary.json",
        collab_output="outputs/base_tuteng/packaging_v2/seed-coder-instruct_llada_remask_humaneval_t0.9_pkgv2.jsonl",
        collab_summary="outputs/base_tuteng/packaging_v2/seed-coder-instruct_llada_remask_humaneval_t0.9_pkgv2_summary.json",
        refiner="llada",
        refiner_model_id="GSAI-ML/LLaDA-8B-Instruct",
    ),
    # ── MBPP ───────────────────────────────────────────────────────────────
    PairConfig(
        slug="deepseek_dream_mbpp_t0.9",
        dataset="mbpp",
        tau_conf=0.9,
        ar_label="DeepSeek-Coder 6.7B",
        refiner_label="Dream-Coder 7B",
        ar_input="outputs/base_tuteng/deepseek_mbpp.jsonl",
        ar_summary="outputs/base_tuteng/deepseek_mbpp_summary.json",
        collab_output="outputs/remask_kodai/remask_mbpp_t0.9.jsonl",
        collab_summary="outputs/remask_kodai/remask_mbpp_t0.9_summary.json",
        refiner="dream",
        refiner_model_id="Dream-org/Dream-Coder-v0-Instruct-7B",
    ),
    PairConfig(
        slug="qwen_dream_mbpp_t0.9",
        dataset="mbpp",
        tau_conf=0.9,
        ar_label="Qwen2.5-Coder 7B",
        refiner_label="Dream-Coder 7B",
        ar_input="outputs/base_tuteng/qwen_mbpp.jsonl",
        ar_summary="outputs/base_tuteng/qwen_mbpp_summary.json",
        collab_output="outputs/base_tuteng/qwen_dream_remask_mbpp_t0.9.jsonl",
        collab_summary="outputs/base_tuteng/qwen_dream_remask_mbpp_t0.9_summary.json",
        refiner="dream",
        refiner_model_id="Dream-org/Dream-Coder-v0-Instruct-7B",
    ),
    PairConfig(
        slug="llama31_dream_mbpp_t0.9",
        dataset="mbpp",
        tau_conf=0.9,
        ar_label="Llama-3.1 8B",
        refiner_label="Dream-Coder 7B",
        ar_input="outputs/base_tuteng/llama31_mbpp.jsonl",
        ar_summary="outputs/base_tuteng/llama31_mbpp_summary.json",
        collab_output="outputs/base_tuteng/llama31_dream_remask_mbpp_t0.9.jsonl",
        collab_summary="outputs/base_tuteng/llama31_dream_remask_mbpp_t0.9_summary.json",
        refiner="dream",
        refiner_model_id="Dream-org/Dream-Coder-v0-Instruct-7B",
    ),
    PairConfig(
        slug="starcoder2_dream_mbpp_t0.9",
        dataset="mbpp",
        tau_conf=0.9,
        ar_label="StarCoder2 7B",
        refiner_label="Dream-Coder 7B",
        ar_input="outputs/base_tuteng/starcoder2_mbpp.jsonl",
        ar_summary="outputs/base_tuteng/starcoder2_mbpp_summary.json",
        collab_output="outputs/base_tuteng/starcoder2_dream_remask_mbpp_t0.9.jsonl",
        collab_summary="outputs/base_tuteng/starcoder2_dream_remask_mbpp_t0.9_summary.json",
        refiner="dream",
        refiner_model_id="Dream-org/Dream-Coder-v0-Instruct-7B",
    ),
    PairConfig(
        slug="mistral_dream_mbpp_t0.9",
        dataset="mbpp",
        tau_conf=0.9,
        ar_label="Mistral 7B",
        refiner_label="Dream-Coder 7B",
        ar_input="outputs/base_tuteng/mistral_mbpp.jsonl",
        ar_summary="outputs/base_tuteng/mistral_mbpp_summary.json",
        collab_output="outputs/base_tuteng/mistral_dream_remask_mbpp_t0.9.jsonl",
        collab_summary="outputs/base_tuteng/mistral_dream_remask_mbpp_t0.9_summary.json",
        refiner="dream",
        refiner_model_id="Dream-org/Dream-Coder-v0-Instruct-7B",
    ),
    PairConfig(
        slug="codellama_dream_mbpp_t0.9",
        dataset="mbpp",
        tau_conf=0.9,
        ar_label="CodeLlama 7B",
        refiner_label="Dream-Coder 7B",
        ar_input="outputs/base_tuteng/codellama_mbpp.jsonl",
        ar_summary="outputs/base_tuteng/codellama_mbpp_summary.json",
        collab_output="outputs/base_tuteng/codellama_dream_remask_mbpp_t0.9.jsonl",
        collab_summary="outputs/base_tuteng/codellama_dream_remask_mbpp_t0.9_summary.json",
        refiner="dream",
        refiner_model_id="Dream-org/Dream-Coder-v0-Instruct-7B",
    ),
    PairConfig(
        slug="deepseek_llada_mbpp_t0.9",
        dataset="mbpp",
        tau_conf=0.9,
        ar_label="DeepSeek-Coder 6.7B",
        refiner_label="LLaDA 8B",
        ar_input="outputs/base_tuteng/deepseek_mbpp.jsonl",
        ar_summary="outputs/base_tuteng/deepseek_mbpp_summary.json",
        collab_output="outputs/base_tuteng/deepseek_llada_remask_mbpp_t0.9.jsonl",
        collab_summary="outputs/base_tuteng/deepseek_llada_remask_mbpp_t0.9_summary.json",
        refiner="llada",
        refiner_model_id="GSAI-ML/LLaDA-8B-Instruct",
    ),
    PairConfig(
        slug="qwen_llada_mbpp_t0.9",
        dataset="mbpp",
        tau_conf=0.9,
        ar_label="Qwen2.5-Coder 7B",
        refiner_label="LLaDA 8B",
        ar_input="outputs/base_tuteng/qwen_mbpp.jsonl",
        ar_summary="outputs/base_tuteng/qwen_mbpp_summary.json",
        collab_output="outputs/base_tuteng/qwen_llada_remask_mbpp_t0.9.jsonl",
        collab_summary="outputs/base_tuteng/qwen_llada_remask_mbpp_t0.9_summary.json",
        refiner="llada",
        refiner_model_id="GSAI-ML/LLaDA-8B-Instruct",
    ),
    PairConfig(
        slug="llama31_llada_mbpp_t0.9",
        dataset="mbpp",
        tau_conf=0.9,
        ar_label="Llama-3.1 8B",
        refiner_label="LLaDA 8B",
        ar_input="outputs/base_tuteng/llama31_mbpp.jsonl",
        ar_summary="outputs/base_tuteng/llama31_mbpp_summary.json",
        collab_output="outputs/base_tuteng/llama31_llada_remask_mbpp_t0.9.jsonl",
        collab_summary="outputs/base_tuteng/llama31_llada_remask_mbpp_t0.9_summary.json",
        refiner="llada",
        refiner_model_id="GSAI-ML/LLaDA-8B-Instruct",
    ),
    PairConfig(
        slug="starcoder2_llada_mbpp_t0.9",
        dataset="mbpp",
        tau_conf=0.9,
        ar_label="StarCoder2 7B",
        refiner_label="LLaDA 8B",
        ar_input="outputs/base_tuteng/starcoder2_mbpp.jsonl",
        ar_summary="outputs/base_tuteng/starcoder2_mbpp_summary.json",
        collab_output="outputs/base_tuteng/starcoder2_llada_remask_mbpp_t0.9.jsonl",
        collab_summary="outputs/base_tuteng/starcoder2_llada_remask_mbpp_t0.9_summary.json",
        refiner="llada",
        refiner_model_id="GSAI-ML/LLaDA-8B-Instruct",
    ),
    PairConfig(
        slug="mistral_llada_mbpp_t0.9",
        dataset="mbpp",
        tau_conf=0.9,
        ar_label="Mistral 7B",
        refiner_label="LLaDA 8B",
        ar_input="outputs/base_tuteng/mistral_mbpp.jsonl",
        ar_summary="outputs/base_tuteng/mistral_mbpp_summary.json",
        collab_output="outputs/base_tuteng/mistral_llada_remask_mbpp_t0.9.jsonl",
        collab_summary="outputs/base_tuteng/mistral_llada_remask_mbpp_t0.9_summary.json",
        refiner="llada",
        refiner_model_id="GSAI-ML/LLaDA-8B-Instruct",
    ),
    PairConfig(
        slug="codellama_llada_mbpp_t0.9",
        dataset="mbpp",
        tau_conf=0.9,
        ar_label="CodeLlama 7B",
        refiner_label="LLaDA 8B",
        ar_input="outputs/base_tuteng/codellama_mbpp.jsonl",
        ar_summary="outputs/base_tuteng/codellama_mbpp_summary.json",
        collab_output="outputs/base_tuteng/codellama_llada_remask_mbpp_t0.9.jsonl",
        collab_summary="outputs/base_tuteng/codellama_llada_remask_mbpp_t0.9_summary.json",
        refiner="llada",
        refiner_model_id="GSAI-ML/LLaDA-8B-Instruct",
    ),
    PairConfig(
        slug="seed_coder_instruct_dream_mbpp_t0.9",
        dataset="mbpp",
        tau_conf=0.9,
        ar_label="Seed-Coder-Instruct 8B",
        refiner_label="Dream-Coder 7B",
        ar_input="outputs/base_tuteng/seed-coder-instruct_mbpp.jsonl",
        ar_summary="outputs/base_tuteng/seed-coder-instruct_mbpp_summary.json",
        collab_output="outputs/base_tuteng/seed-coder-instruct_dream_remask_mbpp_t0.9.jsonl",
        collab_summary="outputs/base_tuteng/seed-coder-instruct_dream_remask_mbpp_t0.9_summary.json",
        refiner="dream",
        refiner_model_id="Dream-org/Dream-Coder-v0-Instruct-7B",
    ),
    PairConfig(
        slug="seed_coder_instruct_llada_mbpp_t0.9",
        dataset="mbpp",
        tau_conf=0.9,
        ar_label="Seed-Coder-Instruct 8B",
        refiner_label="LLaDA 8B",
        ar_input="outputs/base_tuteng/seed-coder-instruct_mbpp.jsonl",
        ar_summary="outputs/base_tuteng/seed-coder-instruct_mbpp_summary.json",
        collab_output="outputs/base_tuteng/seed-coder-instruct_llada_remask_mbpp_t0.9.jsonl",
        collab_summary="outputs/base_tuteng/seed-coder-instruct_llada_remask_mbpp_t0.9_summary.json",
        refiner="llada",
        refiner_model_id="GSAI-ML/LLaDA-8B-Instruct",
    ),
)


def _expected_task_ids(dataset: str) -> list[str]:
    if dataset == "humaneval":
        return sorted(get_human_eval_plus().keys())
    if dataset == "mbpp":
        return sorted(get_mbpp_plus().keys())
    raise ValueError(f"Unsupported dataset: {dataset}")


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl_task_ids(path: str) -> list[str]:
    ids: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.append(json.loads(line)["task_id"])
    return ids


def _read_jsonl_records(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _pct(numer: int | None, denom: int | None) -> float | None:
    if numer is None or denom in (None, 0):
        return None
    return round(100.0 * numer / denom, 1)


def _task_pass_map(eval_result_path: str | None, use_plus: bool = True) -> dict[str, bool] | None:
    """Return {task_id: passed} from an evalplus eval_results.json file."""
    if eval_result_path is None or not os.path.exists(eval_result_path):
        return None
    data = _read_json(eval_result_path)
    key = "plus_status" if use_plus else "base_status"
    result: dict[str, bool] = {}
    for task_id, samples in data.get("eval", {}).items():
        result[task_id] = any(s.get(key) == "pass" for s in samples)
    return result


def _compute_flip_counts(
    ar_eval_path: str | None,
    collab_eval_path: str | None,
) -> dict[str, int | None]:
    """Count wrong→right (fixed) and right→wrong (broken) tasks between AR and collab (plus%)."""
    ar_map = _task_pass_map(ar_eval_path)
    co_map = _task_pass_map(collab_eval_path)
    if ar_map is None or co_map is None:
        return {"wrong_to_right": None, "right_to_wrong": None}
    shared = set(ar_map) & set(co_map)
    wrong_to_right = sum(1 for t in shared if not ar_map[t] and co_map[t])
    right_to_wrong = sum(1 for t in shared if ar_map[t] and not co_map[t])
    return {"wrong_to_right": wrong_to_right, "right_to_wrong": right_to_wrong}


def _summary_metrics(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    data = _read_json(path)
    summary = data.get("summary", {})
    n_tasks = summary.get("n_tasks")
    n_plus_pass = summary.get("n_plus_pass")
    return {
        "model": data.get("model"),
        "dataset": data.get("dataset"),
        "n_tasks": n_tasks,
        "n_samples_total": summary.get("n_samples_total"),
        "n_plus_pass": n_plus_pass,
        "pass_at_1_pct": _pct(n_plus_pass, n_tasks),
    }


def _existing_eval_result_path(samples_path: str) -> str | None:
    for candidate in _candidate_evalplus_result_paths(samples_path):
        if os.path.exists(candidate):
            return candidate
    return None


def _needs_refresh(source_path: str, target_path: str) -> bool:
    if not os.path.exists(target_path):
        return True
    return os.path.getmtime(source_path) > os.path.getmtime(target_path)


def _backup_path(path: str) -> str:
    return path + ".repair_backup"


def _unique_backup_path(path: str, suffix: str) -> str:
    candidate = path + suffix
    if not os.path.exists(candidate):
        return candidate
    idx = 2
    while True:
        candidate = f"{path}{suffix}.{idx}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def _repair_collab_output(pair: PairConfig) -> None:
    if not os.path.exists(pair.collab_output):
        return

    merged_records: list[dict[str, Any]] = []
    for input_path in (pair.collab_output, *pair.extra_collab_inputs):
        if os.path.exists(input_path):
            merged_records.extend(_read_jsonl_records(input_path))

    if not merged_records:
        return

    deduped_by_task: dict[str, dict[str, Any]] = {}
    first_seen_order: list[str] = []
    for rec in merged_records:
        task_id = rec["task_id"]
        if task_id not in deduped_by_task:
            first_seen_order.append(task_id)
        deduped_by_task[task_id] = rec

    deduped_records = [deduped_by_task[task_id] for task_id in first_seen_order]
    original_rows = len(_read_jsonl_records(pair.collab_output))
    if len(deduped_records) == original_rows and not any(os.path.exists(p) for p in pair.extra_collab_inputs):
        return

    backup_path = _backup_path(pair.collab_output)
    if not os.path.exists(backup_path):
        Path(backup_path).write_text(Path(pair.collab_output).read_text(encoding="utf-8"), encoding="utf-8")

    with open(pair.collab_output, "w", encoding="utf-8") as f:
        for rec in deduped_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(
        f"[repair] {pair.slug}: "
        f"main_rows={original_rows} merged_rows={len(merged_records)} deduped_rows={len(deduped_records)}"
    )


def _stash_existing_eval_results(samples_path: str) -> None:
    for candidate in _candidate_evalplus_result_paths(samples_path):
        if not os.path.exists(candidate):
            continue
        backup_path = _unique_backup_path(candidate, ".stale_backup")
        Path(candidate).replace(backup_path)
        print(f"[evalplus] moved stale result {candidate} -> {backup_path}")


def _run(cmd: list[str]) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _status_for_pair(pair: PairConfig) -> dict[str, Any]:
    expected_ids = _expected_task_ids(pair.dataset)
    expected_set = set(expected_ids)

    collab_rows = 0
    collab_unique_rows = 0
    missing_ids = expected_ids
    duplicate_rows = 0

    if os.path.exists(pair.collab_output):
        task_ids = _read_jsonl_task_ids(pair.collab_output)
        collab_rows = len(task_ids)
        collab_unique_rows = len(set(task_ids))
        duplicate_rows = collab_rows - collab_unique_rows
        missing_ids = [task_id for task_id in expected_ids if task_id not in set(task_ids)]

    sanitized_path = guess_sanitized_path(pair.collab_output)
    eval_result_path = (
        _existing_eval_result_path(sanitized_path)
        or _existing_eval_result_path(pair.collab_output)
    )
    collab_summary = _summary_metrics(pair.collab_summary)
    ar_summary = _summary_metrics(pair.ar_summary)

    ar_sanitized_path = guess_sanitized_path(pair.ar_input)
    ar_eval_result_path = (
        _existing_eval_result_path(ar_sanitized_path)
        or _existing_eval_result_path(pair.ar_input)
    )
    flip_counts = _compute_flip_counts(ar_eval_result_path, eval_result_path)

    return {
        "slug": pair.slug,
        "dataset": pair.dataset,
        "tau_conf": pair.tau_conf,
        "ar_drafter": pair.ar_label,
        "dllm_refiner": pair.refiner_label,
        "ar_only_pass_at_1_pct": None if ar_summary is None else ar_summary["pass_at_1_pct"],
        "collab_pass_at_1_pct": None if collab_summary is None else collab_summary["pass_at_1_pct"],
        "wrong_to_right": flip_counts["wrong_to_right"],
        "right_to_wrong": flip_counts["right_to_wrong"],
        "paths": {
            "ar_input": pair.ar_input,
            "ar_summary": pair.ar_summary,
            "ar_eval_results": ar_eval_result_path,
            "collab_output": pair.collab_output,
            "collab_sanitized": sanitized_path,
            "collab_eval_results": eval_result_path,
            "collab_summary": pair.collab_summary,
        },
        "collab_status": {
            "output_exists": os.path.exists(pair.collab_output),
            "rows": collab_rows,
            "unique_task_ids": collab_unique_rows,
            "expected_tasks": len(expected_ids),
            "duplicate_rows": duplicate_rows,
            "missing_tasks": len(missing_ids),
            "is_generation_complete": collab_unique_rows == len(expected_set),
            "sanitize_exists": os.path.exists(sanitized_path),
            "eval_results_exists": eval_result_path is not None,
            "summary_exists": os.path.exists(pair.collab_summary),
            "is_fully_complete": (
                collab_unique_rows == len(expected_set)
                and eval_result_path is not None
                and os.path.exists(pair.collab_summary)
            ),
            "first_missing_task_ids": missing_ids[:10],
        },
    }


def _resume_generation(pair: PairConfig) -> None:
    cmd = [
        sys.executable,
        "-m",
        "coder.scripts.gen_remask",
        "--input",
        pair.ar_input,
        "--out",
        pair.collab_output,
        "--refiner",
        pair.refiner,
        "--model_id",
        pair.refiner_model_id,
        "--confidence_threshold",
        str(pair.tau_conf),
        "--temperature",
        "0.0",
        "--top_p",
        "1.0",
        "--seed",
        "3407",
        "--resume",
    ]
    _run(cmd)


def _materialize_eval_outputs(pair: PairConfig) -> None:
    sanitized_path = guess_sanitized_path(pair.collab_output)
    if _needs_refresh(pair.collab_output, sanitized_path):
        _run(
            [
                sys.executable,
                "-m",
                "coder.scripts.postprocess_evalplus",
                "--dataset",
                pair.dataset,
                "--samples",
                pair.collab_output,
            ]
        )

    eval_result_path = _existing_eval_result_path(sanitized_path)
    if eval_result_path is None or _needs_refresh(sanitized_path, eval_result_path):
        if eval_result_path is not None:
            _stash_existing_eval_results(sanitized_path)
        _run(
            [
                sys.executable,
                "-m",
                "coder.scripts.eval_evalplus",
                "--backend",
                "local",
                "--dataset",
                pair.dataset,
                "--samples",
                sanitized_path,
                "--summary_out",
                pair.collab_summary,
                "--summary_model",
                pair.slug,
            ]
        )
        return

    if _needs_refresh(eval_result_path, pair.collab_summary):
        print(f"[summary] refreshing stale summary from {eval_result_path}")
    _write_summary(
        eval_result_path=eval_result_path,
        samples_path=sanitized_path,
        dataset=pair.dataset,
        summary_out=pair.collab_summary,
        model_name=pair.slug,
        include_per_task=False,
    )
    print(f"[summary] wrote {pair.collab_summary}")


def _write_table(rows: list[dict[str, Any]], out_path: str) -> None:
    payload = {
        "tau_conf": 0.9,
        "rows": rows,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[table] wrote {out}")


def _select_pairs(slugs: list[str] | None) -> list[PairConfig]:
    if not slugs:
        return list(PAIR_CONFIGS)
    wanted = set(slugs)
    return [pair for pair in PAIR_CONFIGS if pair.slug in wanted]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Check or run the HumanEval+ model-pair experiments used by tab:model_pairs."
    )
    ap.add_argument(
        "--pairs",
        nargs="*",
        default=None,
        help="Optional pair slugs to restrict execution.",
    )
    ap.add_argument(
        "--resume-missing",
        action="store_true",
        help="Resume incomplete collab generation with gen_remask --resume.",
    )
    ap.add_argument(
        "--repair-outputs",
        action="store_true",
        help="Dedupe collab outputs by task_id and merge any configured shard files before resuming.",
    )
    ap.add_argument(
        "--eval-complete",
        action="store_true",
        help="For complete generations, materialize sanitize/eval/summary outputs if missing.",
    )
    ap.add_argument(
        "--out",
        default="outputs/base_tuteng/model_pairs_all_t0.9.json",
        help="Where to write the consolidated table/status JSON.",
    )
    args = ap.parse_args()

    selected_pairs = _select_pairs(args.pairs)
    statuses: list[dict[str, Any]] = []

    for pair in selected_pairs:
        if args.repair_outputs:
            _repair_collab_output(pair)

        status = _status_for_pair(pair)
        statuses.append(status)
        print(
            f"[status] {pair.slug}: "
            f"rows={status['collab_status']['rows']}/"
            f"{status['collab_status']['expected_tasks']} "
            f"complete={status['collab_status']['is_fully_complete']} "
            f"summary={status['collab_pass_at_1_pct']}"
        )

        if args.resume_missing and not status["collab_status"]["is_generation_complete"]:
            _resume_generation(pair)
            status = _status_for_pair(pair)

        if args.eval_complete and status["collab_status"]["is_generation_complete"]:
            _materialize_eval_outputs(pair)
            status = _status_for_pair(pair)

        statuses[-1] = status

    _write_table(statuses, args.out)


if __name__ == "__main__":
    main()

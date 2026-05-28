from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "qwen" / "Qwen3-VL-4B-Instruct"
RESULT_DIR = ROOT / "results" / "qwen"
CURRICULUM_TEST = ROOT / "data" / "circuit_curricula" / "test.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def extract_answer(text: str) -> str:
    matches = re.findall(r"Answer\s*:\s*(.+)", text, flags=re.I)
    if matches:
        return matches[-1].strip()
    return text.strip()


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = text.replace("¡ò", "∫")
    text = text.replace("∫", " integral ")
    text = re.sub(r"[^a-z0-9가-힣+\-*/=().,% ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def answer_match(prediction: str, expected: str) -> dict[str, bool]:
    pred = normalize_answer(prediction)
    exp = normalize_answer(expected)
    exact = pred == exp
    substring = bool(exp) and (exp in pred or pred in exp)
    token_overlap = False
    pred_tokens = set(pred.split())
    exp_tokens = set(exp.split())
    if exp_tokens:
        token_overlap = len(pred_tokens & exp_tokens) / len(exp_tokens) >= 0.75
    return {
        "normalized_exact": exact,
        "normalized_relaxed": exact or substring or token_overlap,
    }


def parse_circuitvqa_ref(ref: str) -> tuple[str, int] | None:
    match = re.search(r"/(train|test)/(\d+)/images/0$", ref)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def selected_circuitvqa_refs(limit_qa_pairs: int) -> list[tuple[str, int, int, str]]:
    refs: list[tuple[str, int, int, str]] = []
    with CURRICULUM_TEST.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("source_dataset") != "ayoubkirouane/CircuitVQA":
                continue
            parsed = parse_circuitvqa_ref(record["image"]["image_path_or_ref"])
            if parsed is None:
                continue
            source_split, row_index = parsed
            for qa_index in range(6):
                refs.append((source_split, row_index, qa_index, record["sample_id"]))
                if len(refs) >= limit_qa_pairs:
                    return refs
    return refs


def load_needed_rows(refs: list[tuple[str, int, int, str]]) -> dict[tuple[str, int], dict[str, Any]]:
    needed: dict[str, set[int]] = {}
    for split, row_index, _, _ in refs:
        needed.setdefault(split, set()).add(row_index)

    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for split, indices in needed.items():
        max_index = max(indices)
        dataset = load_dataset("ayoubkirouane/CircuitVQA", split=split, streaming=True)
        for row_index, row in enumerate(dataset):
            if row_index in indices:
                rows[(split, row_index)] = row
            if row_index >= max_index:
                break
    return rows


def current_process_vram_mb() -> int | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    pid = str(os.getpid())
    total = 0
    found = False
    for raw_line in output.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 2:
            continue
        if parts[0] == pid:
            found = True
            try:
                total += int(parts[1])
            except ValueError:
                pass
    return total if found else None


def gpu_used_vram_mb() -> int | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    first = output.splitlines()[0].strip() if output.splitlines() else ""
    try:
        return int(first)
    except ValueError:
        return None


def gpu_summary() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "total_vram_mb": round(props.total_memory / 1024 / 1024),
        "capability": list(torch.cuda.get_device_capability(0)),
    }


def run_qwen3(args: argparse.Namespace) -> dict[str, Any]:
    process = psutil.Process(os.getpid())
    gpu_used_before_load = gpu_used_vram_mb()
    refs = selected_circuitvqa_refs(args.limit_qa_pairs)
    rows = load_needed_rows(refs)

    load_start = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    load_sec = time.perf_counter() - load_start
    total_parameters = sum(param.numel() for param in model.parameters())
    trainable_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    resident_after_load_process = current_process_vram_mb()
    gpu_used_after_load = gpu_used_vram_mb()
    torch.cuda.reset_peak_memory_stats()

    eval_records: list[dict[str, Any]] = []
    exact_hits = 0
    relaxed_hits = 0
    latencies: list[float] = []
    prompt_suffix = "\nAnswer with a short final answer only."

    for ordinal, (source_split, row_index, qa_index, sample_id) in enumerate(refs):
        row = rows[(source_split, row_index)]
        qa = row["texts"][qa_index]
        expected = extract_answer(qa["assistant"])
        question = qa["user"]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": row["images"][0]},
                    {"type": "text", "text": question + prompt_suffix},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        start = time.perf_counter()
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        latency_ms = (time.perf_counter() - start) * 1000
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        prediction = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        match = answer_match(prediction, expected)
        exact_hits += int(match["normalized_exact"])
        relaxed_hits += int(match["normalized_relaxed"])
        latencies.append(latency_ms)
        eval_records.append(
            {
                "ordinal": ordinal,
                "sample_id": sample_id,
                "source_split": source_split,
                "row_index": row_index,
                "qa_index": qa_index,
                "latency_ms": round(latency_ms, 3),
                "normalized_exact": match["normalized_exact"],
                "normalized_relaxed": match["normalized_relaxed"],
                "prediction_chars": len(prediction),
                "expected_answer_chars": len(expected),
            }
        )
        del inputs, generated_ids, generated_ids_trimmed
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    count = len(eval_records)
    resident_after_eval_process = current_process_vram_mb()
    gpu_used_after_eval = gpu_used_vram_mb()
    peak_allocated = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 3) if torch.cuda.is_available() else None
    peak_reserved = round(torch.cuda.max_memory_reserved() / 1024 / 1024, 3) if torch.cuda.is_available() else None
    ram_mb = round(process.memory_info().rss / 1024 / 1024, 3)

    return {
        "result_version": 1,
        "status": "completed",
        "generated_at_utc": utc_now(),
        "script": "scripts/run_qwen_baseline.py",
        "command": {
            "model_dir": rel(args.model_dir),
            "baseline_id": args.baseline_id,
            "limit_qa_pairs": args.limit_qa_pairs,
            "max_new_tokens": args.max_new_tokens,
        },
        "dataset": {
            "name": "ayoubkirouane/CircuitVQA",
            "source_url": "https://huggingface.co/datasets/ayoubkirouane/CircuitVQA",
            "license": "not_declared_on_hf",
            "curriculum": "data/circuit_curricula/test.jsonl",
            "selected_subset_policy": "first deterministic CircuitVQA test QA refs from M2 curriculum",
            "qa_pairs": count,
            "prompt_answer_payload_committed": False,
        },
        "model": {
            "baseline_id": args.baseline_id,
            "model_family": "Qwen3-VL",
            "model_path": rel(args.model_dir),
            "dtype": "bfloat16",
            "device_map": "auto",
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
        },
        "hardware": gpu_summary(),
        "metrics": {
            "circuit_vqa_score": relaxed_hits / count if count else None,
            "normalized_exact_match": exact_hits / count if count else None,
            "normalized_relaxed_match": relaxed_hits / count if count else None,
            "structured_extraction_score_if_available": None,
            "resident_vram_mb_after_load": resident_after_load_process,
            "resident_vram_mb_after_eval": resident_after_eval_process,
            "gpu_used_vram_mb_before_load": gpu_used_before_load,
            "gpu_used_vram_mb_after_load": gpu_used_after_load,
            "gpu_used_vram_mb_after_eval": gpu_used_after_eval,
            "resident_vram_mb_after_load_delta": (
                gpu_used_after_load - gpu_used_before_load
                if gpu_used_after_load is not None and gpu_used_before_load is not None
                else None
            ),
            "resident_vram_mb_after_eval_delta": (
                gpu_used_after_eval - gpu_used_before_load
                if gpu_used_after_eval is not None and gpu_used_before_load is not None
                else None
            ),
            "peak_vram_mb_torch_allocated": peak_allocated,
            "peak_vram_mb_torch_reserved": peak_reserved,
            "latency_ms_mean": sum(latencies) / len(latencies) if latencies else None,
            "latency_ms_min": min(latencies) if latencies else None,
            "latency_ms_max": max(latencies) if latencies else None,
            "load_seconds": load_sec,
            "process_ram_mb_after_eval": ram_mb,
        },
        "per_question": eval_records,
        "payload_policy": {
            "stores_raw_prompts": False,
            "stores_expected_answers": False,
            "stores_predictions": False,
        },
    }


def write_deferred(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "result_version": 1,
        "status": "deferred",
        "generated_at_utc": utc_now(),
        "script": "scripts/run_qwen_baseline.py",
        "baseline_id": args.baseline_id,
        "reason": args.defer_reason,
        "pass_condition_interpretation": "M3 allows the smaller_or_quantized baseline to be explicitly deferred.",
        "local_model_inventory_note": "No smaller or quantized Qwen VLM checkpoint was present under models/qwen at run time.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or defer M3 Qwen baselines.")
    parser.add_argument("--baseline-id", default="qwen3_vl_single")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=RESULT_DIR / "qwen3_single_backbone.json")
    parser.add_argument("--limit-qa-pairs", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--defer", action="store_true")
    parser.add_argument("--defer-reason", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.model_dir = args.model_dir.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.defer:
        payload = write_deferred(args)
    else:
        payload = run_qwen3(args)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": rel(args.output)}, ensure_ascii=False, sort_keys=True))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

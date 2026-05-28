from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import datasets
import transformers
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "qwen" / "Qwen3-VL-4B-Instruct"
DEFAULT_MANIFEST = ROOT / "data" / "circuit_curricula" / "test.jsonl"
DEFAULT_OUTPUT = ROOT / "results" / "qwen" / "qwen3_single_backbone.json"
DEFAULT_DEFERRED = ROOT / "results" / "qwen" / "qwen_small_or_quantized.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_answer(value: str) -> str:
    text = value.lower().strip()
    text = re.sub(r"^answer\s*:\s*", "", text)
    text = re.sub(r"[^a-z0-9가-힣]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_expected(value: str) -> str:
    match = re.search(r"Answer:\s*(.+)", value or "", flags=re.I | re.S)
    answer = match.group(1) if match else value
    return answer.strip().splitlines()[0].strip()


def is_correct(prediction: str, expected: str) -> bool:
    pred_norm = normalize_answer(prediction)
    expected_norm = normalize_answer(expected)
    if not pred_norm or not expected_norm:
        return False
    return expected_norm == pred_norm or expected_norm in pred_norm or pred_norm in expected_norm


def short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def gpu_used_mb() -> int | None:
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
    first = output.strip().splitlines()[0]
    return int(first.strip()) if first.strip().isdigit() else None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def artifact_ref(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.name


def load_vqa_samples(manifest_path: Path, max_samples: int) -> list[dict[str, Any]]:
    samples = [row for row in read_jsonl(manifest_path) if row.get("sample_family") == "vqa_evaluation"]
    return samples[:max_samples]


def run(args: argparse.Namespace) -> dict[str, Any]:
    model_path = Path(args.model_path).resolve()
    manifest_path = Path(args.manifest).resolve()
    samples = load_vqa_samples(manifest_path, args.max_samples)
    if not samples:
        raise RuntimeError(f"no VQA samples found in {manifest_path}")

    result: dict[str, Any] = {
        "status": "running",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model": {
            "name": "Qwen3-VL-4B-Instruct",
            "path": artifact_ref(model_path),
            "dtype": args.dtype,
            "local_files_only": True,
        },
        "dataset": {
            "source_dataset": "ayoubkirouane/CircuitVQA",
            "source_split": "test",
            "manifest": artifact_ref(manifest_path),
            "selected_vqa_samples": len(samples),
            "selection_policy": "first N vqa_evaluation rows from M2 test manifest",
            "license_caveat": "HF dataset card does not declare a license; result stores refs/hashes, not raw prompts or expected answers.",
        },
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "transformers_loader": "Qwen3VLForConditionalGeneration",
            "clean_process": True,
            "max_new_tokens": args.max_new_tokens,
        },
        "metrics": {},
        "records": [],
    }

    write_json(Path(args.output), result)

    vram_before_load = gpu_used_mb()
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path),
        dtype=dtype,
        device_map="cuda",
        local_files_only=True,
    )
    model.eval()
    parameter_count = sum(param.numel() for param in model.parameters())
    vram_after_load = gpu_used_mb()

    dataset = load_dataset("ayoubkirouane/CircuitVQA", split="test")

    correct = 0
    latencies: list[float] = []
    max_resident_vram_mb = vram_after_load or 0
    peak_allocated_mb = 0.0
    peak_reserved_mb = 0.0

    for index, sample in enumerate(samples, start=1):
        row_idx = int(sample["source"]["row_idx"])
        question_index = int(sample["source"]["question_index"])
        row = dataset[row_idx]
        image = row["images"][0].convert("RGB")
        qa = row["texts"][question_index]
        prompt = qa["user"]
        expected = extract_expected(qa["assistant"])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"{prompt}\nAnswer in a short phrase."},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        generated_ids_trimmed = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        prediction = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        sample_correct = is_correct(prediction, expected)
        correct += int(sample_correct)
        latencies.append(elapsed_ms)
        resident_now = gpu_used_mb()
        if resident_now:
            max_resident_vram_mb = max(max_resident_vram_mb, resident_now)
        peak_allocated_mb = max(peak_allocated_mb, torch.cuda.max_memory_allocated() / 1024 / 1024)
        peak_reserved_mb = max(peak_reserved_mb, torch.cuda.max_memory_reserved() / 1024 / 1024)

        result["records"].append(
            {
                "sample_id": sample["sample_id"],
                "source_group_key": sample["source_group_key"],
                "source_row_idx": row_idx,
                "question_index": question_index,
                "expected_answer_hash": short_hash(normalize_answer(expected)),
                "prediction_hash": short_hash(normalize_answer(prediction)),
                "prediction_chars": len(prediction),
                "correct_contains_normalized": sample_correct,
                "latency_ms": round(elapsed_ms, 3),
            }
        )
        if index % args.checkpoint_every == 0 or index == len(samples):
            result["status"] = "partial" if index < len(samples) else "complete"
            result["metrics"] = summarize_metrics(
                correct=correct,
                total=index,
                latencies=latencies,
                parameter_count=parameter_count,
                vram_before_load=vram_before_load,
                vram_after_load=vram_after_load,
                max_resident_vram_mb=max_resident_vram_mb,
                peak_allocated_mb=peak_allocated_mb,
                peak_reserved_mb=peak_reserved_mb,
            )
            write_json(Path(args.output), result)
            print(f"completed {index}/{len(samples)}")

    return result


def summarize_metrics(
    *,
    correct: int,
    total: int,
    latencies: list[float],
    parameter_count: int,
    vram_before_load: int | None,
    vram_after_load: int | None,
    max_resident_vram_mb: int | None,
    peak_allocated_mb: float,
    peak_reserved_mb: float,
) -> dict[str, Any]:
    sorted_latencies = sorted(latencies)
    p50 = sorted_latencies[len(sorted_latencies) // 2] if sorted_latencies else None
    p95 = sorted_latencies[int(len(sorted_latencies) * 0.95) - 1] if sorted_latencies else None
    return {
        "sample_count": total,
        "correct_contains_normalized": correct,
        "circuit_vqa_score_contains_normalized": round(correct / total, 6) if total else None,
        "latency_ms_mean": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "latency_ms_p50": round(p50, 3) if p50 is not None else None,
        "latency_ms_p95": round(p95, 3) if p95 is not None else None,
        "total_parameters": parameter_count,
        "vram_before_load_mb": vram_before_load,
        "resident_vram_after_load_mb": vram_after_load,
        "max_resident_vram_observed_mb": max_resident_vram_mb,
        "peak_torch_allocated_mb": round(peak_allocated_mb, 3),
        "peak_torch_reserved_mb": round(peak_reserved_mb, 3),
    }


def write_deferred_small(path: Path) -> None:
    payload = {
        "status": "deferred",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "baseline": "smaller_or_quantized_qwen",
        "reason": "No smaller or quantized Qwen VLM checkpoint is present under local models/qwen at M3 time.",
        "required_before_claim": [
            "download or build a smaller/quantized Qwen VLM checkpoint",
            "run the same selected CircuitVQA test subset in a separate clean process",
            "record memory and latency with the same schema as qwen3_single_backbone.json",
        ],
    }
    write_json(path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M3 Qwen VLM baseline on CircuitVQA test refs.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--deferred-small-output", default=str(DEFAULT_DEFERRED))
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=8)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args)
    write_deferred_small(Path(args.deferred_small_output))
    print(json.dumps(result["metrics"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

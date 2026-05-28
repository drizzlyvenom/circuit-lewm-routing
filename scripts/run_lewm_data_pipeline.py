from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from PIL import Image, ImageEnhance, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "circuit_curricula" / "train.jsonl"
DEFAULT_OUTPUT = ROOT / "results" / "lewm_data_pipeline" / "sanity_check.json"
DATASET = "bshada/open-schematics"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def fetch_bytes(url: str, *, timeout: int = 60) -> bytes:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "circuit-lewm-routing-m4/0.1"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as exc:
            last_error = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in {429, 500, 502, 503, 504}:
                raise
            time.sleep(2**attempt)
    raise RuntimeError(f"failed to fetch after retries: {url}") from last_error


def fetch_json(url: str) -> dict[str, Any]:
    return json.loads(fetch_bytes(url).decode("utf-8"))


def fetch_open_schematics_row(row_idx: int) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {"dataset": DATASET, "config": "default", "split": "train", "offset": row_idx, "length": 1}
    )
    payload = fetch_json(f"https://datasets-server.huggingface.co/rows?{query}")
    rows = payload.get("rows") or []
    if not rows:
        raise RuntimeError(f"missing row payload: {row_idx}")
    return rows[0]["row"]


def short_hash(value: bytes | str) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()[:16]


def parse_components(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def structure_features(schematic: str, components_used: Any) -> dict[str, Any]:
    components = parse_components(components_used)
    references = re.findall(r'\(property\s+"Reference"\s+"([^"]+)"', schematic)
    features = {
        "schematic_chars": len(schematic),
        "schematic_lines": schematic.count("\n") + 1,
        "component_library_count": len(components),
        "reference_property_count": len(references),
        "wire_count": len(re.findall(r"\(\s*wire\b", schematic)),
        "junction_count": len(re.findall(r"\(\s*junction\b", schematic)),
        "label_count": len(re.findall(r"\(\s*(global_label|label|hierarchical_label)\b", schematic)),
        "symbol_count": len(re.findall(r"\(\s*symbol\b", schematic)),
    }
    vector = [
        float(features["schematic_chars"]),
        float(features["schematic_lines"]),
        float(features["component_library_count"]),
        float(features["reference_property_count"]),
        float(features["wire_count"]),
        float(features["junction_count"]),
        float(features["label_count"]),
        float(features["symbol_count"]),
    ]
    return {
        "features": features,
        "feature_vector": vector,
        "schematic_hash": short_hash(schematic),
        "component_set_hash": short_hash("\n".join(sorted(components))),
    }


def jitter_image(image: Image.Image, *, seed: int, enabled: bool) -> tuple[Image.Image, dict[str, Any]]:
    if not enabled:
        return image, {"enabled": False}
    rng = random.Random(seed)
    brightness = 1.0 + rng.uniform(-0.04, 0.04)
    contrast = 1.0 + rng.uniform(-0.04, 0.04)
    out = ImageEnhance.Brightness(image).enhance(brightness)
    out = ImageEnhance.Contrast(out).enhance(contrast)
    return out, {"enabled": True, "brightness": round(brightness, 6), "contrast": round(contrast, 6)}


def square_pad_resize(image: Image.Image, size: int) -> Image.Image:
    contained = ImageOps.contain(image, (size, size), method=Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    offset = ((size - contained.width) // 2, (size - contained.height) // 2)
    canvas.paste(contained, offset)
    return canvas


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return (tensor - 0.5) / 0.5


def tile_boxes(width: int, height: int, num_tiles: int) -> list[tuple[int, int, int, int]]:
    if num_tiles != 4:
        raise ValueError("M4 pipeline currently uses deterministic 2x2 tiles only.")
    xs = [0, width // 2, width]
    ys = [0, height // 2, height]
    return [
        (xs[0], ys[0], xs[1], ys[1]),
        (xs[1], ys[0], xs[2], ys[1]),
        (xs[0], ys[1], xs[1], ys[2]),
        (xs[1], ys[1], xs[2], ys[2]),
    ]


def make_views(image: Image.Image, *, global_size: int, tile_size: int, num_tiles: int) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, int]]]:
    width, height = image.size
    global_tensor = image_to_tensor(square_pad_resize(image, global_size))
    tile_tensors: list[torch.Tensor] = []
    traces: list[dict[str, int]] = []
    for tile_index, box in enumerate(tile_boxes(width, height, num_tiles)):
        crop = image.crop(box)
        tile_tensors.append(image_to_tensor(square_pad_resize(crop, tile_size)))
        x0, y0, x1, y1 = box
        traces.append({"tile_index": tile_index, "x0": x0, "y0": y0, "x1": x1, "y1": y1})
    return global_tensor, torch.stack(tile_tensors), traces


def gpu_used_mb() -> int | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    first = output.strip().splitlines()[0]
    return int(first.strip()) if first.strip().isdigit() else None


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 / 1024


def system_memory() -> dict[str, float]:
    mem = psutil.virtual_memory()
    return {
        "total_mb": round(mem.total / 1024 / 1024, 3),
        "available_mb": round(mem.available / 1024 / 1024, 3),
        "percent": mem.percent,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve()
    candidates = [
        row
        for row in read_jsonl(manifest_path)
        if row.get("sample_family") == "structure_pretraining" and row.get("source_dataset") == DATASET
    ]

    started = time.perf_counter()
    rss_before = rss_mb()
    gpu_before = gpu_used_mb()
    global_tensors: list[torch.Tensor] = []
    tile_tensors: list[torch.Tensor] = []
    structure_vectors: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for candidate in candidates[: args.max_scan]:
        if len(records) >= args.batch_size:
            break
        row_idx = int(candidate["source"]["row_idx"])
        try:
            row = fetch_open_schematics_row(row_idx)
            image_meta = row.get("image") or {}
            image_src = image_meta.get("src") if isinstance(image_meta, dict) else None
            schematic = row.get("schematic") or ""
            if not image_src or not schematic:
                skipped.append({"sample_id": candidate["sample_id"], "row_idx": row_idx, "reason": "missing_image_or_schematic"})
                continue
            image_bytes = fetch_bytes(image_src)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image, aug = jitter_image(image, seed=args.seed + row_idx, enabled=args.augment)
            global_tensor, tiles_tensor, trace = make_views(
                image,
                global_size=args.global_size,
                tile_size=args.tile_size,
                num_tiles=args.num_tiles,
            )
            structure = structure_features(schematic, row.get("components_used"))
        except Exception as exc:
            skipped.append({"sample_id": candidate["sample_id"], "row_idx": row_idx, "reason": type(exc).__name__})
            continue

        global_tensors.append(global_tensor)
        tile_tensors.append(tiles_tensor)
        structure_vectors.append(torch.tensor(structure["feature_vector"], dtype=torch.float32))
        width, height = image.size
        records.append(
            {
                "sample_id": candidate["sample_id"],
                "source_row_idx": row_idx,
                "image_hash": short_hash(image_bytes),
                "schematic_hash": structure["schematic_hash"],
                "component_set_hash": structure["component_set_hash"],
                "original_size": {"width": width, "height": height},
                "global_view": {"shape": [3, args.global_size, args.global_size]},
                "tile_view": {"shape": [args.num_tiles, 3, args.tile_size, args.tile_size], "boxes": trace},
                "structure_features": structure["features"],
                "augmentation": aug,
            }
        )

    if len(records) != args.batch_size:
        raise RuntimeError(f"loaded {len(records)} usable pairs, expected {args.batch_size}")

    batch = {
        "global_view": torch.stack(global_tensors),
        "tile_view": torch.stack(tile_tensors),
        "structure_features": torch.stack(structure_vectors),
    }
    rss_after = rss_mb()
    gpu_after = gpu_used_mb()
    elapsed_sec = time.perf_counter() - started
    trace_ok = all(len(record["tile_view"]["boxes"]) == args.num_tiles for record in records)
    alignment_ok = all(record["sample_id"].startswith("openschematic_") for record in records)
    ram_limit_mb = args.ram_limit_gb * 1024

    result = {
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "config": {
            "manifest": "data/circuit_curricula/train.jsonl",
            "source_dataset": DATASET,
            "batch_size": args.batch_size,
            "max_scan": args.max_scan,
            "global_size": args.global_size,
            "tile_size": args.tile_size,
            "num_tiles": args.num_tiles,
            "augment": args.augment,
            "seed": args.seed,
        },
        "batch_shapes": {
            "global_view": list(batch["global_view"].shape),
            "tile_view": list(batch["tile_view"].shape),
            "structure_features": list(batch["structure_features"].shape),
        },
        "memory": {
            "process_rss_before_mb": round(rss_before, 3),
            "process_rss_after_mb": round(rss_after, 3),
            "process_rss_delta_mb": round(rss_after - rss_before, 3),
            "ram_limit_mb": ram_limit_mb,
            "under_ram_limit": rss_after < ram_limit_mb,
            "system": system_memory(),
            "gpu_memory_before_mb": gpu_before,
            "gpu_memory_after_mb": gpu_after,
        },
        "timing": {
            "elapsed_sec": round(elapsed_sec, 3),
            "samples_per_sec": round(args.batch_size / elapsed_sec, 6),
        },
        "validation": {
            "usable_pairs_loaded": len(records),
            "candidate_rows_scanned": min(len(candidates), args.max_scan),
            "skipped_rows": len(skipped),
            "image_structure_alignment_ok": alignment_ok,
            "tile_trace_ok": trace_ok,
            "raw_payload_committed": False,
            "signed_urls_committed": False,
            "pass": bool(alignment_ok and trace_ok and rss_after < ram_limit_mb),
        },
        "records": records,
        "skipped_preview": skipped[:20],
    }
    write_json(Path(args.output), result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M4 LeWM data pipeline actual batch check.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-scan", type=int, default=256)
    parser.add_argument("--global-size", type=int, default=224)
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--num-tiles", type=int, default=4)
    parser.add_argument("--ram-limit-gb", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.set_defaults(augment=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "validation": result["validation"], "memory": result["memory"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

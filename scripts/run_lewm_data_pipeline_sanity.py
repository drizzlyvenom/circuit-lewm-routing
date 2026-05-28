from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image as PILImage
from PIL import ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
CURRICULUM_TRAIN = ROOT / "data" / "circuit_curricula" / "train.jsonl"
RESULT_PATH = ROOT / "results" / "lewm_data_pipeline" / "sanity_check.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_hf_dataset_ref(ref: str) -> tuple[str, str, str, int, str]:
    match = re.match(r"^hf://datasets/([^/]+/[^/]+)/([^/]+)/([^/]+)/(\d+)/(.+)$", ref)
    if not match:
        raise ValueError(f"Unsupported HF dataset ref: {ref}")
    dataset_id, config, split, row_index, field = match.groups()
    return dataset_id, config, split, int(row_index), field


def load_open_schematics_rows(indices: set[int]) -> dict[int, dict[str, Any]]:
    if not indices:
        return {}
    max_index = max(indices)
    rows: dict[int, dict[str, Any]] = {}
    dataset = load_dataset("bshada/open-schematics", split="train", streaming=True).cast_column(
        "image",
        HFImage(decode=True),
    )
    for row_index, row in enumerate(dataset):
        if row_index in indices:
            copied = dict(row)
            image = copied.get("image")
            if isinstance(image, PILImage.Image):
                copied["image"] = image.convert("RGB").copy()
            rows[row_index] = copied
        if row_index >= max_index:
            break
    missing = sorted(indices - set(rows))
    if missing:
        raise RuntimeError(f"Missing open-schematics rows: {missing[:8]}")
    return rows


def select_records(records: list[dict[str, Any]], open_count: int, cghd_count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    open_records = [
        record
        for record in records
        if record.get("source_dataset") == "bshada/open-schematics"
        and record.get("structure", {}).get("structure_path_or_ref")
    ]
    cghd_records = [
        record
        for record in records
        if record.get("source_dataset") == "lowercaseonly/cghd"
        and record.get("structure", {}).get("structure_path_or_ref")
    ]
    selected.extend(open_records[:open_count])
    selected.extend(cghd_records[:cghd_count])
    return selected


def letterbox_resize(image: PILImage.Image, size: int) -> PILImage.Image:
    image = image.convert("RGB")
    ratio = min(size / image.width, size / image.height)
    new_size = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
    resized = image.resize(new_size, PILImage.Resampling.BICUBIC)
    canvas = PILImage.new("RGB", (size, size), "white")
    offset = ((size - new_size[0]) // 2, (size - new_size[1]) // 2)
    canvas.paste(resized, offset)
    return canvas


def deterministic_augment(image: PILImage.Image, sample_id: str) -> PILImage.Image:
    bucket = int(hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:8], 16)
    image = image.convert("RGB")
    if bucket % 2 == 0:
        image = ImageOps.autocontrast(image)
    brightness = 0.96 + ((bucket % 9) * 0.01)
    contrast = 0.96 + (((bucket // 9) % 9) * 0.01)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    return ImageEnhance.Contrast(image).enhance(contrast)


def image_to_tensor(image: PILImage.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def grid_boxes(width: int, height: int, count: int) -> list[list[int]]:
    columns = math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    boxes: list[list[int]] = []
    for row in range(rows):
        for col in range(columns):
            if len(boxes) >= count:
                break
            left = round(col * width / columns)
            top = round(row * height / rows)
            right = round((col + 1) * width / columns)
            bottom = round((row + 1) * height / rows)
            boxes.append([left, top, max(left + 1, right), max(top + 1, bottom)])
    return boxes


def expand_box(box: list[float], width: int, height: int, context_ratio: float = 0.25) -> list[int]:
    left, top, right, bottom = box
    box_width = right - left
    box_height = bottom - top
    pad_x = box_width * context_ratio
    pad_y = box_height * context_ratio
    return [
        max(0, math.floor(left - pad_x)),
        max(0, math.floor(top - pad_y)),
        min(width, math.ceil(right + pad_x)),
        min(height, math.ceil(bottom + pad_y)),
    ]


def tile_boxes_for_record(record: dict[str, Any], width: int, height: int, tiles_per_sample: int) -> tuple[str, list[list[int]]]:
    roi_boxes = record.get("supervision", {}).get("roi_or_tile_boxes") or []
    expanded: list[list[int]] = []
    if roi_boxes:
        sorted_boxes = sorted(
            roi_boxes,
            key=lambda item: (item.get("bbox_xyxy", [0, 0, 0, 0])[2] - item.get("bbox_xyxy", [0, 0, 0, 0])[0])
            * (item.get("bbox_xyxy", [0, 0, 0, 0])[3] - item.get("bbox_xyxy", [0, 0, 0, 0])[1]),
            reverse=True,
        )
        for item in sorted_boxes:
            raw_box = item.get("bbox_xyxy")
            if isinstance(raw_box, list) and len(raw_box) == 4:
                expanded.append(expand_box(raw_box, width, height))
            if len(expanded) >= tiles_per_sample:
                break
    if len(expanded) < tiles_per_sample:
        expanded.extend(grid_boxes(width, height, tiles_per_sample - len(expanded)))
    return ("roi_then_grid" if roi_boxes else "grid"), expanded[:tiles_per_sample]


def parse_cghd_labels(xml_path: Path) -> Counter[str]:
    labels: Counter[str] = Counter()
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return labels
    for obj in root.findall(".//object"):
        label = (obj.findtext("name") or "unknown").strip() or "unknown"
        labels[label] += 1
    return labels


def component_counter(value: Any) -> Counter[str]:
    if not isinstance(value, list):
        return Counter()
    return Counter(str(item) for item in value if item is not None)


class CircuitLeWMSanityDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[dict[str, Any]],
        open_rows: dict[int, dict[str, Any]],
        global_resolution: int,
        tile_resolution: int,
        tiles_per_sample: int,
    ) -> None:
        self.records = records
        self.open_rows = open_rows
        self.global_resolution = global_resolution
        self.tile_resolution = tile_resolution
        self.tiles_per_sample = tiles_per_sample

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image, structure_target = self.load_image_and_structure(record)
        width, height = image.width, image.height
        augmented = deterministic_augment(image, record["sample_id"])
        global_view = image_to_tensor(letterbox_resize(augmented, self.global_resolution))

        crop_policy, boxes = tile_boxes_for_record(record, width, height, self.tiles_per_sample)
        tile_tensors: list[torch.Tensor] = []
        tile_metadata: list[dict[str, Any]] = []
        for tile_index, box in enumerate(boxes):
            crop = image.crop(tuple(box))
            tile_tensors.append(image_to_tensor(letterbox_resize(crop, self.tile_resolution)))
            tile_metadata.append(
                {
                    "sample_id": record["sample_id"],
                    "source_dataset": record["source_dataset"],
                    "source_image_ref": record["image"]["image_path_or_ref"],
                    "tile_index": tile_index,
                    "crop_policy": crop_policy,
                    "crop_xyxy_original": box,
                    "original_width": width,
                    "original_height": height,
                }
            )

        return {
            "sample_id": record["sample_id"],
            "source_dataset": record["source_dataset"],
            "global_view": global_view,
            "tile_views": torch.stack(tile_tensors),
            "structure_target": structure_target,
            "tile_metadata": tile_metadata,
        }

    def load_image_and_structure(self, record: dict[str, Any]) -> tuple[PILImage.Image, dict[str, Any]]:
        source_dataset = record["source_dataset"]
        if source_dataset == "bshada/open-schematics":
            _, _, _, row_index, _ = parse_hf_dataset_ref(record["image"]["image_path_or_ref"])
            row = self.open_rows[row_index]
            image = row["image"].convert("RGB")
            schematic = row.get("schematic") or ""
            components = component_counter(record.get("structure", {}).get("component_list"))
            return image, {
                "sample_id": record["sample_id"],
                "target_type": "kicad_schematic_text",
                "structure_ref": record["structure"]["structure_path_or_ref"],
                "structure_sha1": sha1_text(schematic),
                "structure_chars": len(schematic),
                "component_count": sum(components.values()),
                "unique_component_count": len(components),
                "top_components": components.most_common(8),
                "metadata_text_sha1": sha1_text(record.get("structure", {}).get("metadata_text") or ""),
            }

        if source_dataset == "lowercaseonly/cghd":
            image_path = ROOT / record["image"]["image_path_or_ref"]
            xml_path = ROOT / record["structure"]["structure_path_or_ref"]
            labels = parse_cghd_labels(xml_path)
            return PILImage.open(image_path).convert("RGB"), {
                "sample_id": record["sample_id"],
                "target_type": "cghd_xml_symbol_annotations",
                "structure_ref": record["structure"]["structure_path_or_ref"],
                "structure_sha1": sha1_file(xml_path),
                "structure_chars": xml_path.stat().st_size,
                "component_count": sum(labels.values()),
                "unique_component_count": len(labels),
                "top_components": labels.most_common(8),
                "metadata_text_sha1": None,
            }

        raise ValueError(f"Unsupported source dataset for M4 sanity: {source_dataset}")


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_ids": [item["sample_id"] for item in items],
        "source_datasets": [item["source_dataset"] for item in items],
        "global_views": torch.stack([item["global_view"] for item in items]),
        "tile_views": torch.stack([item["tile_views"] for item in items]),
        "structure_targets": [item["structure_target"] for item in items],
        "tile_metadata": [tile for item in items for tile in item["tile_metadata"]],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run M4 LeWM data pipeline batch sanity check.")
    parser.add_argument("--train-manifest", type=Path, default=CURRICULUM_TRAIN)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--open-count", type=int, default=32)
    parser.add_argument("--cghd-count", type=int, default=32)
    parser.add_argument("--global-resolution", type=int, default=224)
    parser.add_argument("--tile-resolution", type=int, default=224)
    parser.add_argument("--tiles-per-sample", type=int, default=4)
    parser.add_argument("--ram-limit-mb", type=int, default=26 * 1024)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.train_manifest = args.train_manifest.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    process = psutil.Process()
    rss_before_mb = round(process.memory_info().rss / 1024 / 1024, 3)
    start = time.perf_counter()
    records = read_jsonl(args.train_manifest)
    selected = select_records(records, args.open_count, args.cghd_count)
    open_indices = {
        parse_hf_dataset_ref(record["image"]["image_path_or_ref"])[3]
        for record in selected
        if record["source_dataset"] == "bshada/open-schematics"
    }
    open_rows = load_open_schematics_rows(open_indices)

    dataset = CircuitLeWMSanityDataset(
        selected,
        open_rows,
        args.global_resolution,
        args.tile_resolution,
        args.tiles_per_sample,
    )
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0, collate_fn=collate_batch)
    batch = next(iter(loader))
    elapsed_seconds = time.perf_counter() - start
    rss_after_mb = round(process.memory_info().rss / 1024 / 1024, 3)

    source_counts = Counter(batch["source_datasets"])
    target_types = Counter(target["target_type"] for target in batch["structure_targets"])
    tile_metadata = batch["tile_metadata"]
    structure_targets = batch["structure_targets"]

    pass_checks = {
        "one_batch_loaded": len(batch["sample_ids"]) == len(selected) and len(selected) > 0,
        "batch_size_matches_requested_total": len(batch["sample_ids"]) == args.open_count + args.cghd_count,
        "ram_under_limit": rss_after_mb <= args.ram_limit_mb,
        "global_view_shape_ok": list(batch["global_views"].shape)
        == [len(selected), 3, args.global_resolution, args.global_resolution],
        "tile_view_shape_ok": list(batch["tile_views"].shape)
        == [len(selected), args.tiles_per_sample, 3, args.tile_resolution, args.tile_resolution],
        "image_structure_targets_aligned": all(
            sample_id == target["sample_id"] and bool(target.get("structure_sha1"))
            for sample_id, target in zip(batch["sample_ids"], structure_targets)
        ),
        "tile_metadata_traceable": len(tile_metadata) == len(selected) * args.tiles_per_sample
        and all(
            item.get("sample_id")
            and item.get("source_image_ref")
            and isinstance(item.get("crop_xyxy_original"), list)
            and len(item["crop_xyxy_original"]) == 4
            for item in tile_metadata
        ),
        "structure_sources_mixed": source_counts.get("bshada/open-schematics", 0) > 0
        and source_counts.get("lowercaseonly/cghd", 0) > 0,
    }

    output = {
        "result_version": 1,
        "status": "closed" if all(pass_checks.values()) else "closed_with_caveats",
        "generated_at_utc": utc_now(),
        "script": "scripts/run_lewm_data_pipeline_sanity.py",
        "command": {
            "train_manifest": rel(args.train_manifest),
            "open_count": args.open_count,
            "cghd_count": args.cghd_count,
            "global_resolution": args.global_resolution,
            "tile_resolution": args.tile_resolution,
            "tiles_per_sample": args.tiles_per_sample,
            "ram_limit_mb": args.ram_limit_mb,
        },
        "dataset": {
            "manifest": rel(args.train_manifest),
            "selected_records": len(selected),
            "source_counts": dict(source_counts),
            "target_types": dict(target_types),
            "raw_payload_policy": {
                "stores_raw_images": False,
                "stores_raw_schematics": False,
                "stores_raw_xml": False,
                "stores_tile_pixels": False,
                "stores_hashes_and_shapes_only": True,
            },
        },
        "pipeline": {
            "global_view": {
                "resolution": args.global_resolution,
                "preprocess": "RGB convert -> deterministic light augmentation -> aspect-preserving letterbox resize",
            },
            "tile_view": {
                "resolution": args.tile_resolution,
                "tiles_per_sample": args.tiles_per_sample,
                "policy": "CGHD ROI crops first, then grid fallback; open-schematics grid crops",
            },
            "structure_encoder": {
                "open_schematics": "schematic text sha1/length + component counts + metadata hash",
                "cghd": "XML sha1/length + symbol label counts",
            },
            "augmentation": {
                "seed_policy": "deterministic per sample_id",
                "operations": ["autocontrast_if_hash_even", "small_brightness_jitter", "small_contrast_jitter"],
            },
        },
        "batch": {
            "sample_ids_preview": batch["sample_ids"][:8],
            "global_views_shape": list(batch["global_views"].shape),
            "tile_views_shape": list(batch["tile_views"].shape),
            "global_tensor_dtype": str(batch["global_views"].dtype),
            "tile_tensor_dtype": str(batch["tile_views"].dtype),
            "tile_metadata_count": len(tile_metadata),
            "tile_metadata_preview": tile_metadata[:8],
            "structure_target_preview": structure_targets[:8],
        },
        "resources": {
            "rss_mb_before": rss_before_mb,
            "rss_mb_after": rss_after_mb,
            "rss_mb_delta": round(rss_after_mb - rss_before_mb, 3),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "ram_limit_mb": args.ram_limit_mb,
        },
        "pass_checks": pass_checks,
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "output": rel(args.output)}, ensure_ascii=False, sort_keys=True))
    return 0 if output["status"] == "closed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

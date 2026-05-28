from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "circuit_sources" / "source_manifest.json"


DATASETS: list[dict[str, Any]] = [
    {
        "dataset_id": "bshada/open-schematics",
        "priority": "primary",
        "roles": ["structure_world_model_pretraining", "image_to_structure_alignment"],
        "expected_fields": ["schematic", "image", "components_used", "json", "yaml", "name", "description", "type"],
        "m1_decision": "selected_for_structure_pretraining",
        "split_policy": "Use row references only in M1; M2 must build a usable image+schematic pair manifest.",
    },
    {
        "dataset_id": "microsoft/SchGen_dataset",
        "priority": "primary",
        "roles": ["circuit_structure_text_prior", "kicad_generation_prior", "teacher_curriculum_support"],
        "expected_fields": ["messages", "meta"],
        "m1_decision": "selected_for_structure_text_prior",
        "split_policy": "Use as structure/text prior only; do not mix into VQA evaluation.",
    },
    {
        "dataset_id": "ayoubkirouane/CircuitVQA",
        "priority": "primary",
        "roles": ["vqa_evaluation", "qwen_baseline_comparison", "end_to_end_system_evaluation"],
        "expected_fields": ["texts", "images"],
        "m1_decision": "selected_for_vqa_evaluation_with_license_caveat",
        "split_policy": "Use provided train/test split first; keep prompt/answer payload out of committed artifacts.",
    },
    {
        "dataset_id": "lowercaseonly/cghd",
        "priority": "primary",
        "roles": ["symbol_grounding_probe", "object_detection_probe", "segmentation_probe"],
        "expected_fields": ["images", "annotations", "instances", "segmentation", "classes"],
        "m1_decision": "selected_for_perception_probe",
        "split_policy": "File-tree dataset; M2 must pair image/XML by file stem and split by drafter+circuit group.",
    },
    {
        "dataset_id": "MirandaAbhilash/circuitvqa-dataset",
        "priority": "secondary",
        "roles": ["supplementary_image_baseline_candidate"],
        "expected_fields": ["image", "label"],
        "m1_decision": "optional_supplementary_image_candidate",
        "split_policy": "Use only after license and label semantics audit.",
    },
    {
        "dataset_id": "Ailiance-fr/kicad9plus-copyleft",
        "priority": "secondary_restricted",
        "roles": ["optional_kicad_text_corpus"],
        "expected_fields": ["messages", "metadata"],
        "m1_decision": "exclude_from_initial_training_due_to_copyleft_scope",
        "split_policy": "Keep out of initial training unless GPL/copyleft obligations are explicitly accepted.",
    },
    {
        "dataset_id": "hanky2397/schematic_images",
        "priority": "excluded_initially",
        "roles": ["possible_netlist_extraction_dataset"],
        "expected_fields": ["images.zip", "components.zip", "pkl.zip", "sp.zip"],
        "m1_decision": "exclude_initially_due_to_gated_access_and_missing_license",
        "split_policy": "Do not use until access terms and license are reviewed.",
    },
]


def fetch_json(url: str) -> tuple[dict[str, Any] | None, str | None]:
    request = urllib.request.Request(url, headers={"User-Agent": "circuit-lewm-routing-m1-audit/0.2"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8")), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def dataset_server(endpoint: str, dataset_id: str, extra: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, str | None]:
    params = {"dataset": dataset_id}
    if extra:
        params.update(extra)
    query = urllib.parse.urlencode(params)
    return fetch_json(f"https://datasets-server.huggingface.co/{endpoint}?{query}")


def size_h(num_bytes: int | None) -> str | None:
    if num_bytes is None:
        return None
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    index = 0
    while value >= 1024 and index < len(units) - 1:
        value /= 1024
        index += 1
    return f"{value:.2f} {units[index]}"


def sanitize_preview_value(key: str, value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if not isinstance(value, str):
        return f"<{type(value).__name__}>"
    text = value.strip()
    if re.search(r"(Expires=|Signature=|Key-Pair-Id|X-Amz-)", text):
        return "<signed_url_redacted>"
    key_lower = key.lower()
    looks_like_path = bool(re.search(r"(^[A-Za-z]:[\\/]|^/home/|^/users/|^/mnt/|^\\\\)", text, flags=re.I))
    key_suggests_path = key_lower.endswith("path") or key_lower.endswith("_path")
    if looks_like_path or (key_suggests_path and bool(re.search(r"[\\/]", text))):
        basename = re.split(r"[\\/]", text.rstrip("/\\"))[-1]
        return f"<path_redacted>/{basename}" if basename else "<path_redacted>"
    return text[:120]


def summarize_features(first_rows: dict[str, Any] | None) -> list[dict[str, str]]:
    return [
        {"name": item.get("name"), "type": json.dumps(item.get("type"), ensure_ascii=False, sort_keys=True)}
        for item in (first_rows or {}).get("features", [])
    ]


def summarize_first_row(first_rows: dict[str, Any] | None) -> dict[str, Any]:
    rows = (first_rows or {}).get("rows", [])
    if not rows:
        return {"row_keys": [], "field_shapes": {}}
    row = rows[0].get("row", {})
    shapes: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, list):
            shapes[key] = {"kind": "list", "length": len(value)}
        elif isinstance(value, dict):
            clean = {k: v for k, v in value.items() if k not in {"src", "bytes"}}
            shapes[key] = {
                "kind": "dict",
                "keys": sorted(value.keys()),
                "preview": {k: sanitize_preview_value(k, v) for k, v in list(clean.items())[:8]},
            }
        else:
            shapes[key] = {"kind": type(value).__name__, "preview": sanitize_preview_value(key, value)}
    return {"row_keys": list(row.keys()), "field_shapes": shapes}


def summarize_files(siblings: list[dict[str, Any]]) -> dict[str, Any]:
    filenames = [item.get("rfilename", "") for item in siblings]
    extensions = Counter((os.path.splitext(name)[1].lower() or "<none>") for name in filenames)
    top_dirs = Counter(name.split("/")[0] for name in filenames if name)
    return {
        "total_siblings": len(filenames),
        "extensions_top": extensions.most_common(20),
        "top_level_dirs_top": top_dirs.most_common(20),
        "image_file_count": sum(1 for name in filenames if re.search(r"\.(jpg|jpeg|png|webp)$", name, flags=re.I)),
        "xml_file_count": sum(1 for name in filenames if name.lower().endswith(".xml")),
        "json_file_count": sum(1 for name in filenames if name.lower().endswith(".json")),
        "parquet_file_count": sum(1 for name in filenames if name.lower().endswith(".parquet")),
        "zip_file_count": sum(1 for name in filenames if name.lower().endswith(".zip")),
    }


def download_footprint(size_payload: dict[str, Any] | None, parquet_payload: dict[str, Any] | None, file_inventory: dict[str, Any]) -> dict[str, Any]:
    size_root = (size_payload or {}).get("size") or {}
    dataset_size = size_root.get("dataset") or {}
    parquet_files = (parquet_payload or {}).get("parquet_files") or []
    parquet_bytes = sum(item.get("size") or 0 for item in parquet_files)
    memory_bytes = dataset_size.get("num_bytes_memory")
    row_count = dataset_size.get("num_rows")
    row_average_parquet = int(parquet_bytes / row_count) if parquet_bytes and row_count else None
    subset_5k_estimate = row_average_parquet * 5000 if row_average_parquet else None
    return {
        "viewer_num_rows": row_count,
        "viewer_original_bytes": dataset_size.get("num_bytes_original_files"),
        "viewer_parquet_bytes": dataset_size.get("num_bytes_parquet_files") or parquet_bytes or None,
        "viewer_memory_bytes": memory_bytes,
        "viewer_original_h": size_h(dataset_size.get("num_bytes_original_files")),
        "viewer_parquet_h": size_h(dataset_size.get("num_bytes_parquet_files") or parquet_bytes or None),
        "viewer_memory_h": size_h(memory_bytes),
        "parquet_file_count": len(parquet_files),
        "parquet_api_total_bytes": parquet_bytes or None,
        "parquet_api_total_h": size_h(parquet_bytes) if parquet_bytes else None,
        "avg_parquet_bytes_per_row": row_average_parquet,
        "estimated_5k_parquet_h": size_h(subset_5k_estimate) if subset_5k_estimate else None,
        "file_inventory_total_siblings": file_inventory.get("total_siblings"),
        "file_tree_size_caveat": "Hub file-tree byte size is not always exposed by cheap APIs; avoid full snapshot_download unless explicitly budgeted.",
    }


def recommendation(dataset_id: str, footprint: dict[str, Any], license_value: str | None, gated: Any) -> dict[str, Any]:
    if gated:
        return {"local_download": "avoid_initially", "reason": "dataset is gated or requires access review"}
    if dataset_id == "bshada/open-schematics":
        return {
            "local_download": "avoid_full_snapshot_for_m1",
            "reason": "full parquet is about 6.2 GiB and decoded/cache footprint may be much larger; build a 5k usable-pair cache in M2 instead",
        }
    if dataset_id == "lowercaseonly/cghd":
        return {
            "local_download": "partial_or_pattern_only",
            "reason": "file-tree dataset; pair image/XML first and avoid full snapshot until exact cache budget is known",
        }
    if license_value is None and "CircuitVQA" in dataset_id:
        return {
            "local_download": "metadata_or_eval_cache_only",
            "reason": "license is not declared on HF; keep raw payload out of committed artifacts",
        }
    parquet_bytes = footprint.get("viewer_parquet_bytes") or 0
    if parquet_bytes and parquet_bytes < 1024 * 1024 * 1024:
        return {"local_download": "acceptable_if_needed", "reason": "sub-1GiB parquet footprint"}
    return {"local_download": "review_before_download", "reason": "size or license requires explicit decision"}


def main() -> int:
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    entries: list[dict[str, Any]] = []
    for spec in DATASETS:
        dataset_id = spec["dataset_id"]
        hub, hub_error = fetch_json(f"https://huggingface.co/api/datasets/{dataset_id}")
        hub_full, hub_full_error = fetch_json(f"https://huggingface.co/api/datasets/{dataset_id}?full=true")
        size, size_error = dataset_server("size", dataset_id)
        splits, splits_error = dataset_server("splits", dataset_id)
        parquet, parquet_error = dataset_server("parquet", dataset_id)

        first_rows_by_split: dict[str, Any] = {}
        for split in (splits or {}).get("splits", [])[:3]:
            config_name = split.get("config")
            split_name = split.get("split")
            first_rows, first_rows_error = dataset_server(
                "first-rows", dataset_id, {"config": config_name, "split": split_name}
            )
            first_rows_by_split[f"{config_name}/{split_name}"] = {
                "error": first_rows_error,
                "features": summarize_features(first_rows),
                "sample": summarize_first_row(first_rows),
            }

        tags = (hub or {}).get("tags") or []
        license_value = (hub or {}).get("cardData", {}).get("license") or (hub or {}).get("license")
        if not license_value:
            for tag in tags:
                if isinstance(tag, str) and tag.startswith("license:"):
                    license_value = tag.split(":", 1)[1]
                    break
        file_inventory = summarize_files((hub_full or {}).get("siblings", []))
        footprint = download_footprint(size, parquet, file_inventory)

        entries.append(
            {
                **spec,
                "audit_date_utc": generated_at,
                "source_url": f"https://huggingface.co/datasets/{dataset_id}",
                "hub_api_url": f"https://huggingface.co/api/datasets/{dataset_id}",
                "hub": {
                    "ok": hub is not None,
                    "error": hub_error,
                    "id": (hub or {}).get("id"),
                    "private": (hub or {}).get("private"),
                    "gated": (hub or {}).get("gated"),
                    "license": license_value,
                    "last_modified_utc": (hub or {}).get("lastModified"),
                    "downloads": (hub or {}).get("downloads"),
                    "likes": (hub or {}).get("likes"),
                    "tags": tags,
                    "task_categories": (hub or {}).get("cardData", {}).get("task_categories"),
                    "size_categories": (hub or {}).get("cardData", {}).get("size_categories"),
                },
                "dataset_server": {
                    "size_ok": size is not None,
                    "size_error": size_error,
                    "size": (size or {}).get("size"),
                    "splits_ok": splits is not None,
                    "splits_error": splits_error,
                    "splits": (splits or {}).get("splits", []),
                    "parquet_ok": parquet is not None,
                    "parquet_error": parquet_error,
                    "parquet_file_count": len((parquet or {}).get("parquet_files", [])),
                    "first_rows_by_split": first_rows_by_split,
                },
                "file_inventory": file_inventory,
                "download_footprint": footprint,
                "local_download_recommendation": recommendation(dataset_id, footprint, license_value, (hub or {}).get("gated")),
                "api_caveats": {
                    "hub_full_error": hub_full_error,
                    "dataset_viewer_may_500_on_large_row_pages": dataset_id == "bshada/open-schematics",
                    "file_tree_size_may_be_unknown": dataset_id in {"lowercaseonly/cghd", "MirandaAbhilash/circuitvqa-dataset"},
                },
            }
        )

    payload = {
        "manifest_version": 1,
        "generated_at_utc": generated_at,
        "audit_scope": "M1 dataset source audit for circuit-domain Perception LeWM training and routing evaluation",
        "source_count": len(entries),
        "entries": entries,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} with {len(entries)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

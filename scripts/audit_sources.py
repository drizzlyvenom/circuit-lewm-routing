from __future__ import annotations

import json
import os
import re
import sys
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "circuit_sources" / "source_manifest.json"


SOURCE_PLAN: list[dict[str, Any]] = [
    {
        "dataset_id": "bshada/open-schematics",
        "priority": "primary",
        "m1_decision": "selected_for_structure_pretraining",
        "roles": ["structure_world_model_pretraining", "image_to_structure_alignment"],
        "expected_fields": ["schematic", "image", "components_used", "json", "yaml", "name", "description", "type"],
        "split_policy": "Do not use random row split for final experiments if project/name near-duplicates are detected; prefer project/name grouped split.",
    },
    {
        "dataset_id": "microsoft/SchGen_dataset",
        "priority": "primary",
        "m1_decision": "selected_for_structure_text_prior",
        "roles": ["circuit_structure_text_prior", "kicad_generation_prior", "teacher_curriculum_support"],
        "expected_fields": ["messages", "meta"],
        "split_policy": "Use as text/structure prior only; do not mix train rows into CircuitVQA answer evaluation.",
    },
    {
        "dataset_id": "ayoubkirouane/CircuitVQA",
        "priority": "primary_conditional",
        "m1_decision": "selected_for_vqa_evaluation_with_license_caveat",
        "roles": ["vqa_evaluation", "qwen_baseline_comparison", "end_to_end_accuracy_benchmark"],
        "expected_fields": ["texts", "images"],
        "split_policy": "Use provided train/test split first; deduplicate repeated image ids across splits before paper evidence.",
    },
    {
        "dataset_id": "lowercaseonly/cghd",
        "priority": "primary",
        "m1_decision": "selected_for_perception_probe",
        "roles": ["symbol_grounding_probe", "object_detection_probe", "segmentation_probe", "handwritten_robustness_probe"],
        "expected_fields": ["images", "annotations", "instances", "segmentation", "classes"],
        "split_policy": "Dataset is file-tree based; create explicit grouped train/holdout/test manifest after pairing images and annotations.",
    },
    {
        "dataset_id": "MirandaAbhilash/circuitvqa-dataset",
        "priority": "secondary",
        "m1_decision": "optional_supplementary_image_candidate",
        "roles": ["supplementary_image_baseline_candidate"],
        "expected_fields": ["image", "label"],
        "split_policy": "Use only after license clarification and label semantics audit; current HF preview labels are not enough for VQA supervision.",
    },
    {
        "dataset_id": "Ailiance-fr/kicad9plus-copyleft",
        "priority": "secondary_restricted",
        "m1_decision": "exclude_from_initial_training_due_to_copyleft_scope",
        "roles": ["optional_kicad_text_corpus"],
        "expected_fields": ["messages", "metadata"],
        "split_policy": "Keep out of initial training unless GPL-3.0-or-later obligations are explicitly accepted.",
    },
    {
        "dataset_id": "hanky2397/schematic_images",
        "priority": "excluded_initially",
        "m1_decision": "exclude_initially_due_to_gated_access_and_missing_license",
        "roles": ["possible_netlist_extraction_dataset"],
        "expected_fields": ["images.zip", "components.zip", "pkl.zip", "sp.zip"],
        "split_policy": "Do not use until access terms and license are reviewed.",
    },
]


def fetch_json(url: str) -> tuple[dict[str, Any] | None, str | None]:
    request = urllib.request.Request(url, headers={"User-Agent": "circuit-lewm-routing-audit/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return payload, None
    except Exception as exc:  # network/API failures should be recorded, not fatal
        return None, str(exc)


def dataset_server(endpoint: str, params: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    query = urllib.parse.urlencode(params)
    return fetch_json(f"https://datasets-server.huggingface.co/{endpoint}?{query}")


def summarize_features(first_rows: dict[str, Any] | None) -> list[dict[str, str]]:
    features = []
    for feature in (first_rows or {}).get("features", []):
        name = feature.get("name")
        feature_type = feature.get("type")
        features.append({"name": name, "type": json.dumps(feature_type, ensure_ascii=False, sort_keys=True)})
    return features


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
            clean_dict = {k: v for k, v in value.items() if k not in {"src", "bytes"}}
            preview = {
                nested_key: sanitize_preview_value(nested_key, nested_value)
                for nested_key, nested_value in list(clean_dict.items())[:8]
            }
            shapes[key] = {"kind": "dict", "keys": sorted(value.keys()), "preview": preview}
        else:
            shapes[key] = {"kind": type(value).__name__, "preview": sanitize_preview_value(key, value)}
    return {"row_keys": list(row.keys()), "field_shapes": shapes}


def summarize_files(siblings: list[dict[str, Any]]) -> dict[str, Any]:
    filenames = [s.get("rfilename", "") for s in siblings]
    extensions = Counter((os.path.splitext(name)[1].lower() or "<none>") for name in filenames)
    top_dirs = Counter(name.split("/")[0] for name in filenames if name)
    image_files = sum(1 for name in filenames if re.search(r"\.(jpg|jpeg|png|webp)$", name, flags=re.I))
    xml_files = sum(1 for name in filenames if name.lower().endswith(".xml"))
    json_files = sum(1 for name in filenames if name.lower().endswith(".json"))
    parquet_files = sum(1 for name in filenames if name.lower().endswith(".parquet"))
    zip_files = sum(1 for name in filenames if name.lower().endswith(".zip"))
    return {
        "total_siblings": len(filenames),
        "extensions_top": extensions.most_common(20),
        "top_level_dirs_top": top_dirs.most_common(20),
        "image_file_count": image_files,
        "xml_file_count": xml_files,
        "json_file_count": json_files,
        "parquet_file_count": parquet_files,
        "zip_file_count": zip_files,
    }


def compact_size(size_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not size_payload or "size" not in size_payload:
        return None
    size = size_payload["size"]
    return {
        "dataset": size.get("dataset"),
        "configs": size.get("configs"),
        "splits": size.get("splits"),
        "partial": size_payload.get("partial"),
        "failed": size_payload.get("failed"),
    }


def main() -> int:
    audit_date = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    entries: list[dict[str, Any]] = []

    for plan in SOURCE_PLAN:
        dataset_id = plan["dataset_id"]
        hub, hub_error = fetch_json(f"https://huggingface.co/api/datasets/{dataset_id}?full=true")
        card_data = (hub or {}).get("cardData") or {}
        tags = (hub or {}).get("tags") or []
        siblings = (hub or {}).get("siblings") or []

        splits, splits_error = dataset_server("splits", {"dataset": dataset_id})
        size, size_error = dataset_server("size", {"dataset": dataset_id})
        parquet, parquet_error = dataset_server("parquet", {"dataset": dataset_id})

        first_rows_by_split: dict[str, Any] = {}
        for split in (splits or {}).get("splits", [])[:3]:
            config_name = split.get("config")
            split_name = split.get("split")
            first_rows, first_rows_error = dataset_server(
                "first-rows",
                {"dataset": dataset_id, "config": config_name, "split": split_name},
            )
            first_rows_by_split[f"{config_name}/{split_name}"] = {
                "error": first_rows_error,
                "features": summarize_features(first_rows),
                "sample": summarize_first_row(first_rows),
            }

        license_value = card_data.get("license")
        if not license_value:
            for tag in tags:
                if tag.startswith("license:"):
                    license_value = tag.split(":", 1)[1]
                    break

        entry = {
            **plan,
            "audit_date_utc": audit_date,
            "source_url": f"https://huggingface.co/datasets/{dataset_id}",
            "hub_api_url": f"https://huggingface.co/api/datasets/{dataset_id}",
            "dataset_server_urls": {
                "splits": f"https://datasets-server.huggingface.co/splits?dataset={urllib.parse.quote(dataset_id, safe='')}",
                "size": f"https://datasets-server.huggingface.co/size?dataset={urllib.parse.quote(dataset_id, safe='')}",
                "parquet": f"https://datasets-server.huggingface.co/parquet?dataset={urllib.parse.quote(dataset_id, safe='')}",
            },
            "hub": {
                "ok": hub is not None,
                "error": hub_error,
                "id": (hub or {}).get("id"),
                "last_modified_utc": (hub or {}).get("lastModified"),
                "gated": (hub or {}).get("gated"),
                "private": (hub or {}).get("private"),
                "downloads": (hub or {}).get("downloads"),
                "likes": (hub or {}).get("likes"),
                "tags": tags,
                "license": license_value,
                "card_license": card_data.get("license"),
                "task_categories": card_data.get("task_categories"),
                "size_categories": card_data.get("size_categories"),
            },
            "dataset_server": {
                "splits_ok": splits is not None,
                "splits_error": splits_error,
                "splits": (splits or {}).get("splits", []),
                "size_ok": size is not None,
                "size_error": size_error,
                "size": compact_size(size),
                "parquet_ok": parquet is not None,
                "parquet_error": parquet_error,
                "parquet_file_count": len((parquet or {}).get("parquet_files", [])),
                "first_rows_by_split": first_rows_by_split,
            },
            "file_inventory": summarize_files(siblings),
        }
        entries.append(entry)

    manifest = {
        "manifest_version": 1,
        "generated_at_utc": audit_date,
        "audit_scope": "M1 dataset source audit for circuit-domain Perception LeWM + routing",
        "source_count": len(entries),
        "entries": entries,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)} with {len(entries)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

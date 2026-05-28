from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "circuit_curricula"


TARGETS = {
    "bshada/open-schematics": {"train": 2048, "holdout": 512, "test": 512},
    "microsoft/SchGen_dataset": {"train": 1024, "holdout": 128, "test": 128},
    "ayoubkirouane/CircuitVQA": {"train": 4096, "holdout": 1024, "test": 1024},
    "lowercaseonly/cghd": {"train": 2048, "holdout": 512, "test": 512},
}


SOURCE_TOTALS = {
    ("bshada/open-schematics", "train"): 84470,
    ("microsoft/SchGen_dataset", "train"): 8420,
    ("ayoubkirouane/CircuitVQA", "train"): 8376,
    ("ayoubkirouane/CircuitVQA", "test"): 2094,
}


CIRCUITVQA_QA_PER_ROW = 6


LICENSES = {
    "bshada/open-schematics": "cc-by-4.0",
    "microsoft/SchGen_dataset": "mit",
    "ayoubkirouane/CircuitVQA": None,
    "lowercaseonly/cghd": "cc-by-3.0",
}


def stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:12], 16)


def grouped_split(group_key: str) -> str:
    bucket = stable_int(group_key) % 10
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "holdout"
    return "train"


def fetch_json(url: str) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(5):
        request = urllib.request.Request(url, headers={"User-Agent": "circuit-lewm-routing-split/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except (TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as exc:
            last_error = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in {429, 500, 502, 503, 504}:
                raise
            if attempt == 4:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"failed to fetch json after retries: {url}") from last_error


def dataset_rows(dataset: str, split: str, *, config: str = "default", length: int = 10):
    offset = 0
    page_length = length
    total = None
    while True:
        query = urllib.parse.urlencode(
            {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": page_length}
        )
        try:
            payload = fetch_json(f"https://datasets-server.huggingface.co/rows?{query}")
        except RuntimeError:
            if page_length > 1:
                page_length = max(1, page_length // 2)
                continue
            print(f"warning: skipping unreadable HF viewer row {dataset}/{split}/{offset}")
            offset += 1
            page_length = length
            continue
        rows = payload.get("rows", [])
        if not rows:
            break
        for item in rows:
            yield item
        offset += len(rows)
        total = payload.get("num_rows_total") if total is None else total
        if total is not None and offset >= total:
            break
        page_length = length


def source_row_url(dataset: str, split: str, row_idx: int, *, config: str = "default") -> str:
    return f"https://huggingface.co/datasets/{dataset}/viewer/{config}/{split}?row={row_idx}"


def provenance(dataset: str) -> dict[str, Any]:
    license_value = LICENSES[dataset]
    caveat = None
    if dataset == "ayoubkirouane/CircuitVQA":
        caveat = "HF dataset card does not declare a license; manifest stores row/question refs only."
    return {
        "source_url": f"https://huggingface.co/datasets/{dataset}",
        "license": license_value,
        "license_caveat": caveat,
    }


def base_sample(
    *,
    sample_id: str,
    family: str,
    dataset: str,
    split: str,
    source: dict[str, Any],
    image: dict[str, Any],
    query: dict[str, Any],
    structure: dict[str, Any],
    taxonomy: dict[str, Any],
    source_group_key: str,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "sample_family": family,
        "source_dataset": dataset,
        "source_group_key": source_group_key,
        "split": split,
        "source": source,
        "image": image,
        "query": query,
        "structure": structure,
        "supervision": {
            "taxonomy": taxonomy,
            "hard_negatives": [],
            "roi_or_tile_boxes": None,
        },
        "provenance": provenance(dataset),
    }


def append_if_needed(buckets: dict[str, list[dict[str, Any]]], sample: dict[str, Any]) -> bool:
    dataset = sample["source_dataset"]
    split = sample["split"]
    if len(buckets[split]) >= TARGETS[dataset][split]:
        return False
    buckets[split].append(sample)
    return True


def done_for_dataset(buckets: dict[str, list[dict[str, Any]]], dataset: str) -> bool:
    return all(len(buckets[split]) >= TARGETS[dataset][split] for split in ("train", "holdout", "test"))


def hashed_indices(namespace: str, total: int) -> list[int]:
    return sorted(range(total), key=lambda row_idx: stable_int(f"{namespace}:{row_idx}"))


def split_indices(dataset: str, source_split: str = "train") -> dict[str, list[int]]:
    total = SOURCE_TOTALS[(dataset, source_split)]
    ordered = hashed_indices(f"{dataset}:{source_split}", total)
    buckets: dict[str, list[int]] = {}
    cursor = 0
    for split in ("train", "holdout", "test"):
        count = TARGETS[dataset][split]
        buckets[split] = ordered[cursor : cursor + count]
        cursor += count
    return buckets


def build_open_schematics() -> dict[str, list[dict[str, Any]]]:
    dataset = "bshada/open-schematics"
    buckets: dict[str, list[dict[str, Any]]] = {"train": [], "holdout": [], "test": []}
    for split, row_indices in split_indices(dataset).items():
        for row_idx in row_indices:
            group_key = f"row:{row_idx}"
            sample = base_sample(
                sample_id=f"openschematic_{split}_{len(buckets[split]):06d}",
                family="structure_pretraining",
                dataset=dataset,
                split=split,
                source_group_key=group_key,
                source={
                    "dataset": dataset,
                    "config": "default",
                    "source_split": "train",
                    "row_idx": row_idx,
                    "row_url": source_row_url(dataset, "train", row_idx),
                    "fields_used": [
                        "schematic",
                        "image",
                        "components_used",
                        "json",
                        "yaml",
                        "name",
                        "description",
                        "type",
                    ],
                },
                image={
                    "image_path_or_ref": f"hf://datasets/{dataset}/default/train/{row_idx}/image",
                    "width": None,
                    "height": None,
                },
                query={"prompt": None, "expected_answers": None, "answer_type": None},
                structure={
                    "structure_path_or_ref": f"hf://datasets/{dataset}/default/train/{row_idx}/schematic",
                    "component_list_ref": f"hf://datasets/{dataset}/default/train/{row_idx}/components_used",
                    "netlist_or_edges_ref": None,
                    "metadata_text": f"source_row_idx={row_idx}",
                    "json_ref": f"hf://datasets/{dataset}/default/train/{row_idx}/json",
                    "yaml_ref": f"hf://datasets/{dataset}/default/train/{row_idx}/yaml",
                },
                taxonomy={
                    "domain": "circuit",
                    "evidence_type": "schematic_structure",
                    "operation": "image_to_structure_alignment",
                    "failure_mode": "structure_mismatch",
                },
            )
            append_if_needed(buckets, sample)
    return buckets


def build_schgen() -> dict[str, list[dict[str, Any]]]:
    dataset = "microsoft/SchGen_dataset"
    buckets: dict[str, list[dict[str, Any]]] = {"train": [], "holdout": [], "test": []}
    for split, row_indices in split_indices(dataset).items():
        for row_idx in row_indices:
            group_key = f"row:{row_idx}"
            sample = base_sample(
                sample_id=f"schgen_{split}_{len(buckets[split]):06d}",
                family="structure_text_prior",
                dataset=dataset,
                split=split,
                source_group_key=group_key,
                source={
                    "dataset": dataset,
                    "config": "default",
                    "source_split": "train",
                    "row_idx": row_idx,
                    "row_url": source_row_url(dataset, "train", row_idx),
                    "fields_used": ["messages", "meta"],
                },
                image={"image_path_or_ref": None, "width": None, "height": None},
                query={
                    "prompt": None,
                    "prompt_ref": f"hf://datasets/{dataset}/default/train/{row_idx}/messages",
                    "expected_answers": None,
                    "answer_type": "kicad_generation_text",
                },
                structure={
                    "structure_path_or_ref": f"hf://datasets/{dataset}/default/train/{row_idx}/meta/schematic",
                    "component_list_ref": None,
                    "netlist_or_edges_ref": None,
                    "metadata_text": f"source_row_idx={row_idx}",
                    "json_ref": None,
                    "yaml_ref": None,
                },
                taxonomy={
                    "domain": "circuit",
                    "evidence_type": "kicad_text_structure",
                    "operation": "structure_generation_prior",
                    "failure_mode": "invalid_or_misaligned_kicad",
                },
            )
            append_if_needed(buckets, sample)
    return buckets


def extract_answer_type(text: str) -> str:
    answer_match = re.search(r"Answer:\s*(.+)", text or "", flags=re.I | re.S)
    answer = answer_match.group(1).strip() if answer_match else text or ""
    if re.fullmatch(r"[-+]?\d+(\.\d+)?\s*([A-Za-z%]+)?", answer):
        return "numeric_or_unit"
    if "," in answer:
        return "list_or_multi_entity"
    return "free_text"


def circuitvqa_samples_for_rows(source_split: str, target_split: str, row_indices: list[int], target_count: int) -> list[dict[str, Any]]:
    dataset = "ayoubkirouane/CircuitVQA"
    rows: list[dict[str, Any]] = []
    for row_idx in row_indices:
        for q_idx in range(CIRCUITVQA_QA_PER_ROW):
            if len(rows) >= target_count:
                break
            sample = base_sample(
                sample_id=f"circuitvqa_{target_split}_{len(rows):06d}",
                family="vqa_evaluation",
                dataset=dataset,
                split=target_split,
                source_group_key=f"{source_split}:{row_idx}",
                source={
                    "dataset": dataset,
                    "config": "default",
                    "source_split": source_split,
                    "row_idx": row_idx,
                    "question_index": q_idx,
                    "row_url": source_row_url(dataset, source_split, row_idx),
                    "fields_used": ["texts", "images"],
                },
                image={
                    "image_path_or_ref": f"hf://datasets/{dataset}/default/{source_split}/{row_idx}/images/0",
                    "width": None,
                    "height": None,
                    "image_count": 1,
                },
                query={
                    "prompt": None,
                    "prompt_ref": f"hf://datasets/{dataset}/default/{source_split}/{row_idx}/texts/{q_idx}/user",
                    "expected_answers": None,
                    "expected_answer_ref": f"hf://datasets/{dataset}/default/{source_split}/{row_idx}/texts/{q_idx}/assistant",
                    "answer_type": "referenced_circuit_vqa_answer",
                },
                structure={
                    "structure_path_or_ref": None,
                    "component_list_ref": None,
                    "netlist_or_edges_ref": None,
                    "metadata_text": None,
                    "json_ref": None,
                    "yaml_ref": None,
                },
                taxonomy={
                    "domain": "circuit",
                    "evidence_type": "diagram_qa",
                    "operation": "visual_question_answering",
                    "failure_mode": "wrong_circuit_evidence_or_answer",
                },
            )
            rows.append(sample)
        if len(rows) >= target_count:
            break
    return rows


def build_circuitvqa() -> dict[str, list[dict[str, Any]]]:
    dataset = "ayoubkirouane/CircuitVQA"
    train_row_count = (TARGETS[dataset]["train"] + CIRCUITVQA_QA_PER_ROW - 1) // CIRCUITVQA_QA_PER_ROW
    train_rows = hashed_indices(f"{dataset}:train", SOURCE_TOTALS[(dataset, "train")])[:train_row_count]

    test_rows_ordered = hashed_indices(f"{dataset}:test", SOURCE_TOTALS[(dataset, "test")])
    holdout_row_count = (TARGETS[dataset]["holdout"] + CIRCUITVQA_QA_PER_ROW - 1) // CIRCUITVQA_QA_PER_ROW
    test_row_count = (TARGETS[dataset]["test"] + CIRCUITVQA_QA_PER_ROW - 1) // CIRCUITVQA_QA_PER_ROW
    holdout_rows = test_rows_ordered[:holdout_row_count]
    test_rows = test_rows_ordered[holdout_row_count : holdout_row_count + test_row_count]

    return {
        "train": circuitvqa_samples_for_rows("train", "train", train_rows, TARGETS[dataset]["train"]),
        "holdout": circuitvqa_samples_for_rows("test", "holdout", holdout_rows, TARGETS[dataset]["holdout"]),
        "test": circuitvqa_samples_for_rows("test", "test", test_rows, TARGETS[dataset]["test"]),
    }


def stem_group(filename: str) -> str:
    stem = Path(filename).stem
    match = re.match(r"(.+)_P\d+$", stem)
    return match.group(1) if match else stem


def build_cghd() -> dict[str, list[dict[str, Any]]]:
    dataset = "lowercaseonly/cghd"
    hub = fetch_json(f"https://huggingface.co/api/datasets/{dataset}?full=true")
    siblings = [s.get("rfilename", "") for s in hub.get("siblings", [])]
    images = [name for name in siblings if re.search(r"/images/.+\.(jpg|jpeg|png)$", name, flags=re.I)]
    annotations = {Path(name).stem: name for name in siblings if "/annotations/" in name and name.endswith(".xml")}
    instances = {Path(name).stem: name for name in siblings if "/instances/" in name and name.endswith(".json")}
    segmentation = {
        Path(name).stem: name
        for name in siblings
        if "/segmentation/" in name and re.search(r"\.(jpg|jpeg|png)$", name, flags=re.I)
    }
    buckets: dict[str, list[dict[str, Any]]] = {"train": [], "holdout": [], "test": []}
    grouped_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image_path in sorted(images):
        stem = Path(image_path).stem
        if stem not in annotations:
            continue
        group_key = f"{image_path.split('/')[0]}:{stem_group(stem)}"
        grouped_candidates[group_key].append(
            {
                "group_key": group_key,
                "image_path": image_path,
                "stem": stem,
                "annotation_xml": annotations.get(stem),
                "instance_json": instances.get(stem),
                "segmentation": segmentation.get(stem),
            }
        )

    used_groups: set[str] = set()
    ordered_groups = sorted(grouped_candidates.items(), key=lambda item: stable_int(f"{dataset}:{item[0]}"))
    for split in ("train", "holdout", "test"):
        for group_key, candidates in ordered_groups:
            if group_key in used_groups:
                continue
            for candidate in candidates:
                if len(buckets[split]) >= TARGETS[dataset][split]:
                    break
                image_path = candidate["image_path"]
                stem = candidate["stem"]
                sample = base_sample(
                    sample_id=f"cghd_{split}_{len(buckets[split]):06d}",
                    family="perception_probe",
                    dataset=dataset,
                    split=split,
                    source_group_key=group_key,
                    source={
                        "dataset": dataset,
                        "config": "file_tree",
                        "source_split": "train",
                        "row_idx": None,
                        "file_paths": {
                            "image": image_path,
                            "annotation_xml": candidate["annotation_xml"],
                            "instance_json": candidate["instance_json"],
                            "segmentation": candidate["segmentation"],
                        },
                        "fields_used": ["images", "annotations", "instances", "segmentation"],
                    },
                    image={
                        "image_path_or_ref": f"hf://datasets/{dataset}/{image_path}",
                        "width": None,
                        "height": None,
                    },
                    query={"prompt": None, "expected_answers": None, "answer_type": None},
                    structure={
                        "structure_path_or_ref": f"hf://datasets/{dataset}/{candidate['annotation_xml']}",
                        "component_list_ref": "hf://datasets/lowercaseonly/cghd/classes.json",
                        "netlist_or_edges_ref": None,
                        "metadata_text": f"drafter={image_path.split('/')[0]}; image_stem={stem}",
                        "json_ref": f"hf://datasets/{dataset}/{candidate['instance_json']}"
                        if candidate["instance_json"]
                        else None,
                        "yaml_ref": None,
                    },
                    taxonomy={
                        "domain": "circuit",
                        "evidence_type": "handwritten_symbol_grounding",
                        "operation": "detect_or_segment",
                        "failure_mode": "missed_symbol_or_stroke",
                    },
                )
                append_if_needed(buckets, sample)
            used_groups.add(group_key)
            if len(buckets[split]) >= TARGETS[dataset][split]:
                break
    return buckets


def merge_buckets(all_buckets: list[dict[str, list[dict[str, Any]]]]) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = {"train": [], "holdout": [], "test": []}
    for buckets in all_buckets:
        for split in merged:
            merged[split].extend(buckets[split])
    for split, rows in merged.items():
        for index, row in enumerate(rows):
            row["manifest_index"] = index
    return merged


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summarize(merged: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    by_split_source: dict[str, dict[str, int]] = {}
    by_family: dict[str, dict[str, int]] = {}
    for split, rows in merged.items():
        by_split_source[split] = dict(sorted(Counter(row["source_dataset"] for row in rows).items()))
        by_family[split] = dict(sorted(Counter(row["sample_family"] for row in rows).items()))
    return {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "manifest_version": 1,
        "split_files": {
            "train": "data/circuit_curricula/train.jsonl",
            "holdout": "data/circuit_curricula/holdout.jsonl",
            "test": "data/circuit_curricula/test.jsonl",
        },
        "targets": TARGETS,
        "counts_by_split_and_source": by_split_source,
        "counts_by_split_and_family": by_family,
        "total_rows_by_split": {split: len(rows) for split, rows in merged.items()},
        "split_policy": {
            "open_schematics": "deterministic row-reference split over audited train rows; project/name duplicate audit remains required before paper-ready training",
            "schgen_dataset": "deterministic row-reference split over audited train rows; module duplicate audit remains required before paper-ready training",
            "circuitvqa": "provided train split for train; provided test split divided by image row references into holdout/test",
            "cghd": "drafter+circuit grouped deterministic split over paired image/xml files",
        },
        "raw_payload_policy": "Manifests store HF row/file references and provenance, not signed asset URLs or raw dataset payloads.",
    }


def main() -> int:
    print("building open-schematics references")
    open_schematics = build_open_schematics()
    print("building SchGen references")
    schgen = build_schgen()
    print("building CircuitVQA references")
    circuitvqa = build_circuitvqa()
    print("building CGHD references")
    cghd = build_cghd()
    merged = merge_buckets([open_schematics, schgen, circuitvqa, cghd])

    for split, rows in merged.items():
        write_jsonl(OUT_DIR / f"{split}.jsonl", rows)
    summary = summarize(merged)
    write_json(OUT_DIR / "split_summary.json", summary)
    print(json.dumps(summary["total_rows_by_split"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

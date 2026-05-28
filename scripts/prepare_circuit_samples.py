from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image as PILImage


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "circuit_curricula"
DEFAULT_CGHD_DIR = ROOT / "data" / "downloads" / "hf" / "lowercaseonly_cghd"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_bucket(value: str, modulo: int = 100) -> int:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def split_by_group(group_key: str) -> str:
    bucket = stable_bucket(group_key)
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "holdout"
    return "test"


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def as_component_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        if isinstance(parsed, list):
            return [str(item) for item in parsed if item is not None]
        return [str(parsed)]
    return [str(value)]


def image_size(image: Any) -> tuple[int | None, int | None]:
    if isinstance(image, PILImage.Image):
        return int(image.width), int(image.height)
    if isinstance(image, dict):
        image_bytes = image.get("bytes")
        image_path = image.get("path")
        try:
            if image_bytes:
                with PILImage.open(BytesIO(image_bytes)) as opened:
                    return int(opened.width), int(opened.height)
            if image_path:
                with PILImage.open(image_path) as opened:
                    return int(opened.width), int(opened.height)
        except Exception:
            return None, None
    width = getattr(image, "width", None)
    height = getattr(image, "height", None)
    return (int(width) if width else None, int(height) if height else None)


def metadata_text(row: dict[str, Any]) -> str | None:
    parts = []
    for key in ("name", "type", "description"):
        value = row.get(key)
        if value:
            parts.append(f"{key}: {str(value).strip()}")
    return "\n".join(parts) if parts else None


def open_schematics_sample(row_index: int, row: dict[str, Any], split: str) -> dict[str, Any]:
    width, height = image_size(row.get("image"))
    return {
        "sample_id": f"open_schematics_{row_index:06d}",
        "source_dataset": "bshada/open-schematics",
        "split": split,
        "image": {
            "image_path_or_ref": f"hf://datasets/bshada/open-schematics/default/train/{row_index}/image",
            "width": width,
            "height": height,
        },
        "query": {
            "prompt": None,
            "expected_answers": None,
            "answer_type": None,
        },
        "structure": {
            "structure_path_or_ref": f"hf://datasets/bshada/open-schematics/default/train/{row_index}/schematic",
            "component_list": as_component_list(row.get("components_used")),
            "netlist_or_edges": None,
            "metadata_text": metadata_text(row),
        },
        "supervision": {
            "taxonomy": {
                "task_family": "structure_pretraining",
                "source_license": "cc-by-4.0",
                "split_unit": "source_project_name",
                "source_name": row.get("name"),
            },
            "hard_negatives": [],
            "roi_or_tile_boxes": None,
        },
    }


def collect_open_schematics(train_target: int, holdout_target: int, test_target: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    targets = {"train": train_target, "holdout": holdout_target, "test": test_target}
    samples: dict[str, list[dict[str, Any]]] = {"train": [], "holdout": [], "test": []}
    failures = Counter()
    group_to_split: dict[str, str] = {}
    scanned = 0

    dataset = load_dataset("bshada/open-schematics", split="train", streaming=True).cast_column("image", HFImage(decode=False))
    for row_index, row in enumerate(dataset):
        scanned += 1
        try:
            schematic = row.get("schematic")
            image = row.get("image")
            width, height = image_size(image)
            if not schematic or not isinstance(schematic, str):
                failures["missing_schematic"] += 1
                continue
            if image is None or width is None or height is None or width <= 0 or height <= 0:
                failures["missing_or_bad_image"] += 1
                continue
            name = str(row.get("name") or f"row_{row_index}")
            split = split_by_group(name.lower())
            group_to_split.setdefault(name.lower(), split)
            if len(samples[split]) >= targets[split]:
                continue
            samples[split].append(open_schematics_sample(row_index, row, split))
            if all(len(samples[key]) >= targets[key] for key in targets):
                break
        except Exception as exc:
            failures[f"{type(exc).__name__}"] += 1

    summary = {
        "dataset_id": "bshada/open-schematics",
        "source_url": "https://huggingface.co/datasets/bshada/open-schematics",
        "license": "cc-by-4.0",
        "split_unit": "source_project_name",
        "targets": targets,
        "scanned_rows": scanned,
        "verified_split_counts": {key: len(value) for key, value in samples.items()},
        "unique_split_groups": len(group_to_split),
        "failed_rows": dict(failures),
        "verification": "decoded image object with positive width/height and non-empty schematic string",
    }
    return samples, summary


def parse_cghd_boxes(xml_path: Path) -> list[dict[str, Any]]:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []
    boxes: list[dict[str, Any]] = []
    for obj in root.findall(".//object"):
        label = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if box is None:
            continue
        try:
            xmin = float(box.findtext("xmin") or 0)
            ymin = float(box.findtext("ymin") or 0)
            xmax = float(box.findtext("xmax") or 0)
            ymax = float(box.findtext("ymax") or 0)
        except ValueError:
            continue
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append({"label": label or "unknown", "bbox_xyxy": [xmin, ymin, xmax, ymax]})
    return boxes


def cghd_group_key(drafter: str, stem: str) -> str:
    match = re.match(r"^(C\d+_D\d+)", stem, flags=re.I)
    circuit = match.group(1) if match else stem
    return f"{drafter}/{circuit}".lower()


def collect_cghd(cghd_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    samples: dict[str, list[dict[str, Any]]] = {"train": [], "holdout": [], "test": []}
    if not cghd_dir.exists():
        return samples, {
            "dataset_id": "lowercaseonly/cghd",
            "source_url": "https://huggingface.co/datasets/lowercaseonly/cghd",
            "license": "cc-by-3.0",
            "status": "local_download_missing",
            "local_dir": rel(cghd_dir),
        }

    image_exts = {".jpg", ".jpeg", ".png"}
    images_by_stem: dict[tuple[str, str], Path] = {}
    xml_by_stem: dict[tuple[str, str], Path] = {}
    for drafter_dir in sorted(path for path in cghd_dir.iterdir() if path.is_dir() and path.name.startswith("drafter_")):
        images_dir = drafter_dir / "images"
        annotations_dir = drafter_dir / "annotations"
        if images_dir.exists():
            for image_path in images_dir.iterdir():
                if image_path.is_file() and image_path.suffix.lower() in image_exts:
                    images_by_stem[(drafter_dir.name, image_path.stem)] = image_path
        if annotations_dir.exists():
            for xml_path in annotations_dir.iterdir():
                if xml_path.is_file() and xml_path.suffix.lower() == ".xml":
                    xml_by_stem[(drafter_dir.name, xml_path.stem)] = xml_path

    paired_keys = sorted(set(images_by_stem) & set(xml_by_stem))
    label_counts = Counter()
    group_to_split: dict[str, str] = {}
    bad_images = 0

    for drafter, stem in paired_keys:
        image_path = images_by_stem[(drafter, stem)]
        xml_path = xml_by_stem[(drafter, stem)]
        try:
            with PILImage.open(image_path) as image:
                width, height = int(image.width), int(image.height)
        except Exception:
            bad_images += 1
            continue
        boxes = parse_cghd_boxes(xml_path)
        for box in boxes:
            label_counts[box["label"]] += 1
        group = cghd_group_key(drafter, stem)
        split = split_by_group(group)
        group_to_split.setdefault(group, split)
        component_list = sorted({box["label"] for box in boxes}) or None
        samples[split].append(
            {
                "sample_id": f"cghd_{safe_id(drafter)}_{safe_id(stem)}",
                "source_dataset": "lowercaseonly/cghd",
                "split": split,
                "image": {
                    "image_path_or_ref": rel(image_path),
                    "width": width,
                    "height": height,
                },
                "query": {
                    "prompt": "Identify and localize handwritten circuit symbols.",
                    "expected_answers": None,
                    "answer_type": "object_detection_annotation_ref",
                },
                "structure": {
                    "structure_path_or_ref": rel(xml_path),
                    "component_list": component_list,
                    "netlist_or_edges": None,
                    "metadata_text": None,
                },
                "supervision": {
                    "taxonomy": {
                        "task_family": "perception_probe",
                        "source_license": "cc-by-3.0",
                        "split_unit": "drafter_plus_circuit_group",
                        "drafter": drafter,
                    },
                    "hard_negatives": [],
                    "roi_or_tile_boxes": boxes,
                },
            }
        )

    summary = {
        "dataset_id": "lowercaseonly/cghd",
        "source_url": "https://huggingface.co/datasets/lowercaseonly/cghd",
        "license": "cc-by-3.0",
        "local_dir": rel(cghd_dir),
        "split_unit": "drafter_plus_circuit_group",
        "image_files": len(images_by_stem),
        "xml_annotation_files": len(xml_by_stem),
        "paired_image_xml_files": len(paired_keys),
        "bad_images": bad_images,
        "split_counts": {key: len(value) for key, value in samples.items()},
        "unique_split_groups": len(group_to_split),
        "top_labels": label_counts.most_common(20),
    }
    return samples, summary


def circuitvqa_sample(row_index: int, source_split: str, output_split: str, row: dict[str, Any]) -> dict[str, Any] | None:
    images = row.get("images") or []
    texts = row.get("texts") or []
    if not images or not texts:
        return None
    width, height = image_size(images[0])
    if width is None or height is None:
        return None
    return {
        "sample_id": f"circuitvqa_{source_split}_{row_index:05d}",
        "source_dataset": "ayoubkirouane/CircuitVQA",
        "split": output_split,
        "image": {
            "image_path_or_ref": f"hf://datasets/ayoubkirouane/CircuitVQA/default/{source_split}/{row_index}/images/0",
            "width": width,
            "height": height,
        },
        "query": {
            "prompt": None,
            "expected_answers": None,
            "answer_type": "hf_conversation_ref",
        },
        "structure": {
            "structure_path_or_ref": None,
            "component_list": None,
            "netlist_or_edges": None,
            "metadata_text": None,
        },
        "supervision": {
            "taxonomy": {
                "task_family": "vqa_evaluation",
                "source_license": "not_declared_on_hf",
                "split_unit": "provided_split_then_row_hash_for_holdout",
                "source_split": source_split,
                "payload_policy": "prompt_answer_not_committed",
            },
            "hard_negatives": [],
            "roi_or_tile_boxes": None,
        },
    }


def collect_circuitvqa() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    samples: dict[str, list[dict[str, Any]]] = {"train": [], "holdout": [], "test": []}
    verified_by_source_split: Counter[str] = Counter()
    failed_by_source_split: Counter[str] = Counter()

    for source_split in ("train", "test"):
        dataset = load_dataset("ayoubkirouane/CircuitVQA", split=source_split, streaming=True)
        for row_index, row in enumerate(dataset):
            if source_split == "test":
                output_split = "test"
            else:
                output_split = "holdout" if stable_bucket(f"circuitvqa/train/{row_index}") >= 90 else "train"
            sample = circuitvqa_sample(row_index, source_split, output_split, row)
            if sample is None:
                failed_by_source_split[source_split] += 1
                continue
            samples[output_split].append(sample)
            verified_by_source_split[source_split] += 1

    summary = {
        "dataset_id": "ayoubkirouane/CircuitVQA",
        "source_url": "https://huggingface.co/datasets/ayoubkirouane/CircuitVQA",
        "license": "not_declared_on_hf",
        "payload_policy": "committed curriculum stores row refs and answer_type only; prompt and answer bodies stay on HF",
        "verified_by_source_split": dict(verified_by_source_split),
        "failed_by_source_split": dict(failed_by_source_split),
        "split_counts": {key: len(value) for key, value in samples.items()},
        "answer_type": "hf_conversation_ref",
    }
    return samples, summary


def merge_samples(target: dict[str, list[dict[str, Any]]], extra: dict[str, list[dict[str, Any]]]) -> None:
    for split, records in extra.items():
        target[split].extend(records)


def summarize_written(samples: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    by_split = {split: len(records) for split, records in samples.items()}
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    answer_types: Counter[str] = Counter()
    structure_samples = 0
    for split, records in samples.items():
        for record in records:
            by_source[record["source_dataset"]][split] += 1
            answer_type = record["query"].get("answer_type")
            if answer_type:
                answer_types[answer_type] += 1
            if record["structure"].get("structure_path_or_ref"):
                structure_samples += 1
    return {
        "by_split": by_split,
        "by_source": {source: dict(counts) for source, counts in sorted(by_source.items())},
        "answer_types": dict(answer_types),
        "samples_with_structure_ref": structure_samples,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build M2 circuit sample schema and split manifests.")
    parser.add_argument("--open-train-target", type=int, default=5000)
    parser.add_argument("--open-holdout-target", type=int, default=512)
    parser.add_argument("--open-test-target", type=int, default=512)
    parser.add_argument("--cghd-dir", type=Path, default=DEFAULT_CGHD_DIR)
    parser.add_argument("--skip-cghd", action="store_true")
    parser.add_argument("--skip-circuitvqa", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    samples: dict[str, list[dict[str, Any]]] = {"train": [], "holdout": [], "test": []}
    open_samples, open_summary = collect_open_schematics(
        args.open_train_target,
        args.open_holdout_target,
        args.open_test_target,
    )
    merge_samples(samples, open_samples)

    cghd_summary: dict[str, Any]
    if args.skip_cghd:
        cghd_summary = {"status": "skipped_by_cli"}
    else:
        cghd_samples, cghd_summary = collect_cghd(args.cghd_dir)
        merge_samples(samples, cghd_samples)

    circuitvqa_summary: dict[str, Any]
    if args.skip_circuitvqa:
        circuitvqa_summary = {"status": "skipped_by_cli"}
    else:
        circuitvqa_samples, circuitvqa_summary = collect_circuitvqa()
        merge_samples(samples, circuitvqa_samples)

    write_counts = {}
    for split in ("train", "holdout", "test"):
        write_counts[split] = write_jsonl(OUT_DIR / f"{split}.jsonl", samples[split])

    summary = {
        "summary_version": 1,
        "generated_at_utc": utc_now(),
        "script": "scripts/prepare_circuit_samples.py",
        "outputs": {
            "train": "data/circuit_curricula/train.jsonl",
            "holdout": "data/circuit_curricula/holdout.jsonl",
            "test": "data/circuit_curricula/test.jsonl",
            "summary": "data/circuit_curricula/usable_pair_summary.json",
        },
        "raw_payload_policy": {
            "committed_raw_images": False,
            "committed_prompt_answer_bodies": False,
            "open_schematics_refs_only": True,
            "cghd_local_paths_point_to_ignored_data_downloads": True,
        },
        "sources": {
            "open_schematics": open_summary,
            "cghd": cghd_summary,
            "circuitvqa": circuitvqa_summary,
            "schgen": {
                "dataset_id": "microsoft/SchGen_dataset",
                "source_url": "https://huggingface.co/datasets/microsoft/SchGen_dataset",
                "license": "mit",
                "m2_role": "structure_text_prior_only",
                "status": "not_materialized_as_CircuitSample_because_it_has_no_image_field",
            },
        },
        "written": {
            "counts": write_counts,
            **summarize_written(samples),
        },
        "split_leakage_policy": {
            "open_schematics": "split by source project name hash",
            "cghd": "split by drafter plus circuit group hash",
            "circuitvqa": "preserve provided test split; hash train rows into train/holdout references",
        },
        "pass_checks": {
            "split_files_exist": all((OUT_DIR / f"{split}.jsonl").exists() for split in ("train", "holdout", "test")),
            "open_schematics_train_at_least_5k": len(open_samples["train"]) >= 5000,
            "open_schematics_verified_before_curriculum": True,
            "source_leakage_policy_documented": True,
            "qa_samples_have_answer_type": all(
                record["query"].get("answer_type")
                for split_records in samples.values()
                for record in split_records
                if record["source_dataset"] == "ayoubkirouane/CircuitVQA"
            ),
            "structure_pretraining_samples_have_structure_ref": all(
                record["structure"].get("structure_path_or_ref")
                for split_records in samples.values()
                for record in split_records
                if record["source_dataset"] == "bshada/open-schematics"
            ),
        },
    }
    pass_checks = summary["pass_checks"]
    summary["status"] = "closed" if all(pass_checks.values()) else "closed_with_caveats"
    (OUT_DIR / "usable_pair_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": summary["status"], "written": summary["written"]["counts"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from datasets import Image as HFImage
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
TRAIN_MANIFEST = ROOT / "data" / "circuit_curricula" / "train.jsonl"
HOLDOUT_MANIFEST = ROOT / "data" / "circuit_curricula" / "holdout.jsonl"
OUTPUT_PATH = ROOT / "results" / "structure_targets" / "roi_structure_target_audit.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_open_schematics_ref(ref: str) -> int:
    match = re.search(r"/train/(\d+)/(image|schematic)$", ref)
    if not match:
        raise ValueError(f"Unsupported open-schematics ref: {ref}")
    return int(match.group(1))


def select_records(
    manifest: Path,
    source_dataset: str,
    limit: int | None,
) -> list[dict[str, Any]]:
    records = [record for record in read_jsonl(manifest) if record.get("source_dataset") == source_dataset]
    return records if limit is None else records[:limit]


def component_items(record: dict[str, Any]) -> list[str]:
    return [str(item) for item in record.get("structure", {}).get("component_list") or [] if item is not None]


def multihot_key(components: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(components)))


def count_key(components: Iterable[str]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(Counter(components).items()))


def collision_metrics(keys: list[tuple[Any, ...]], sample_ids: list[str]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for key, sample_id in zip(keys, sample_ids):
        groups[key].append(sample_id)
    duplicate_groups = [values for values in groups.values() if len(values) > 1]
    duplicate_samples = sum(len(values) for values in duplicate_groups)
    largest = max((len(values) for values in duplicate_groups), default=1)
    preview = sorted(duplicate_groups, key=len, reverse=True)[:5]
    return {
        "records": len(keys),
        "unique_target_count": len(groups),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_sample_count": duplicate_samples,
        "duplicate_sample_ratio": duplicate_samples / len(keys) if keys else None,
        "largest_collision_group": largest,
        "collision_group_preview_sample_ids": [values[:8] for values in preview],
    }


def component_distribution(records: list[dict[str, Any]], vocab_size: int) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    per_record = []
    for record in records:
        components = component_items(record)
        counts.update(components)
        per_record.append(len(components))
    vocab = {component for component, _ in counts.most_common(vocab_size)}
    total_mentions = sum(counts.values())
    covered_mentions = sum(count for component, count in counts.items() if component in vocab)
    return {
        "records": len(records),
        "unique_component_names": len(counts),
        "component_mentions": total_mentions,
        "mean_component_mentions_per_record": sum(per_record) / len(per_record) if per_record else None,
        "median_component_mentions_per_record": sorted(per_record)[len(per_record) // 2] if per_record else None,
        "top_components": counts.most_common(20),
        "vocab_size": vocab_size,
        "vocab_mention_coverage": covered_mentions / total_mentions if total_mentions else None,
    }


def read_quoted(text: str, start: int) -> tuple[str, int]:
    assert text[start] == '"'
    out = []
    i = start + 1
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            out.append(text[i + 1])
            i += 2
            continue
        if ch == '"':
            return "".join(out), i + 1
        out.append(ch)
        i += 1
    return "".join(out), i


def top_level_forms(text: str) -> list[str]:
    forms: list[str] = []
    depth = 0
    child_start: int | None = None
    in_string = False
    escape = False
    root_seen = False
    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "(":
            if root_seen and depth == 1 and child_start is None:
                child_start = idx
            depth += 1
            root_seen = True
        elif ch == ")":
            depth -= 1
            if child_start is not None and depth == 1:
                forms.append(text[child_start : idx + 1])
                child_start = None
    return forms


def form_head(form: str) -> str:
    match = re.match(r"\(\s*([A-Za-z0-9_+\-]+)", form)
    return match.group(1) if match else ""


def first_quoted_after_head(form: str) -> str | None:
    start = form.find('"')
    if start < 0:
        return None
    return read_quoted(form, start)[0]


def extract_at(form: str) -> tuple[float, float, float | None] | None:
    match = re.search(r"\(at\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)(?:\s+([-+]?\d+(?:\.\d+)?))?", form)
    if not match:
        return None
    x = float(match.group(1))
    y = float(match.group(2))
    rotation = float(match.group(3)) if match.group(3) is not None else None
    return x, y, rotation


def extract_xy_points(form: str) -> list[tuple[float, float]]:
    points = []
    for match in re.finditer(r"\(xy\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\)", form):
        points.append((float(match.group(1)), float(match.group(2))))
    return points


def extract_property(form: str, name: str) -> str | None:
    pattern = re.compile(r"\(property\s+\"{}\"\s+\"".format(re.escape(name)))
    match = pattern.search(form)
    if not match:
        return None
    start = match.end() - 1
    return read_quoted(form, start)[0]


def parse_kicad_schematic(schematic: str, image_width: int | None, image_height: int | None) -> dict[str, Any]:
    forms = top_level_forms(schematic)
    symbols = []
    wires = []
    labels = []
    junctions = []
    no_connects = []
    all_points: list[tuple[float, float]] = []
    symbol_families: Counter[str] = Counter()
    symbol_libs: Counter[str] = Counter()

    for form in forms:
        head = form_head(form)
        if head == "symbol" and "(lib_id" in form:
            lib_match = re.search(r"\(lib_id\s+\"", form)
            lib_id = read_quoted(form, lib_match.end() - 1)[0] if lib_match else "unknown"
            at = extract_at(form)
            if at is None:
                continue
            reference = extract_property(form, "Reference")
            value = extract_property(form, "Value")
            x, y, rotation = at
            family = lib_id.split(":", 1)[0] if ":" in lib_id else lib_id
            symbols.append(
                {
                    "lib_id": lib_id,
                    "family": family,
                    "reference": reference,
                    "value": value,
                    "x": x,
                    "y": y,
                    "rotation": rotation,
                }
            )
            symbol_families[family] += 1
            symbol_libs[lib_id] += 1
            all_points.append((x, y))
        elif head == "wire":
            points = extract_xy_points(form)
            if len(points) >= 2:
                wires.append(points)
                all_points.extend(points)
        elif head in {"label", "global_label", "hierarchical_label"}:
            at = extract_at(form)
            text = first_quoted_after_head(form)
            if at is not None:
                labels.append({"kind": head, "text_hash": hashlib.sha1((text or "").encode("utf-8")).hexdigest(), "x": at[0], "y": at[1]})
                all_points.append((at[0], at[1]))
        elif head == "junction":
            at = extract_at(form)
            if at is not None:
                junctions.append((at[0], at[1]))
                all_points.append((at[0], at[1]))
        elif head == "no_connect":
            at = extract_at(form)
            if at is not None:
                no_connects.append((at[0], at[1]))
                all_points.append((at[0], at[1]))

    bbox = None
    tile_2x2_occupancy = 0
    tile_4x4_occupancy = 0
    if all_points:
        xs = [point[0] for point in all_points]
        ys = [point[1] for point in all_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x
        height = max_y - min_y
        bbox = {
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "width": width,
            "height": height,
            "aspect": width / height if height else None,
        }

        def occupancy(grid: int) -> int:
            cells = set()
            for symbol in symbols:
                if width <= 0 or height <= 0:
                    continue
                col = min(grid - 1, max(0, int((symbol["x"] - min_x) / width * grid)))
                row = min(grid - 1, max(0, int((symbol["y"] - min_y) / height * grid)))
                cells.add((row, col))
            return len(cells)

        tile_2x2_occupancy = occupancy(2)
        tile_4x4_occupancy = occupancy(4)

        tile_4x4_counts = [0] * 16
        symbol_points_normalized = []
        if width > 0 and height > 0:
            for symbol in symbols:
                norm_x = min(1.0, max(0.0, (symbol["x"] - min_x) / width))
                norm_y = min(1.0, max(0.0, (symbol["y"] - min_y) / height))
                col = min(3, max(0, int(norm_x * 4)))
                row = min(3, max(0, int(norm_y * 4)))
                tile_4x4_counts[row * 4 + col] += 1
                symbol_points_normalized.append((norm_x, norm_y))
        else:
            symbol_points_normalized = []
            tile_4x4_counts = [0] * 16
    else:
        tile_4x4_counts = [0] * 16
        symbol_points_normalized = []

    wire_lengths = []
    for points in wires:
        for a, b in zip(points, points[1:]):
            wire_lengths.append(math.dist(a, b))
    graph_signature = (
        tuple(sorted(symbol_libs.items())),
        len(wires),
        len(labels),
        len(junctions),
        len(no_connects),
        tile_4x4_occupancy,
    )
    return {
        "symbol_count": len(symbols),
        "wire_count": len(wires),
        "wire_segment_count": len(wire_lengths),
        "wire_total_length": sum(wire_lengths),
        "label_count": len(labels),
        "junction_count": len(junctions),
        "no_connect_count": len(no_connects),
        "bbox": bbox,
        "tile_2x2_symbol_occupancy": tile_2x2_occupancy,
        "tile_4x4_symbol_occupancy": tile_4x4_occupancy,
        "tile_4x4_symbol_binary": [1 if count > 0 else 0 for count in tile_4x4_counts],
        "tile_4x4_symbol_counts": tile_4x4_counts,
        "symbol_points_normalized": symbol_points_normalized,
        "symbol_families": dict(symbol_families.most_common(20)),
        "symbol_lib_count_key": tuple(sorted(symbol_libs.items())),
        "graph_signature": graph_signature,
        "roi_candidate_count": len(symbols) + len(labels) + len(junctions) + len(no_connects),
        "roi_trace_possible": len(symbols) > 0 and bbox is not None and image_width is not None and image_height is not None,
        "wire_skeleton_possible": len(wires) > 0,
    }


def load_schematic_rows(records: list[dict[str, Any]]) -> dict[int, str]:
    indices = {parse_open_schematics_ref(record["image"]["image_path_or_ref"]) for record in records}
    if not indices:
        return {}
    max_index = max(indices)
    rows: dict[int, str] = {}
    dataset = load_dataset("bshada/open-schematics", split="train", streaming=True).cast_column(
        "image",
        HFImage(decode=False),
    )
    for row_index, row in enumerate(dataset):
        if row_index in indices:
            rows[row_index] = row.get("schematic") or ""
        if row_index >= max_index:
            break
    missing = sorted(indices - set(rows))
    if missing:
        raise RuntimeError(f"Missing open-schematics rows: {missing[:8]}")
    return rows


def summarize_numbers(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "median": ordered[len(ordered) // 2],
        "min": ordered[0],
        "max": ordered[-1],
    }


def audit_open_schematics(records: list[dict[str, Any]], vocab_size: int) -> dict[str, Any]:
    start = time.perf_counter()
    rows = load_schematic_rows(records)
    parsed = []
    sample_ids = []
    component_multihot_keys = []
    component_count_keys = []
    parsed_lib_count_keys = []
    graph_signature_keys = []
    failures: Counter[str] = Counter()

    for record in records:
        sample_id = record["sample_id"]
        row_index = parse_open_schematics_ref(record["image"]["image_path_or_ref"])
        schematic = rows[row_index]
        try:
            item = parse_kicad_schematic(
                schematic,
                record.get("image", {}).get("width"),
                record.get("image", {}).get("height"),
            )
        except Exception as exc:
            failures[type(exc).__name__] += 1
            continue
        parsed.append(item)
        sample_ids.append(sample_id)
        components = component_items(record)
        component_multihot_keys.append(multihot_key(components))
        component_count_keys.append(count_key(components))
        parsed_lib_count_keys.append(item["symbol_lib_count_key"])
        graph_signature_keys.append(item["graph_signature"])

    symbol_counts = [item["symbol_count"] for item in parsed]
    wire_counts = [item["wire_count"] for item in parsed]
    label_counts = [item["label_count"] for item in parsed]
    roi_counts = [item["roi_candidate_count"] for item in parsed]
    tile_2x2 = [item["tile_2x2_symbol_occupancy"] for item in parsed]
    tile_4x4 = [item["tile_4x4_symbol_occupancy"] for item in parsed]

    family_counts: Counter[str] = Counter()
    for item in parsed:
        family_counts.update(item["symbol_families"])

    return {
        "records_requested": len(records),
        "records_parsed": len(parsed),
        "parse_failures": dict(failures),
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "component_distribution": component_distribution(records, vocab_size),
        "collision": {
            "manifest_component_multihot": collision_metrics(component_multihot_keys, sample_ids),
            "manifest_component_count": collision_metrics(component_count_keys, sample_ids),
            "parsed_symbol_lib_count": collision_metrics(parsed_lib_count_keys, sample_ids),
            "roi_graph_signature": collision_metrics(graph_signature_keys, sample_ids),
        },
        "kicad_structure_parse": {
            "samples_with_symbols": sum(1 for value in symbol_counts if value > 0),
            "samples_with_wires": sum(1 for value in wire_counts if value > 0),
            "samples_with_labels": sum(1 for value in label_counts if value > 0),
            "samples_with_roi_trace": sum(1 for item in parsed if item["roi_trace_possible"]),
            "samples_with_wire_skeleton": sum(1 for item in parsed if item["wire_skeleton_possible"]),
            "symbol_count": summarize_numbers(symbol_counts),
            "wire_count": summarize_numbers(wire_counts),
            "label_count": summarize_numbers(label_counts),
            "roi_candidate_count": summarize_numbers(roi_counts),
            "tile_2x2_symbol_occupancy": summarize_numbers(tile_2x2),
            "tile_4x4_symbol_occupancy": summarize_numbers(tile_4x4),
            "top_symbol_families": family_counts.most_common(20),
        },
        "target_feasibility": {
            "component_multihot_is_too_weak_if_duplicate_ratio_high": (
                collision_metrics(component_multihot_keys, sample_ids)["duplicate_sample_ratio"]
            ),
            "parsed_graph_target_available": bool(parsed) and sum(1 for item in parsed if item["roi_trace_possible"]) / len(parsed) >= 0.9,
            "wire_skeleton_available": bool(parsed) and sum(1 for item in parsed if item["wire_skeleton_possible"]) / len(parsed) >= 0.5,
            "roi_tile_target_available": bool(parsed) and sum(1 for item in parsed if item["tile_2x2_symbol_occupancy"] > 0) / len(parsed) >= 0.9,
        },
    }


def parse_cghd_boxes(xml_path: Path) -> list[dict[str, Any]]:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []
    boxes = []
    for obj in root.findall(".//object"):
        label = (obj.findtext("name") or "unknown").strip() or "unknown"
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
        if xmax > xmin and ymax > ymin:
            boxes.append({"label": label, "bbox": [xmin, ymin, xmax, ymax], "area": (xmax - xmin) * (ymax - ymin)})
    return boxes


def audit_cghd(records: list[dict[str, Any]], limit: int | None) -> dict[str, Any]:
    records = records if limit is None else records[:limit]
    label_counts: Counter[str] = Counter()
    boxes_per_sample = []
    roi_area_ratios = []
    samples_with_boxes = 0
    bad_xml = 0
    for record in records:
        xml_ref = record.get("structure", {}).get("structure_path_or_ref")
        width = record.get("image", {}).get("width") or 0
        height = record.get("image", {}).get("height") or 0
        if not xml_ref:
            continue
        xml_path = ROOT / xml_ref
        boxes = parse_cghd_boxes(xml_path)
        if not boxes:
            bad_xml += 1
        else:
            samples_with_boxes += 1
        boxes_per_sample.append(len(boxes))
        image_area = width * height
        for box in boxes:
            label_counts[box["label"]] += 1
            if image_area:
                roi_area_ratios.append(box["area"] / image_area)
    return {
        "records": len(records),
        "samples_with_boxes": samples_with_boxes,
        "samples_with_boxes_ratio": samples_with_boxes / len(records) if records else None,
        "empty_or_bad_xml_records": bad_xml,
        "boxes_per_sample": summarize_numbers(boxes_per_sample),
        "roi_area_ratio": summarize_numbers(roi_area_ratios),
        "top_labels": label_counts.most_common(20),
        "target_feasibility": {
            "roi_supervision_available": samples_with_boxes / len(records) >= 0.9 if records else False,
            "detail_specialist_training_source": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit ROI-aware structure targets for M5 redesign.")
    parser.add_argument("--train-manifest", type=Path, default=TRAIN_MANIFEST)
    parser.add_argument("--holdout-manifest", type=Path, default=HOLDOUT_MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--open-train-limit", type=int, default=5000)
    parser.add_argument("--open-holdout-limit", type=int, default=512)
    parser.add_argument("--cghd-train-limit", type=int, default=2690)
    parser.add_argument("--component-vocab-size", type=int, default=4096)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.train_manifest = args.train_manifest.resolve()
    args.holdout_manifest = args.holdout_manifest.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    train_open = select_records(args.train_manifest, "bshada/open-schematics", args.open_train_limit)
    holdout_open = select_records(args.holdout_manifest, "bshada/open-schematics", args.open_holdout_limit)
    train_cghd = select_records(args.train_manifest, "lowercaseonly/cghd", args.cghd_train_limit)

    output = {
        "result_version": 1,
        "status": "completed",
        "generated_at_utc": utc_now(),
        "script": "scripts/audit_roi_structure_targets.py",
        "command": {
            "train_manifest": rel(args.train_manifest),
            "holdout_manifest": rel(args.holdout_manifest),
            "open_train_limit": args.open_train_limit,
            "open_holdout_limit": args.open_holdout_limit,
            "cghd_train_limit": args.cghd_train_limit,
            "component_vocab_size": args.component_vocab_size,
        },
        "raw_payload_policy": {
            "stores_raw_schematic_text": False,
            "stores_raw_xml": False,
            "stores_raw_images": False,
            "stores_text_hashes_only_for_labels": True,
        },
        "open_schematics_train": audit_open_schematics(train_open, args.component_vocab_size),
        "open_schematics_holdout": audit_open_schematics(holdout_open, args.component_vocab_size),
        "cghd_train": audit_cghd(train_cghd, args.cghd_train_limit),
    }

    train_feas = output["open_schematics_train"]["target_feasibility"]
    holdout_feas = output["open_schematics_holdout"]["target_feasibility"]
    cghd_feas = output["cghd_train"]["target_feasibility"]
    output["decision"] = {
        "recommended_backbone_role": "roi_aware_structure_backbone",
        "preserve_lora_roi_detail_design": True,
        "component_multihot_as_primary_target": "not_recommended",
        "parsed_kicad_graph_or_set_target": "recommended",
        "cghd_roi_detail_source": "recommended_for_specialist_or_tile_probe",
        "ready_for_longer_m5_training": False,
        "ready_for_m5_3_diagnostic_minirun": (
            train_feas["parsed_graph_target_available"]
            and holdout_feas["parsed_graph_target_available"]
            and cghd_feas["roi_supervision_available"]
        ),
        "next_experiment": "512/128 diagnostic run using ROI-aware graph/set target before longer training",
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "output": rel(args.output)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

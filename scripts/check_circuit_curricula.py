from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CURRICULA = ROOT / "data" / "circuit_curricula"
SPLITS = ("train", "holdout", "test")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_no}: invalid json: {exc}") from exc
    return rows


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_row(row: dict[str, Any], expected_split: str) -> None:
    for key in [
        "sample_id",
        "sample_family",
        "source_dataset",
        "source_group_key",
        "split",
        "source",
        "image",
        "query",
        "structure",
        "supervision",
        "provenance",
    ]:
        require(key in row, f"{row.get('sample_id', '<missing>')}: missing {key}")

    require(row["split"] == expected_split, f"{row['sample_id']}: split mismatch")
    require(row["provenance"].get("source_url"), f"{row['sample_id']}: missing source_url")

    family = row["sample_family"]
    if family == "vqa_evaluation":
        require(row["query"].get("answer_type"), f"{row['sample_id']}: VQA answer_type is required")
        require(row["query"].get("prompt_ref"), f"{row['sample_id']}: VQA prompt_ref is required")
        require(row["query"].get("expected_answer_ref"), f"{row['sample_id']}: VQA expected_answer_ref is required")
        require(row["image"].get("image_path_or_ref"), f"{row['sample_id']}: VQA image ref is required")

    if family in {"structure_pretraining", "structure_text_prior", "perception_probe"}:
        require(
            row["structure"].get("structure_path_or_ref"),
            f"{row['sample_id']}: structure_path_or_ref is required for {family}",
        )

    if family == "perception_probe":
        file_paths = row["source"].get("file_paths") or {}
        require(file_paths.get("image"), f"{row['sample_id']}: CGHD image path missing")
        require(file_paths.get("annotation_xml"), f"{row['sample_id']}: CGHD xml annotation missing")


def main() -> int:
    all_ids: set[str] = set()
    source_group_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    summary: dict[str, Any] = {"split_counts": {}, "source_counts": {}, "family_counts": {}}
    for split in SPLITS:
        path = CURRICULA / f"{split}.jsonl"
        require(path.exists(), f"missing split file: {path}")
        rows = read_jsonl(path)
        require(rows, f"split has no rows: {split}")
        source_counts = Counter()
        family_counts = Counter()
        for row in rows:
            validate_row(row, split)
            require(row["sample_id"] not in all_ids, f"duplicate sample_id: {row['sample_id']}")
            all_ids.add(row["sample_id"])
            source_group_splits[(row["source_dataset"], row["source_group_key"])].add(split)
            source_counts[row["source_dataset"]] += 1
            family_counts[row["sample_family"]] += 1
        summary["split_counts"][split] = len(rows)
        summary["source_counts"][split] = dict(sorted(source_counts.items()))
        summary["family_counts"][split] = dict(sorted(family_counts.items()))

    leaked_groups = {
        f"{dataset}:{group_key}": sorted(splits)
        for (dataset, group_key), splits in source_group_splits.items()
        if len(splits) > 1
    }
    require(not leaked_groups, f"source group leakage across splits: {leaked_groups}")
    summary["unique_source_groups"] = len(source_group_splits)

    output = CURRICULA / "check_summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

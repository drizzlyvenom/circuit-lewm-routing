from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image as PILImage
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
sys.path.insert(0, str(SCRIPT_PATH.parent))
sys.path.insert(0, str(REPO_ROOT))

import train_circuit_lewm_s as base  # noqa: E402
from scripts.audit_roi_structure_targets import parse_kicad_schematic  # noqa: E402


SUMMARY_PATH = REPO_ROOT / "results" / "lewm_s" / "m5_3_roi_graph_diagnostic.json"
SCALAR_NAMES = [
    "symbol_count",
    "wire_count",
    "label_count",
    "junction_count",
    "no_connect_count",
    "roi_candidate_count",
    "wire_total_length",
    "bbox_aspect",
    "tile_2x2_symbol_occupancy",
    "tile_4x4_symbol_occupancy",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def load_schematic_rows(records: list[dict[str, Any]]) -> dict[int, str]:
    row_to_record = {
        base.parse_open_schematics_ref(record["image"]["image_path_or_ref"]): record
        for record in records
    }
    if not row_to_record:
        return {}
    rows: dict[int, str] = {}
    max_index = max(row_to_record)
    dataset = load_dataset("bshada/open-schematics", split="train", streaming=True).cast_column(
        "image",
        HFImage(decode=False),
    )
    for row_index, row in enumerate(dataset):
        if row_index in row_to_record:
            rows[row_index] = row.get("schematic") or ""
        if row_index >= max_index:
            break
    missing = sorted(set(row_to_record) - set(rows))
    if missing:
        raise RuntimeError(f"Missing open-schematics rows: {missing[:8]}")
    return rows


def parse_record_target(record: dict[str, Any], schematic: str) -> dict[str, Any]:
    return parse_kicad_schematic(
        schematic,
        record.get("image", {}).get("width"),
        record.get("image", {}).get("height"),
    )


class RoiGraphTargetBuilder:
    def __init__(self, lib_vocab: list[str], family_vocab: list[str]) -> None:
        self.lib_vocab = lib_vocab
        self.family_vocab = family_vocab
        self.lib_index = {name: idx for idx, name in enumerate(lib_vocab)}
        self.family_index = {name: idx for idx, name in enumerate(family_vocab)}
        self.family_offset = len(lib_vocab)
        self.tile_offset = self.family_offset + len(family_vocab)
        self.scalar_offset = self.tile_offset + 16
        self.target_dim = self.scalar_offset + len(SCALAR_NAMES)

    def metadata(self) -> dict[str, Any]:
        return {
            "mode": "roi_graph_set",
            "target_dim": self.target_dim,
            "lib_vocab_size": len(self.lib_vocab),
            "family_vocab_size": len(self.family_vocab),
            "tile_offset": self.tile_offset,
            "tile_size": 16,
            "scalar_names": SCALAR_NAMES,
            "raw_payload_policy": {
                "stores_raw_schematic_text": False,
                "stores_raw_images": False,
                "stores_raw_label_text": False,
            },
        }

    def encode(self, record: dict[str, Any], schematic: str) -> tuple[torch.Tensor, dict[str, Any]]:
        parsed = parse_record_target(record, schematic)
        vector = torch.zeros(self.target_dim, dtype=torch.float32)

        family_counts: Counter[str] = Counter()
        for lib_id, count in parsed["symbol_lib_count_key"]:
            count = float(count)
            lib_idx = self.lib_index.get(lib_id)
            if lib_idx is not None:
                vector[lib_idx] = min(1.0, math.log1p(count) / math.log1p(32.0))
            family = lib_id.split(":", 1)[0] if ":" in lib_id else lib_id
            family_counts[family] += count

        for family, count in family_counts.items():
            family_idx = self.family_index.get(family)
            if family_idx is not None:
                vector[self.family_offset + family_idx] = min(1.0, math.log1p(float(count)) / math.log1p(64.0))

        tile_binary = list(parsed.get("tile_4x4_symbol_binary") or [0] * 16)
        for idx, value in enumerate(tile_binary[:16]):
            vector[self.tile_offset + idx] = float(value)

        bbox = parsed.get("bbox") or {}
        aspect = bbox.get("aspect") or 1.0
        aspect_feature = (math.log(max(float(aspect), 1e-6)) + 4.0) / 8.0
        scalar_values = {
            "symbol_count": min(1.0, math.log1p(parsed["symbol_count"]) / math.log1p(256.0)),
            "wire_count": min(1.0, math.log1p(parsed["wire_count"]) / math.log1p(800.0)),
            "label_count": min(1.0, math.log1p(parsed["label_count"]) / math.log1p(500.0)),
            "junction_count": min(1.0, math.log1p(parsed["junction_count"]) / math.log1p(1000.0)),
            "no_connect_count": min(1.0, math.log1p(parsed["no_connect_count"]) / math.log1p(500.0)),
            "roi_candidate_count": min(1.0, math.log1p(parsed["roi_candidate_count"]) / math.log1p(1000.0)),
            "wire_total_length": min(1.0, math.log1p(parsed["wire_total_length"]) / math.log1p(10000.0)),
            "bbox_aspect": min(1.0, max(0.0, aspect_feature)),
            "tile_2x2_symbol_occupancy": parsed["tile_2x2_symbol_occupancy"] / 4.0,
            "tile_4x4_symbol_occupancy": parsed["tile_4x4_symbol_occupancy"] / 16.0,
        }
        for idx, name in enumerate(SCALAR_NAMES):
            vector[self.scalar_offset + idx] = float(scalar_values[name])

        meta = {
            "graph_signature_sha1": stable_hash(parsed["graph_signature"]),
            "symbol_count": parsed["symbol_count"],
            "wire_count": parsed["wire_count"],
            "label_count": parsed["label_count"],
            "roi_candidate_count": parsed["roi_candidate_count"],
            "tile_4x4_positive_count": int(sum(tile_binary)),
            "symbol_points_normalized": parsed.get("symbol_points_normalized") or [],
        }
        return vector, meta


def build_target_builder(
    records: list[dict[str, Any]],
    lib_vocab_size: int,
    family_vocab_size: int,
) -> RoiGraphTargetBuilder:
    rows = load_schematic_rows(records)
    lib_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    for record in records:
        row_index = base.parse_open_schematics_ref(record["image"]["image_path_or_ref"])
        parsed = parse_record_target(record, rows[row_index])
        for lib_id, count in parsed["symbol_lib_count_key"]:
            lib_counts[lib_id] += int(count)
            family = lib_id.split(":", 1)[0] if ":" in lib_id else lib_id
            family_counts[family] += int(count)
    return RoiGraphTargetBuilder(
        [name for name, _ in lib_counts.most_common(lib_vocab_size)],
        [name for name, _ in family_counts.most_common(family_vocab_size)],
    )


def roi_crop(image: PILImage.Image, sample_id: str, target_meta: dict[str, Any]) -> PILImage.Image:
    points = target_meta.get("symbol_points_normalized") or []
    if not points:
        return base.deterministic_crop(image, sample_id)
    bucket = int(hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:8], 16) % len(points)
    norm_x, norm_y = points[bucket]
    width, height = image.size
    crop_width = max(1, width // 2)
    crop_height = max(1, height // 2)
    center_x = int(float(norm_x) * width)
    center_y = int(float(norm_y) * height)
    left = min(max(0, center_x - crop_width // 2), max(0, width - crop_width))
    top = min(max(0, center_y - crop_height // 2), max(0, height - crop_height))
    return image.crop((left, top, left + crop_width, top + crop_height))


def materialize_open_samples(
    records: list[dict[str, Any]],
    target_builder: RoiGraphTargetBuilder,
    image_size: int,
) -> list[dict[str, Any]]:
    row_to_record = {
        base.parse_open_schematics_ref(record["image"]["image_path_or_ref"]): record
        for record in records
    }
    if not row_to_record:
        return []
    samples_by_row: dict[int, dict[str, Any]] = {}
    max_index = max(row_to_record)
    dataset = load_dataset("bshada/open-schematics", split="train", streaming=True).cast_column(
        "image",
        HFImage(decode=False),
    )
    for row_index, row in enumerate(dataset):
        record = row_to_record.get(row_index)
        if record is not None:
            image = base.decode_hf_image(row["image"])
            schematic = row.get("schematic") or ""
            target_vec, target_meta = target_builder.encode(record, schematic)
            samples_by_row[row_index] = {
                "sample_id": record["sample_id"],
                "row_index": row_index,
                "global_image": base.pil_to_uint8_tensor(image, image_size),
                "crop_image": base.pil_to_uint8_tensor(roi_crop(image, record["sample_id"], target_meta), image_size),
                "structure_vec": target_vec,
                "structure_sha1": base.sha1_text(schematic),
                "structure_chars": len(schematic),
                "target_meta": target_meta,
            }
        if row_index >= max_index:
            break
    missing = sorted(set(row_to_record) - set(samples_by_row))
    if missing:
        raise RuntimeError(f"Missing open-schematics rows: {missing[:8]}")
    return [samples_by_row[base.parse_open_schematics_ref(record["image"]["image_path_or_ref"])] for record in records]


class RoiGraphDataset(Dataset[dict[str, Any]]):
    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        return {
            "sample_id": sample["sample_id"],
            "row_index": sample["row_index"],
            "global_image": base.uint8_to_normalized_float(sample["global_image"]),
            "crop_image": base.uint8_to_normalized_float(sample["crop_image"]),
            "structure_vec": sample["structure_vec"].clone(),
            "structure_sha1": sample["structure_sha1"],
            "structure_chars": sample["structure_chars"],
            "target_meta": sample["target_meta"],
        }


def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_ids": [item["sample_id"] for item in items],
        "row_indices": torch.tensor([item["row_index"] for item in items], dtype=torch.long),
        "global_image": torch.stack([item["global_image"] for item in items]),
        "crop_image": torch.stack([item["crop_image"] for item in items]),
        "structure_vec": torch.stack([item["structure_vec"] for item in items]),
        "structure_sha1": [item["structure_sha1"] for item in items],
        "structure_chars": torch.tensor([item["structure_chars"] for item in items], dtype=torch.long),
        "target_meta": [item["target_meta"] for item in items],
    }


class CircuitLeWMSRoiGraph(nn.Module):
    def __init__(
        self,
        official_repo: Path,
        target_dim: int,
        image_size: int,
        embed_dim: int,
        predictor_depth: int,
    ) -> None:
        super().__init__()
        sys.path.insert(0, str(official_repo))
        import stable_pretraining as spt
        from module import MLP
        from module import SIGReg

        self.encoder = spt.backbone.utils.vit_hf(
            "tiny",
            patch_size=14,
            image_size=image_size,
            pretrained=False,
            use_mask_token=False,
        )
        self.projector = MLP(
            input_dim=embed_dim,
            hidden_dim=2048,
            output_dim=embed_dim,
            norm_fn=nn.BatchNorm1d,
        )
        self.structure_encoder = nn.Sequential(
            nn.Linear(target_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.crop_predictor = nn.Sequential(
            *[base.ResidualMLPBlock(embed_dim, 2048, dropout=0.1) for _ in range(predictor_depth)],
            nn.LayerNorm(embed_dim),
        )
        self.target_head = nn.Linear(embed_dim, target_dim)
        self.sigreg = SIGReg(knots=17, num_proj=1024)

    def encode_image(self, pixels: torch.Tensor) -> torch.Tensor:
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        cls = output.last_hidden_state[:, 0]
        return self.projector(cls)

    def forward(self, global_image: torch.Tensor, crop_image: torch.Tensor, structure_vec: torch.Tensor) -> dict[str, torch.Tensor]:
        image_z = self.encode_image(global_image)
        crop_z = self.encode_image(crop_image)
        structure_z = self.structure_encoder(structure_vec)
        pred_crop_z = self.crop_predictor(image_z)
        target_logits = self.target_head(image_z)
        return {
            "image_z": image_z,
            "crop_z": crop_z,
            "structure_z": structure_z,
            "pred_crop_z": pred_crop_z,
            "target_logits": target_logits,
        }


def compute_losses(
    output: dict[str, torch.Tensor],
    structure_vec: torch.Tensor,
    sigreg_module: nn.Module,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    align_loss, _ = base.contrastive_loss(output["image_z"], output["structure_z"], args.temperature)
    mask_loss = F.mse_loss(F.normalize(output["pred_crop_z"], dim=-1), F.normalize(output["crop_z"].detach(), dim=-1))
    target_loss = F.mse_loss(torch.sigmoid(output["target_logits"]), structure_vec)
    tile_logits = output["target_logits"][:, args.tile_start : args.tile_end]
    tile_target = structure_vec[:, args.tile_start : args.tile_end]
    tile_loss = F.binary_cross_entropy_with_logits(tile_logits, tile_target)
    sigreg_loss = output["image_z"].new_tensor(0.0)
    if args.lambda_sigreg > 0:
        sigreg_loss = sigreg_module(output["image_z"].unsqueeze(0))
    total = (
        args.lambda_align * align_loss
        + args.lambda_mask * mask_loss
        + args.lambda_target * target_loss
        + args.lambda_tile * tile_loss
        + args.lambda_sigreg * sigreg_loss
    )
    return {
        "loss": total,
        "align_loss": align_loss,
        "mask_loss": mask_loss,
        "target_loss": target_loss,
        "tile_loss": tile_loss,
        "sigreg_loss": sigreg_loss,
    }


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("global_image", "crop_image", "structure_vec", "row_indices", "structure_chars"):
        moved[key] = batch[key].to(device, non_blocking=True)
    return moved


def tile_topk_recall(tile_scores: torch.Tensor, tile_target: torch.Tensor) -> tuple[float, float]:
    recalls = []
    random_recalls = []
    for scores, target in zip(tile_scores, tile_target):
        positives = (target > 0.5).nonzero(as_tuple=False).flatten()
        if positives.numel() == 0:
            continue
        k = int(positives.numel())
        pred = scores.topk(k=min(k, scores.numel())).indices
        hit = torch.isin(positives, pred).float().mean().item()
        recalls.append(hit)
        random_recalls.append(k / scores.numel())
    if not recalls:
        return 0.0, 0.0
    return float(sum(recalls) / len(recalls)), float(sum(random_recalls) / len(random_recalls))


@torch.no_grad()
def evaluate(
    model: CircuitLeWMSRoiGraph,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    tile_start: int,
    tile_end: int,
) -> dict[str, float]:
    model.eval()
    image_vectors: list[torch.Tensor] = []
    structure_vectors: list[torch.Tensor] = []
    target_preds: list[torch.Tensor] = []
    target_true: list[torch.Tensor] = []
    autocast_enabled = device.type == "cuda" and precision == "bf16"
    for batch in loader:
        batch = batch_to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            output = model(batch["global_image"], batch["crop_image"], batch["structure_vec"])
        image_vectors.append(F.normalize(output["image_z"].float(), dim=-1).cpu())
        structure_vectors.append(F.normalize(output["structure_z"].float(), dim=-1).cpu())
        target_preds.append(torch.sigmoid(output["target_logits"].float()).cpu())
        target_true.append(batch["structure_vec"].float().cpu())

    image_matrix = torch.cat(image_vectors, dim=0)
    structure_matrix = torch.cat(structure_vectors, dim=0)
    sims = image_matrix @ structure_matrix.T
    labels = torch.arange(sims.size(0))
    top1 = (sims.argmax(dim=1) == labels).float().mean().item()
    top5 = (sims.topk(k=min(5, sims.size(1)), dim=1).indices == labels[:, None]).any(dim=1).float().mean().item()

    pred_matrix = torch.cat(target_preds, dim=0)
    true_matrix = torch.cat(target_true, dim=0)
    target_mse = F.mse_loss(pred_matrix, true_matrix).item()
    tile_recall, tile_random = tile_topk_recall(pred_matrix[:, tile_start:tile_end], true_matrix[:, tile_start:tile_end])

    return {
        "holdout_alignment_retrieval_top1": top1,
        "holdout_alignment_retrieval_top5": top5,
        "holdout_random_top1": 1.0 / sims.size(0),
        "holdout_random_top5": min(5, sims.size(0)) / sims.size(0),
        "holdout_size": float(sims.size(0)),
        "holdout_target_mse": target_mse,
        "holdout_tile_topk_recall": tile_recall,
        "holdout_tile_random_topk_recall": tile_random,
    }


def write_metrics_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "step",
                "loss",
                "align_loss",
                "mask_loss",
                "target_loss",
                "tile_loss",
                "sigreg_loss",
                "samples_per_second",
            ],
        )
        writer.writeheader()


def append_metric_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writerow(row)


def save_checkpoint(
    path: Path,
    model: CircuitLeWMSRoiGraph,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    target_builder: RoiGraphTargetBuilder,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "target_builder": target_builder.metadata(),
            "config": {
                "model": "CircuitLeWMSRoiGraph",
                "encoder": "vit_tiny",
                "image_size": args.image_size,
                "embed_dim": args.embed_dim,
                "predictor_depth": args.predictor_depth,
            },
        },
        path,
    )


def current_process_vram_mb() -> int | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    pid = str(os.getpid())
    total = 0
    found = False
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0] == pid:
            found = True
            try:
                total += int(parts[1])
            except ValueError:
                pass
    return total if found else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run M5.3 Circuit LeWM-S ROI-aware graph/set diagnostic.")
    parser.add_argument("--official-repo", type=Path, default=os.environ.get("LEWM_OFFICIAL_REPO"))
    parser.add_argument("--train-manifest", type=Path, default=base.TRAIN_MANIFEST)
    parser.add_argument("--holdout-manifest", type=Path, default=base.HOLDOUT_MANIFEST)
    parser.add_argument("--summary-output", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=512)
    parser.add_argument("--holdout-limit", type=int, default=128)
    parser.add_argument("--lib-vocab-size", type=int, default=512)
    parser.add_argument("--family-vocab-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--predictor-depth", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda-align", type=float, default=1.0)
    parser.add_argument("--lambda-mask", type=float, default=1.0)
    parser.add_argument("--lambda-target", type=float, default=0.5)
    parser.add_argument("--lambda-tile", type=float, default=0.25)
    parser.add_argument("--lambda-sigreg", type=float, default=0.09)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.official_repo is None:
        raise SystemExit("--official-repo or LEWM_OFFICIAL_REPO is required")
    args.official_repo = args.official_repo.resolve()
    args.train_manifest = args.train_manifest.resolve()
    args.holdout_manifest = args.holdout_manifest.resolve()
    args.summary_output = args.summary_output.resolve()
    args.run_dir = args.run_dir.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    process = psutil.Process()
    start_time = time.perf_counter()

    train_records = base.select_open_schematics(args.train_manifest, args.train_limit)
    holdout_records = base.select_open_schematics(args.holdout_manifest, args.holdout_limit)
    target_builder = build_target_builder(train_records, args.lib_vocab_size, args.family_vocab_size)
    args.tile_start = target_builder.tile_offset
    args.tile_end = target_builder.tile_offset + 16

    train_samples = materialize_open_samples(train_records, target_builder, args.image_size)
    holdout_samples = materialize_open_samples(holdout_records, target_builder, args.image_size)
    train_dataset = RoiGraphDataset(train_samples)
    holdout_dataset = RoiGraphDataset(holdout_samples)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=True,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CircuitLeWMSRoiGraph(
        args.official_repo,
        target_dim=target_builder.target_dim,
        image_size=args.image_size,
        embed_dim=args.embed_dim,
        predictor_depth=args.predictor_depth,
    ).to(device)
    total_parameters = sum(param.numel() for param in model.parameters())
    trainable_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    autocast_enabled = device.type == "cuda" and args.precision == "bf16"

    metrics_csv = args.run_dir / "metrics.csv"
    write_metrics_header(metrics_csv)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    epoch_summaries: list[dict[str, Any]] = []
    global_step = 0
    failed_reason = None
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_start = time.perf_counter()
            loss_sums = Counter()
            sample_count = 0
            for batch in train_loader:
                step_start = time.perf_counter()
                batch = batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                    output = model(batch["global_image"], batch["crop_image"], batch["structure_vec"])
                    losses = compute_losses(output, batch["structure_vec"], model.sigreg, args)
                if not torch.isfinite(losses["loss"]):
                    raise RuntimeError(f"non_finite_loss_at_epoch_{epoch}_step_{global_step}")
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                batch_size = int(batch["global_image"].shape[0])
                sample_count += batch_size
                global_step += 1
                step_seconds = max(time.perf_counter() - step_start, 1e-9)
                row = {
                    "epoch": epoch,
                    "step": global_step,
                    "loss": float(losses["loss"].detach().float().cpu()),
                    "align_loss": float(losses["align_loss"].detach().float().cpu()),
                    "mask_loss": float(losses["mask_loss"].detach().float().cpu()),
                    "target_loss": float(losses["target_loss"].detach().float().cpu()),
                    "tile_loss": float(losses["tile_loss"].detach().float().cpu()),
                    "sigreg_loss": float(losses["sigreg_loss"].detach().float().cpu()),
                    "samples_per_second": batch_size / step_seconds,
                }
                append_metric_row(metrics_csv, row)
                for key in ("loss", "align_loss", "mask_loss", "target_loss", "tile_loss", "sigreg_loss"):
                    loss_sums[key] += row[key] * batch_size

            retrieval = evaluate(model, holdout_loader, device, args.precision, args.tile_start, args.tile_end)
            epoch_seconds = max(time.perf_counter() - epoch_start, 1e-9)
            epoch_summary = {
                "epoch": epoch,
                "samples_seen": sample_count,
                "seconds": round(epoch_seconds, 3),
                "samples_per_second": round(sample_count / epoch_seconds, 3),
                "loss": round(loss_sums["loss"] / sample_count, 6),
                "align_loss": round(loss_sums["align_loss"] / sample_count, 6),
                "mask_loss": round(loss_sums["mask_loss"] / sample_count, 6),
                "target_loss": round(loss_sums["target_loss"] / sample_count, 6),
                "tile_loss": round(loss_sums["tile_loss"] / sample_count, 6),
                "sigreg_loss": round(loss_sums["sigreg_loss"] / sample_count, 6),
                **{key: round(value, 6) for key, value in retrieval.items()},
            }
            epoch_summaries.append(epoch_summary)
            save_checkpoint(args.checkpoint_dir / "latest.pt", model, optimizer, epoch, target_builder, args)
    except Exception as exc:
        failed_reason = f"{type(exc).__name__}: {exc}"

    elapsed_seconds = time.perf_counter() - start_time
    peak_allocated = None
    peak_reserved = None
    if device.type == "cuda":
        peak_allocated = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 3)
        peak_reserved = round(torch.cuda.max_memory_reserved() / 1024 / 1024, 3)

    final = epoch_summaries[-1] if epoch_summaries else {}
    train_final = {}
    if failed_reason is None and epoch_summaries:
        raw_train_final = evaluate(model, train_eval_loader, device, args.precision, args.tile_start, args.tile_end)
        train_final = {key.replace("holdout_", "train_"): round(value, 6) for key, value in raw_train_final.items()}
    best_by_top1 = max(epoch_summaries, key=lambda item: item["holdout_alignment_retrieval_top1"], default={})
    best_by_top5 = max(epoch_summaries, key=lambda item: item["holdout_alignment_retrieval_top5"], default={})
    retrieval_above_random = bool(
        final
        and final["holdout_alignment_retrieval_top1"] > final["holdout_random_top1"]
        and final["holdout_alignment_retrieval_top5"] > final["holdout_random_top5"]
    )
    tile_probe_above_random = bool(
        final
        and final["holdout_tile_topk_recall"] > final["holdout_tile_random_topk_recall"]
    )
    fits_vram = peak_reserved is None or peak_reserved < 24576
    status = "closed" if failed_reason is None and retrieval_above_random and tile_probe_above_random and fits_vram else "closed_with_caveats"
    if failed_reason is not None:
        status = "failed"

    summary = {
        "result_version": 1,
        "status": status,
        "generated_at_utc": utc_now(),
        "script": "wsl/lewm/scripts/train_circuit_lewm_s_roi_graph.py",
        "runtime": {
            "official_repo": base.safe_runtime_path(args.official_repo, args.official_repo),
            "run_dir": base.safe_runtime_path(args.run_dir, args.official_repo),
            "checkpoint_path": base.safe_runtime_path(args.checkpoint_dir / "latest.pt", args.official_repo),
            "metrics_csv": base.safe_runtime_path(metrics_csv, args.official_repo),
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "dataset": {
            "source_dataset": "bshada/open-schematics",
            "source_url": "https://huggingface.co/datasets/bshada/open-schematics",
            "license": "cc-by-4.0",
            "train_manifest": base.rel_repo(args.train_manifest),
            "holdout_manifest": base.rel_repo(args.holdout_manifest),
            "train_records": len(train_dataset),
            "holdout_records": len(holdout_dataset),
            "augmentation_policy": "none; deterministic ROI crop view is used for masked-crop objective",
            "raw_payload_policy": {
                "stores_raw_images": False,
                "stores_raw_schematics": False,
                "stores_checkpoint_in_git": False,
            },
        },
        "target": target_builder.metadata(),
        "model": {
            "name": "Circuit LeWM-S ROI Graph Diagnostic",
            "encoder": "ViT-tiny",
            "image_size": args.image_size,
            "patch_size": 14,
            "embed_dim": args.embed_dim,
            "predictor_depth": args.predictor_depth,
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
        },
        "training": {
            "epochs_requested": args.epochs,
            "epochs_completed": len(epoch_summaries),
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "precision": args.precision,
            "temperature": args.temperature,
        },
        "loss": {
            "lambda_align": args.lambda_align,
            "lambda_mask": args.lambda_mask,
            "lambda_target": args.lambda_target,
            "lambda_tile": args.lambda_tile,
            "lambda_sigreg": args.lambda_sigreg,
            "terms": [
                "image_to_roi_graph_structure_contrastive_alignment",
                "roi_crop_prediction",
                "roi_graph_set_target_regression",
                "tile_4x4_occupancy_prediction",
                "sigreg_gaussian_latent_regularizer",
            ],
        },
        "metrics": {
            "epoch_summaries": epoch_summaries,
            "final": final,
            "train_final": train_final,
            "best_holdout_by_top1": best_by_top1,
            "best_holdout_by_top5": best_by_top5,
            "peak_vram_mb_torch_allocated": peak_allocated,
            "peak_vram_mb_torch_reserved": peak_reserved,
            "resident_vram_mb_after_train": current_process_vram_mb(),
            "process_ram_mb_after_train": round(process.memory_info().rss / 1024 / 1024, 3),
            "elapsed_seconds": round(elapsed_seconds, 3),
        },
        "pass_checks": {
            "training_is_stable": failed_reason is None and len(epoch_summaries) == args.epochs,
            "holdout_retrieval_above_random": retrieval_above_random,
            "tile_probe_above_random": tile_probe_above_random,
            "peak_vram_fits_24gb": fits_vram,
            "checkpoint_saved_local_only": (args.checkpoint_dir / "latest.pt").exists(),
            "train_retrieval_above_random": bool(
                train_final
                and train_final["train_alignment_retrieval_top1"] > train_final["train_random_top1"]
                and train_final["train_alignment_retrieval_top5"] > train_final["train_random_top5"]
            ),
        },
        "failure": failed_reason,
    }
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "summary": base.rel_repo(args.summary_output)}, ensure_ascii=False, sort_keys=True))
    return 0 if status in {"closed", "closed_with_caveats"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

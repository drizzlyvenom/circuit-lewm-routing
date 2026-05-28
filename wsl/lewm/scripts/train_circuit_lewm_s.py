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
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image as PILImage
from PIL import ImageFile
from torch import nn
from torch.utils.data import DataLoader, Dataset


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
TRAIN_MANIFEST = REPO_ROOT / "data" / "circuit_curricula" / "train.jsonl"
HOLDOUT_MANIFEST = REPO_ROOT / "data" / "circuit_curricula" / "holdout.jsonl"
SUMMARY_PATH = REPO_ROOT / "results" / "lewm_s" / "pretrain_log.json"

IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
ImageFile.LOAD_TRUNCATED_IMAGES = True


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def rel_repo(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def safe_runtime_path(path: Path, official_repo: Path) -> str:
    path = path.resolve()
    try:
        return rel_repo(path)
    except ValueError:
        pass
    try:
        rel = path.relative_to(official_repo.resolve()).as_posix()
        return f"<lewm_official>/{rel}"
    except ValueError:
        return "<external_runtime_path>"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_open_schematics_ref(ref: str) -> int:
    parts = ref.split("/")
    if len(parts) < 7 or parts[0] != "hf:" or parts[2] != "datasets":
        raise ValueError(f"Unsupported ref: {ref}")
    if "/".join(parts[3:5]) != "bshada/open-schematics":
        raise ValueError(f"Unsupported dataset ref: {ref}")
    return int(parts[-2])


def select_open_schematics(path: Path, limit: int) -> list[dict[str, Any]]:
    records = []
    for record in read_jsonl(path):
        if record.get("source_dataset") != "bshada/open-schematics":
            continue
        if not record.get("structure", {}).get("structure_path_or_ref"):
            continue
        records.append(record)
        if len(records) >= limit:
            break
    return records


def build_component_vocab(records: list[dict[str, Any]], size: int) -> list[str]:
    counts: Counter[str] = Counter()
    for record in records:
        for component in record.get("structure", {}).get("component_list") or []:
            counts[str(component)] += 1
    return [component for component, _ in counts.most_common(size)]


def component_vector(record: dict[str, Any], vocab_index: dict[str, int]) -> torch.Tensor:
    vector = torch.zeros(len(vocab_index), dtype=torch.float32)
    for component in record.get("structure", {}).get("component_list") or []:
        idx = vocab_index.get(str(component))
        if idx is not None:
            vector[idx] = 1.0
    return vector


def component_vocab_coverage(records: list[dict[str, Any]], vocab_index: dict[str, int]) -> dict[str, Any]:
    total = 0
    covered = 0
    records_with_any = 0
    records_with_covered = 0
    for record in records:
        components = [str(component) for component in record.get("structure", {}).get("component_list") or []]
        if components:
            records_with_any += 1
        record_covered = 0
        for component in components:
            total += 1
            if component in vocab_index:
                covered += 1
                record_covered += 1
        if record_covered:
            records_with_covered += 1
    return {
        "component_mentions": total,
        "covered_component_mentions": covered,
        "component_mention_coverage": covered / total if total else None,
        "records_with_components": records_with_any,
        "records_with_covered_components": records_with_covered,
    }


def letterbox_resize(image: PILImage.Image, size: int) -> PILImage.Image:
    image = image.convert("RGB")
    ratio = min(size / image.width, size / image.height)
    new_size = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
    resized = image.resize(new_size, PILImage.Resampling.BICUBIC)
    canvas = PILImage.new("RGB", (size, size), "white")
    offset = ((size - new_size[0]) // 2, (size - new_size[1]) // 2)
    canvas.paste(resized, offset)
    return canvas


def deterministic_crop(image: PILImage.Image, sample_id: str) -> PILImage.Image:
    width, height = image.size
    crop_width = max(1, width // 2)
    crop_height = max(1, height // 2)
    bucket = int(hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:8], 16) % 4
    left = 0 if bucket % 2 == 0 else width - crop_width
    top = 0 if bucket < 2 else height - crop_height
    return image.crop((left, top, left + crop_width, top + crop_height))


def pil_to_uint8_tensor(image: PILImage.Image, size: int) -> torch.Tensor:
    resized = letterbox_resize(image, size)
    array = np.array(resized, dtype=np.uint8, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def decode_hf_image(value: Any) -> PILImage.Image:
    if isinstance(value, PILImage.Image):
        return value.convert("RGB").copy()
    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        image_path = value.get("path")
        if image_bytes:
            with PILImage.open(BytesIO(image_bytes)) as image:
                return image.convert("RGB").copy()
        if image_path:
            with PILImage.open(image_path) as image:
                return image.convert("RGB").copy()
    raise ValueError("Unsupported or empty image payload")


def uint8_to_normalized_float(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.float().div_(255.0)
    return (tensor - IMAGE_MEAN) / IMAGE_STD


def materialize_open_samples(
    records: list[dict[str, Any]],
    vocab_index: dict[str, int],
    image_size: int,
) -> list[dict[str, Any]]:
    row_to_record = {
        parse_open_schematics_ref(record["image"]["image_path_or_ref"]): record
        for record in records
    }
    if not row_to_record:
        return []
    max_index = max(row_to_record)
    samples_by_row: dict[int, dict[str, Any]] = {}
    dataset = load_dataset("bshada/open-schematics", split="train", streaming=True).cast_column(
        "image",
        HFImage(decode=False),
    )
    for row_index, row in enumerate(dataset):
        record = row_to_record.get(row_index)
        if record is not None:
            image = decode_hf_image(row["image"])
            schematic = row.get("schematic") or ""
            samples_by_row[row_index] = {
                "sample_id": record["sample_id"],
                "row_index": row_index,
                "global_image": pil_to_uint8_tensor(image, image_size),
                "crop_image": pil_to_uint8_tensor(deterministic_crop(image, record["sample_id"]), image_size),
                "structure_vec": component_vector(record, vocab_index),
                "structure_sha1": sha1_text(schematic),
                "structure_chars": len(schematic),
            }
        if row_index >= max_index:
            break
    missing = sorted(set(row_to_record) - set(samples_by_row))
    if missing:
        raise RuntimeError(f"Missing open-schematics rows: {missing[:8]}")
    return [samples_by_row[parse_open_schematics_ref(record["image"]["image_path_or_ref"])] for record in records]


class CircuitStructureDataset(Dataset[dict[str, Any]]):
    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        return {
            "sample_id": sample["sample_id"],
            "row_index": sample["row_index"],
            "global_image": uint8_to_normalized_float(sample["global_image"]),
            "crop_image": uint8_to_normalized_float(sample["crop_image"]),
            "structure_vec": sample["structure_vec"].clone(),
            "structure_sha1": sample["structure_sha1"],
            "structure_chars": sample["structure_chars"],
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
    }


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CircuitLeWMS(nn.Module):
    def __init__(
        self,
        official_repo: Path,
        component_vocab_size: int,
        image_size: int,
        embed_dim: int,
        predictor_depth: int,
    ) -> None:
        super().__init__()
        sys.path.insert(0, str(official_repo))
        import stable_pretraining as spt
        from module import MLP, SIGReg

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
            nn.Linear(component_vocab_size, 1024),
            nn.GELU(),
            nn.Linear(1024, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.crop_predictor = nn.Sequential(
            *[ResidualMLPBlock(embed_dim, 2048, dropout=0.1) for _ in range(predictor_depth)],
            nn.LayerNorm(embed_dim),
        )
        self.component_head = nn.Linear(embed_dim, component_vocab_size)
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
        component_logits = self.component_head(image_z)
        return {
            "image_z": image_z,
            "crop_z": crop_z,
            "structure_z": structure_z,
            "pred_crop_z": pred_crop_z,
            "component_logits": component_logits,
        }


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


def contrastive_loss(image_z: torch.Tensor, structure_z: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    image_norm = F.normalize(image_z, dim=-1)
    structure_norm = F.normalize(structure_z, dim=-1)
    logits = image_norm @ structure_norm.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    return loss, logits.detach()


def compute_losses(
    output: dict[str, torch.Tensor],
    structure_vec: torch.Tensor,
    sigreg_module: nn.Module,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    align_loss, _ = contrastive_loss(output["image_z"], output["structure_z"], args.temperature)
    mask_loss = F.mse_loss(F.normalize(output["pred_crop_z"], dim=-1), F.normalize(output["crop_z"].detach(), dim=-1))
    struct_loss = F.binary_cross_entropy_with_logits(output["component_logits"], structure_vec)
    sigreg_loss = output["image_z"].new_tensor(0.0)
    if args.lambda_sigreg > 0:
        sigreg_loss = sigreg_module(output["image_z"].unsqueeze(0))
    total = (
        args.lambda_align * align_loss
        + args.lambda_mask * mask_loss
        + args.lambda_struct * struct_loss
        + args.lambda_sigreg * sigreg_loss
    )
    return {
        "loss": total,
        "align_loss": align_loss,
        "mask_loss": mask_loss,
        "struct_loss": struct_loss,
        "sigreg_loss": sigreg_loss,
    }


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("global_image", "crop_image", "structure_vec", "row_indices", "structure_chars"):
        moved[key] = batch[key].to(device, non_blocking=True)
    return moved


@torch.no_grad()
def evaluate_retrieval(
    model: CircuitLeWMS,
    loader: DataLoader,
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    model.eval()
    image_vectors: list[torch.Tensor] = []
    structure_vectors: list[torch.Tensor] = []
    autocast_enabled = device.type == "cuda" and precision == "bf16"
    for batch in loader:
        batch = batch_to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            output = model(batch["global_image"], batch["crop_image"], batch["structure_vec"])
        image_vectors.append(F.normalize(output["image_z"].float(), dim=-1).cpu())
        structure_vectors.append(F.normalize(output["structure_z"].float(), dim=-1).cpu())
    image_matrix = torch.cat(image_vectors, dim=0)
    structure_matrix = torch.cat(structure_vectors, dim=0)
    sims = image_matrix @ structure_matrix.T
    labels = torch.arange(sims.size(0))
    top1 = (sims.argmax(dim=1) == labels).float().mean().item()
    top5 = (sims.topk(k=min(5, sims.size(1)), dim=1).indices == labels[:, None]).any(dim=1).float().mean().item()
    return {
        "holdout_alignment_retrieval_top1": top1,
        "holdout_alignment_retrieval_top5": top5,
        "holdout_random_top1": 1.0 / sims.size(0),
        "holdout_random_top5": min(5, sims.size(0)) / sims.size(0),
        "holdout_size": float(sims.size(0)),
    }


def write_metrics_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "step", "loss", "align_loss", "mask_loss", "struct_loss", "sigreg_loss", "samples_per_second"],
        )
        writer.writeheader()


def append_metric_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "step", "loss", "align_loss", "mask_loss", "struct_loss", "sigreg_loss", "samples_per_second"],
        )
        writer.writerow(row)


def save_checkpoint(
    path: Path,
    model: CircuitLeWMS,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    component_vocab: list[str],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "component_vocab": component_vocab,
            "config": {
                "model": "CircuitLeWMS",
                "encoder": "vit_tiny",
                "image_size": args.image_size,
                "embed_dim": args.embed_dim,
                "predictor_depth": args.predictor_depth,
            },
        },
        path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train M5 Circuit LeWM-S 5k image/structure baseline.")
    parser.add_argument("--official-repo", type=Path, default=os.environ.get("LEWM_OFFICIAL_REPO"))
    parser.add_argument("--train-manifest", type=Path, default=TRAIN_MANIFEST)
    parser.add_argument("--holdout-manifest", type=Path, default=HOLDOUT_MANIFEST)
    parser.add_argument("--summary-output", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--holdout-limit", type=int, default=512)
    parser.add_argument("--component-vocab-size", type=int, default=4096)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--predictor-depth", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda-align", type=float, default=1.0)
    parser.add_argument("--lambda-mask", type=float, default=1.0)
    parser.add_argument("--lambda-struct", type=float, default=0.5)
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

    train_records = select_open_schematics(args.train_manifest, args.train_limit)
    holdout_records = select_open_schematics(args.holdout_manifest, args.holdout_limit)
    component_vocab = build_component_vocab(train_records, args.component_vocab_size)
    vocab_index = {component: idx for idx, component in enumerate(component_vocab)}
    train_vocab_coverage = component_vocab_coverage(train_records, vocab_index)
    holdout_vocab_coverage = component_vocab_coverage(holdout_records, vocab_index)
    train_samples = materialize_open_samples(train_records, vocab_index, args.image_size)
    holdout_samples = materialize_open_samples(holdout_records, vocab_index, args.image_size)
    train_dataset = CircuitStructureDataset(train_samples)
    holdout_dataset = CircuitStructureDataset(holdout_samples)

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
    model = CircuitLeWMS(
        args.official_repo,
        component_vocab_size=len(component_vocab),
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
                samples_per_second = batch_size / step_seconds
                row = {
                    "epoch": epoch,
                    "step": global_step,
                    "loss": float(losses["loss"].detach().float().cpu()),
                    "align_loss": float(losses["align_loss"].detach().float().cpu()),
                    "mask_loss": float(losses["mask_loss"].detach().float().cpu()),
                    "struct_loss": float(losses["struct_loss"].detach().float().cpu()),
                    "sigreg_loss": float(losses["sigreg_loss"].detach().float().cpu()),
                    "samples_per_second": samples_per_second,
                }
                append_metric_row(metrics_csv, row)
                for key in ("loss", "align_loss", "mask_loss", "struct_loss", "sigreg_loss"):
                    loss_sums[key] += row[key] * batch_size

            retrieval = evaluate_retrieval(model, holdout_loader, device, args.precision)
            epoch_seconds = max(time.perf_counter() - epoch_start, 1e-9)
            epoch_summary = {
                "epoch": epoch,
                "samples_seen": sample_count,
                "seconds": round(epoch_seconds, 3),
                "samples_per_second": round(sample_count / epoch_seconds, 3),
                "loss": round(loss_sums["loss"] / sample_count, 6),
                "align_loss": round(loss_sums["align_loss"] / sample_count, 6),
                "mask_loss": round(loss_sums["mask_loss"] / sample_count, 6),
                "struct_loss": round(loss_sums["struct_loss"] / sample_count, 6),
                "sigreg_loss": round(loss_sums["sigreg_loss"] / sample_count, 6),
                **{key: round(value, 6) for key, value in retrieval.items()},
            }
            epoch_summaries.append(epoch_summary)
            save_checkpoint(args.checkpoint_dir / "latest.pt", model, optimizer, epoch, component_vocab, args)
    except Exception as exc:
        failed_reason = f"{type(exc).__name__}: {exc}"

    elapsed_seconds = time.perf_counter() - start_time
    peak_allocated = None
    peak_reserved = None
    if device.type == "cuda":
        peak_allocated = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 3)
        peak_reserved = round(torch.cuda.max_memory_reserved() / 1024 / 1024, 3)

    final_retrieval = epoch_summaries[-1] if epoch_summaries else {}
    retrieval_above_random = bool(
        final_retrieval
        and final_retrieval["holdout_alignment_retrieval_top1"] > final_retrieval["holdout_random_top1"]
        and final_retrieval["holdout_alignment_retrieval_top5"] > final_retrieval["holdout_random_top5"]
    )
    fits_vram = peak_reserved is None or peak_reserved < 24576
    status = "closed" if failed_reason is None and retrieval_above_random and fits_vram else "closed_with_caveats"
    if failed_reason is not None:
        status = "failed"

    summary = {
        "result_version": 1,
        "status": status,
        "generated_at_utc": utc_now(),
        "script": "wsl/lewm/scripts/train_circuit_lewm_s.py",
        "runtime": {
            "official_repo": safe_runtime_path(args.official_repo, args.official_repo),
            "run_dir": safe_runtime_path(args.run_dir, args.official_repo),
            "checkpoint_path": safe_runtime_path(args.checkpoint_dir / "latest.pt", args.official_repo),
            "metrics_csv": safe_runtime_path(metrics_csv, args.official_repo),
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "dataset": {
            "source_dataset": "bshada/open-schematics",
            "source_url": "https://huggingface.co/datasets/bshada/open-schematics",
            "license": "cc-by-4.0",
            "train_manifest": rel_repo(args.train_manifest),
            "holdout_manifest": rel_repo(args.holdout_manifest),
            "train_records": len(train_dataset),
            "holdout_records": len(holdout_dataset),
            "component_vocab_size": len(component_vocab),
            "train_component_vocab_coverage": train_vocab_coverage,
            "holdout_component_vocab_coverage": holdout_vocab_coverage,
            "augmentation_policy": "none; deterministic crop view is used only for masked-crop objective",
            "raw_payload_policy": {
                "stores_raw_images": False,
                "stores_raw_schematics": False,
                "stores_checkpoint_in_git": False,
            },
        },
        "model": {
            "name": "Circuit LeWM-S",
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
            "lambda_struct": args.lambda_struct,
            "lambda_sigreg": args.lambda_sigreg,
            "terms": [
                "image_to_structure_contrastive_alignment",
                "deterministic_crop_prediction",
                "component_multilabel_prediction",
                "sigreg_gaussian_latent_regularizer",
            ],
        },
        "metrics": {
            "epoch_summaries": epoch_summaries,
            "final": final_retrieval,
            "peak_vram_mb_torch_allocated": peak_allocated,
            "peak_vram_mb_torch_reserved": peak_reserved,
            "resident_vram_mb_after_train": current_process_vram_mb(),
            "process_ram_mb_after_train": round(process.memory_info().rss / 1024 / 1024, 3),
            "elapsed_seconds": round(elapsed_seconds, 3),
        },
        "pass_checks": {
            "training_is_stable": failed_reason is None and len(epoch_summaries) == args.epochs,
            "holdout_retrieval_above_random": retrieval_above_random,
            "peak_vram_fits_24gb": fits_vram,
            "checkpoint_saved_local_only": (args.checkpoint_dir / "latest.pt").exists(),
        },
        "failure": failed_reason,
    }
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "summary": rel_repo(args.summary_output)}, ensure_ascii=False, sort_keys=True))
    return 0 if status in {"closed", "closed_with_caveats"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

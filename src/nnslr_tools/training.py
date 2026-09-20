"""Reader-baseline training for NNSLR.

This module is isolated from the core/data CLI import surface. PyTorch,
torchvision and Pillow are imported only after an explicit training request.
prepare_reader_training is stdlib-only and can validate the exact dataset/split
that would be consumed by a GPU run.

The first baseline trains only the sign reader on reviewed bounding-box crops.
It does not train a detector and it does not produce an on-vehicle bundle.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from nnslr_tools.annotations import load_jsonl, validate_dataset
from nnslr_tools.splits import validate_splits

TRAINING_SCHEMA_VERSION = 1
ELIGIBLE_REVIEW_STATES = frozenset({"accepted", "corrected"})
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reader_label(row: dict[str, Any]) -> str | None:
    """Return the classification label for a reviewed sign crop."""
    if row.get("review_state") not in ELIGIBLE_REVIEW_STATES:
        return None
    bbox = row.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None

    family = row.get("sign_family")
    if family == "not_a_sign":
        return None

    value_state = row.get("value_state")
    value = row.get("value_kph")
    if value_state == "read" and isinstance(value, int) and not isinstance(value, bool):
        return f"{family}:{value}"
    if family in {"cancellation", "unreadable", "other_sign"}:
        return str(family)
    if value_state in {"unreadable", "unknown", "not_applicable"}:
        return f"{family}:{value_state}"
    return None


def _safe_image_path(data_root: Path, image_path: str) -> Path:
    root = data_root.resolve()
    candidate = (root / image_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"unsafe_image_path: {image_path}")
    return candidate


def prepare_reader_training(
    dataset_path: Path,
    split_path: Path,
    data_root: Path,
    *,
    dataset_kind: str = "gold",
) -> dict[str, Any]:
    """Validate inputs and build the immutable reader-training plan.

    No torch/Pillow import and no GPU access happens here.
    """
    dataset_path = Path(dataset_path)
    split_path = Path(split_path)
    data_root = Path(data_root)

    rows = load_jsonl(dataset_path)
    validation = validate_dataset(rows, data_root, dataset_kind=dataset_kind)
    if not validation.get("valid"):
        reasons = sorted({str(e.get("reason", "dataset_invalid")) for e in validation.get("errors", [])})
        raise ValueError("dataset_invalid: " + ",".join(reasons))

    split_doc = json.loads(split_path.read_text(encoding="utf-8"))
    split_errors = validate_splits(
        rows,
        split_doc,
        validation["dataset_sha256"],
        dataset_kind=dataset_kind,
    )
    if split_errors:
        reasons = sorted({str(e.get("reason", "split_invalid")) for e in split_errors})
        raise ValueError("split_invalid: " + ",".join(reasons))

    assignments = split_doc.get("assignments")
    if not isinstance(assignments, dict):
        raise ValueError("split_invalid: assignments")

    records: list[dict[str, Any]] = []
    skipped = Counter()
    for row in rows:
        label = _reader_label(row)
        if label is None:
            skipped["not_reader_eligible"] += 1
            continue
        annotation_id = row.get("annotation_id")
        split_name = assignments.get(annotation_id)
        if not isinstance(split_name, str):
            skipped["missing_split"] += 1
            continue
        image_path = row.get("image_path")
        if not isinstance(image_path, str):
            skipped["missing_image_path"] += 1
            continue
        image = _safe_image_path(data_root, image_path)
        if not image.is_file():
            raise ValueError(f"missing_image: {image_path}")
        records.append({
            "annotation_id": annotation_id,
            "split": split_name,
            "image_path": image_path,
            "image_sha256": row.get("image_sha256"),
            "bbox_xyxy": row["bbox_xyxy"],
            "label": label,
            "sign_family": row.get("sign_family"),
            "value_state": row.get("value_state"),
            "value_kph": row.get("value_kph"),
            "route_id": row.get("route_id"),
            "segment_index": row.get("segment_index"),
            "decoded_frame_index": row.get("decoded_frame_index"),
        })

    train_labels = sorted({r["label"] for r in records if r["split"] == TRAIN_SPLIT})
    val_labels = sorted({r["label"] for r in records if r["split"] == VALIDATION_SPLIT})
    unseen_validation = sorted(set(val_labels) - set(train_labels))
    counts_by_split = Counter(r["split"] for r in records)
    counts_by_class = Counter(r["label"] for r in records if r["split"] == TRAIN_SPLIT)

    trainable = (
        counts_by_split[TRAIN_SPLIT] > 0
        and counts_by_split[VALIDATION_SPLIT] > 0
        and len(train_labels) >= 2
        and not unseen_validation
    )

    limitations = [
        "reader_only_no_detector",
        "bbox_crops_require_reviewed_localization",
        "no_runtime_bundle",
    ]
    if dataset_kind == "training-candidate":
        limitations.append("model_review_candidates_are_not_human_gold")

    return {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "task": "reader",
        "dataset_kind": dataset_kind,
        "dataset_path": str(dataset_path),
        "dataset_sha256": validation["dataset_sha256"],
        "split_path": str(split_path),
        "split_sha256": _sha256(split_path),
        "classes": train_labels,
        "records": records,
        "counts_by_split": dict(sorted(counts_by_split.items())),
        "train_counts_by_class": dict(sorted(counts_by_class.items())),
        "skipped": dict(sorted(skipped.items())),
        "unseen_validation_labels": unseen_validation,
        "trainable": trainable,
        "limitations": limitations,
    }


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    input_size: int = 160
    num_workers: int = 4
    seed: int = 0
    device: str = "cuda"
    pretrained: bool = False
    amp: bool = True


def _require_training_stack():
    try:
        import torch
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    except ImportError as exc:
        raise ValueError(
            "training_dependencies_missing: install a V100-compatible PyTorch/torchvision "
            "environment plus Pillow; training dependencies are intentionally not part of "
            "the core CPU profile"
        ) from exc
    return torch, Image, DataLoader, Dataset, transforms, MobileNet_V3_Small_Weights, mobilenet_v3_small


def train_reader(
    plan: dict[str, Any],
    data_root: Path,
    output_dir: Path,
    config: TrainConfig,
) -> dict[str, Any]:
    """Execute the first reader baseline after explicit compute opt-in."""
    if not plan.get("trainable"):
        raise ValueError(
            "training_plan_not_trainable: need >=2 train classes, non-empty train/validation, "
            "and no validation-only classes"
        )

    torch, Image, DataLoader, Dataset, transforms, Weights, mobilenet_v3_small = _require_training_stack()

    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda_unavailable")
    device = torch.device(config.device)

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    classes = list(plan["classes"])
    class_to_index = {name: idx for idx, name in enumerate(classes)}
    root = Path(data_root).resolve()

    class CropDataset(Dataset):
        def __init__(self, records, *, training: bool):
            self.records = records
            pipeline = [transforms.Resize((config.input_size, config.input_size))]
            if training:
                pipeline.extend([
                    transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.1),
                    transforms.RandomAffine(degrees=4, translate=(.03, .03), scale=(.95, 1.05)),
                ])
            pipeline.extend([
                transforms.ToTensor(),
                transforms.Normalize(mean=(.485, .456, .406), std=(.229, .224, .225)),
            ])
            self.transform = transforms.Compose(pipeline)

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            row = self.records[index]
            image_path = _safe_image_path(root, row["image_path"])
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                x1, y1, x2, y2 = [int(v) for v in row["bbox_xyxy"]]
                crop = image.crop((x1, y1, x2, y2))
            return self.transform(crop), class_to_index[row["label"]]

    train_records = [r for r in plan["records"] if r["split"] == TRAIN_SPLIT]
    val_records = [r for r in plan["records"] if r["split"] == VALIDATION_SPLIT]
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        CropDataset(train_records, training=True),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    val_loader = DataLoader(
        CropDataset(val_records, training=False),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    weights = Weights.DEFAULT if config.pretrained else None
    model = mobilenet_v3_small(weights=weights)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(classes))
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()

    use_amp = bool(config.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    plan_path = output_dir / "training-plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    history = []
    best_accuracy = -1.0
    best_path = output_dir / "reader-best.pt"

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        train_seen = 0
        train_correct = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(inputs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach()) * targets.numel()
            train_seen += targets.numel()
            train_correct += int((logits.argmax(1) == targets).sum())

        model.eval()
        val_loss = 0.0
        val_seen = 0
        val_correct = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(inputs)
                    loss = criterion(logits, targets)
                val_loss += float(loss.detach()) * targets.numel()
                val_seen += targets.numel()
                val_correct += int((logits.argmax(1) == targets).sum())

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(train_seen, 1),
            "train_accuracy": train_correct / max(train_seen, 1),
            "validation_loss": val_loss / max(val_seen, 1),
            "validation_accuracy": val_correct / max(val_seen, 1),
        }
        history.append(epoch_result)

        if epoch_result["validation_accuracy"] > best_accuracy:
            best_accuracy = epoch_result["validation_accuracy"]
            torch.save({
                "format_version": 1,
                "architecture": "mobilenet_v3_small",
                "task": "reader",
                "classes": classes,
                "input_size": config.input_size,
                "state_dict": model.state_dict(),
                "dataset_sha256": plan["dataset_sha256"],
                "split_sha256": plan["split_sha256"],
                "epoch": epoch + 1,
                "validation_accuracy": best_accuracy,
            }, best_path)

    gpu = None
    if device.type == "cuda":
        gpu = {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        }

    result = {
        "task": "reader",
        "architecture": "mobilenet_v3_small",
        "classes": classes,
        "best_validation_accuracy": best_accuracy,
        "checkpoint": str(best_path),
        "training_plan": str(plan_path),
        "device": str(device),
        "gpu": gpu,
        "torch_version": torch.__version__,
        "config": config.__dict__,
        "history": history,
        "limitations": plan["limitations"],
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result

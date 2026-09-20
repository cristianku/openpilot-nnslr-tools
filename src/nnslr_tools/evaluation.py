"""Offline evaluation for NNSLR reader and detector baselines.

Evaluation consumes the same hash-bound dataset/split plan used by training.
Torch is imported only by an explicit non-dry-run evaluation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any

from nnslr_tools.training import (
    _require_detector_stack,
    _require_training_stack,
    _safe_image_path,
)


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else intersection / union


def match_detections(
    predicted: list[dict[str, Any]],
    truth: list[list[float]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Greedy score-ordered one-to-one matching for a single detector class."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("invalid_iou_threshold")
    ordered = sorted(predicted, key=lambda p: float(p["score"]), reverse=True)
    unmatched = set(range(len(truth)))
    matches = []
    false_positives = []

    for prediction in ordered:
        box = prediction["box"]
        candidates = [
            (box_iou(box, truth[index]), index)
            for index in unmatched
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_index >= 0 and best_iou >= iou_threshold:
            unmatched.remove(best_index)
            matches.append({
                "score": float(prediction["score"]),
                "iou": best_iou,
                "truth_index": best_index,
            })
        else:
            false_positives.append({
                "score": float(prediction["score"]),
                "best_iou": best_iou,
            })

    return {
        "true_positives": len(matches),
        "false_positives": len(false_positives),
        "false_negatives": len(unmatched),
        "matches": matches,
        "false_positive_details": false_positives,
    }


def _load_checkpoint(torch, checkpoint_path: Path, expected_task: str) -> dict[str, Any]:
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("task") != expected_task:
        raise ValueError(f"checkpoint_task_mismatch: expected {expected_task}")
    if not isinstance(payload.get("state_dict"), dict):
        raise ValueError("checkpoint_missing_state_dict")
    return payload


def evaluate_reader(
    plan: dict[str, Any],
    checkpoint_path: Path,
    data_root: Path,
    *,
    partition: str = "test",
    device_name: str = "cuda",
    batch_size: int = 64,
    num_workers: int = 4,
) -> dict[str, Any]:
    torch, Image, DataLoader, Dataset, transforms, _, mobilenet_v3_small = _require_training_stack()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda_unavailable")
    device = torch.device(device_name)
    checkpoint = _load_checkpoint(torch, checkpoint_path, "reader")

    classes = checkpoint.get("classes")
    if not isinstance(classes, list) or not classes or not all(isinstance(v, str) for v in classes):
        raise ValueError("checkpoint_invalid_classes")
    if checkpoint.get("dataset_sha256") != plan.get("dataset_sha256"):
        raise ValueError("checkpoint_dataset_mismatch")
    if checkpoint.get("split_sha256") != plan.get("split_sha256"):
        raise ValueError("checkpoint_split_mismatch")

    records = [r for r in plan["records"] if r["split"] == partition]
    if not records:
        raise ValueError(f"empty_evaluation_partition: {partition}")
    input_size = int(checkpoint.get("input_size", 160))
    root = Path(data_root).resolve()

    class ReaderDataset(Dataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, index):
            row = records[index]
            path = _safe_image_path(root, row["image_path"])
            with Image.open(path) as source:
                image = source.convert("RGB")
                x1, y1, x2, y2 = [int(v) for v in row["bbox_xyxy"]]
                crop = image.crop((x1, y1, x2, y2))
            transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(.485, .456, .406), std=(.229, .224, .225)),
            ])
            return transform(crop), row["label"], index

    loader = DataLoader(
        ReaderDataset(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(classes))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()

    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    support = Counter()
    correct = Counter()
    total = 0
    total_correct = 0

    samples = []
    with torch.no_grad():
        for inputs, truth_labels, record_indices in loader:
            inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(1).cpu().tolist()
            confidences = probabilities.max(1).values.cpu().tolist()
            for predicted_index, confidence, truth_label, record_index in zip(
                predictions, confidences, truth_labels, record_indices.tolist()
            ):
                predicted_label = classes[predicted_index]
                support[truth_label] += 1
                confusion[truth_label][predicted_label] += 1
                total += 1
                is_correct = predicted_label == truth_label
                if is_correct:
                    correct[truth_label] += 1
                    total_correct += 1
                row = records[record_index]
                samples.append({
                    "annotation_id": row["annotation_id"],
                    "image_path": row["image_path"],
                    "truth": truth_label,
                    "predicted": predicted_label,
                    "confidence": float(confidence),
                    "correct": is_correct,
                })

    per_class = {
        label: {
            "support": support[label],
            "correct": correct[label],
            "accuracy": correct[label] / support[label],
        }
        for label in sorted(support)
    }
    unseen_truth = sorted(set(support) - set(classes))
    return {
        "schema_version": 1,
        "task": "reader",
        "partition": partition,
        "sample_count": total,
        "accuracy": total_correct / total,
        "classes": classes,
        "unseen_truth_labels": unseen_truth,
        "per_class": per_class,
        "confusion": {
            truth: dict(sorted(predictions.items()))
            for truth, predictions in sorted(confusion.items())
        },
        "samples": samples,
        "checkpoint": str(checkpoint_path),
        "dataset_sha256": plan["dataset_sha256"],
        "split_sha256": plan["split_sha256"],
        "device": str(device),
    }


def evaluate_detector(
    plan: dict[str, Any],
    checkpoint_path: Path,
    data_root: Path,
    *,
    partition: str = "test",
    device_name: str = "cuda",
    score_threshold: float = 0.25,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    if not 0 <= score_threshold <= 1:
        raise ValueError("invalid_score_threshold")
    if not 0 < iou_threshold <= 1:
        raise ValueError("invalid_iou_threshold")

    torch, Image, _, _, TF, _, ssdlite320_mobilenet_v3_large = _require_detector_stack()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda_unavailable")
    device = torch.device(device_name)
    checkpoint = _load_checkpoint(torch, checkpoint_path, "detector")
    if checkpoint.get("dataset_sha256") != plan.get("dataset_sha256"):
        raise ValueError("checkpoint_dataset_mismatch")
    if checkpoint.get("split_sha256") != plan.get("split_sha256"):
        raise ValueError("checkpoint_split_mismatch")

    frames = [r for r in plan["frames"] if r["split"] == partition]
    if not frames:
        raise ValueError(f"empty_evaluation_partition: {partition}")
    root = Path(data_root).resolve()

    model = ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=None,
        num_classes=2,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()

    totals = Counter()
    image_results = []
    with torch.no_grad():
        for frame in frames:
            path = _safe_image_path(root, frame["image_path"])
            with Image.open(path) as source:
                image = source.convert("RGB")
                tensor = TF.pil_to_tensor(image).float().div_(255.0).to(device)

            prediction = model([tensor])[0]
            boxes = prediction["boxes"].detach().cpu().tolist()
            scores = prediction["scores"].detach().cpu().tolist()
            labels = prediction["labels"].detach().cpu().tolist()
            selected = [
                {"box": box, "score": score}
                for box, score, label in zip(boxes, scores, labels)
                if label == 1 and score >= score_threshold
            ]
            matched = match_detections(selected, frame["boxes"], iou_threshold=iou_threshold)
            totals["tp"] += matched["true_positives"]
            totals["fp"] += matched["false_positives"]
            totals["fn"] += matched["false_negatives"]
            image_results.append({
                "image_path": frame["image_path"],
                "image_sha256": frame["image_sha256"],
                "route_id": frame["route_id"],
                "segment_index": frame["segment_index"],
                "truth_boxes": frame["boxes"],
                "predictions": selected,
                "truth_count": len(frame["boxes"]),
                "prediction_count": len(selected),
                "true_positives": matched["true_positives"],
                "false_positives": matched["false_positives"],
                "false_negatives": matched["false_negatives"],
                "matches": matched["matches"],
                "false_positive_details": matched["false_positive_details"],
            })

    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "schema_version": 1,
        "task": "detector",
        "partition": partition,
        "image_count": len(frames),
        "truth_box_count": tp + fn,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision_iou": precision,
        "recall_iou": recall,
        "f1_iou": f1,
        "false_positives_per_image": fp / len(frames),
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "checkpoint": str(checkpoint_path),
        "dataset_sha256": plan["dataset_sha256"],
        "split_sha256": plan["split_sha256"],
        "device": str(device),
        "images": image_results,
        "limitations": [
            "single_iou_threshold_not_coco_map",
            "false_positives_per_hour_requires_timed_replay",
        ],
    }


def write_evaluation(report: dict[str, Any], output: Path | None) -> None:
    if output is None:
        return
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

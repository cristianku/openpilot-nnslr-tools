"""Hard-example mining from immutable offline evaluation reports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def mine_hard_examples(report: dict[str, Any]) -> list[dict[str, Any]]:
    task = report.get("task")
    if task == "reader":
        samples = report.get("samples")
        if not isinstance(samples, list):
            raise ValueError("evaluation_missing_samples")
        mined = []
        for sample in samples:
            if not isinstance(sample, dict) or sample.get("correct") is not False:
                continue
            mined.append({
                "task": "reader",
                "kind": "misclassification",
                "annotation_id": sample.get("annotation_id"),
                "image_path": sample.get("image_path"),
                "truth": sample.get("truth"),
                "predicted": sample.get("predicted"),
                "confidence": sample.get("confidence"),
                "partition": report.get("partition"),
            })
        return sorted(mined, key=lambda r: (str(r.get("image_path")), str(r.get("annotation_id"))))

    if task == "detector":
        images = report.get("images")
        if not isinstance(images, list):
            raise ValueError("evaluation_missing_images")
        mined = []
        for image in images:
            if not isinstance(image, dict):
                continue
            fp = image.get("false_positives")
            fn = image.get("false_negatives")
            if type(fp) is not int or type(fn) is not int:
                raise ValueError("evaluation_invalid_detection_counts")
            if fp == 0 and fn == 0:
                continue
            kinds = []
            if fp:
                kinds.append("false_positive")
            if fn:
                kinds.append("false_negative")
            mined.append({
                "task": "detector",
                "kind": "+".join(kinds),
                "image_path": image.get("image_path"),
                "image_sha256": image.get("image_sha256"),
                "route_id": image.get("route_id"),
                "segment_index": image.get("segment_index"),
                "false_positives": fp,
                "false_negatives": fn,
                "truth_boxes": image.get("truth_boxes", []),
                "predictions": image.get("predictions", []),
                "partition": report.get("partition"),
                "score_threshold": report.get("score_threshold"),
                "iou_threshold": report.get("iou_threshold"),
            })
        return sorted(
            mined,
            key=lambda r: (
                -(int(r["false_positives"]) + int(r["false_negatives"])),
                str(r.get("image_path")),
            ),
        )

    raise ValueError(f"unsupported_evaluation_task: {task}")


def mine_file(evaluation_path: Path, output_path: Path) -> dict[str, Any]:
    evaluation_path = Path(evaluation_path)
    output_path = Path(output_path)
    report = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("invalid_evaluation_report")
    examples = mine_hard_examples(report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(example, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            for example in examples
        ),
        encoding="utf-8",
    )
    return {
        "valid": True,
        "task": report.get("task"),
        "evaluation": str(evaluation_path),
        "output": str(output_path),
        "hard_example_count": len(examples),
        "partition": report.get("partition"),
    }

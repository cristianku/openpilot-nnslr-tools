"""CPU reference runtime for a verified NNSLR ONNX model bundle.

This runner exists for offline parity/replay. It is not the target Comma
backend and makes no claim about target-device performance.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nnslr_tools.bundle import verify_bundle
from nnslr_tools.replay import decode_reader_label


def _require_reference_stack():
    try:
        import numpy as np
        import onnxruntime as ort
        from PIL import Image
    except ImportError as exc:
        raise ValueError(
            "reference_runtime_dependencies_missing: install numpy, Pillow and onnxruntime"
        ) from exc
    return np, ort, Image


def _softmax(np, logits):
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    total = np.sum(exp)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("reader_nonfinite_softmax")
    return exp / total


class ReferenceBundleRunner:
    """Verified ONNX detector+reader bundle executed through ONNX Runtime CPU."""

    def __init__(self, bundle_dir: Path):
        self.bundle_dir = Path(bundle_dir).resolve()
        self.bundle_report = verify_bundle(self.bundle_dir)
        self.config = json.loads((self.bundle_dir / "config.json").read_text(encoding="utf-8"))
        self.classes_doc = json.loads((self.bundle_dir / "classes.json").read_text(encoding="utf-8"))
        self.detector_contract = json.loads(
            (self.bundle_dir / "detector.contract.json").read_text(encoding="utf-8")
        )
        self.reader_contract = json.loads(
            (self.bundle_dir / "reader.contract.json").read_text(encoding="utf-8")
        )

        np, ort, Image = _require_reference_stack()
        self.np = np
        self.Image = Image
        self.detector = ort.InferenceSession(
            str(self.bundle_dir / "detector.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self.reader = ort.InferenceSession(
            str(self.bundle_dir / "reader.onnx"),
            providers=["CPUExecutionProvider"],
        )

        self._detector_shape = self._fixed_image_shape(
            self.detector_contract["input"]["shape"],
            "detector",
        )
        self._reader_shape = self._fixed_image_shape(
            self.reader_contract["input"]["shape"],
            "reader",
        )
        self.reader_classes = self.reader_contract["output"]["classes"]
        if not isinstance(self.reader_classes, list) or not self.reader_classes:
            raise ValueError("bundle_reader_classes_invalid")

        detector_input = self.detector.get_inputs()
        reader_input = self.reader.get_inputs()
        if len(detector_input) != 1 or detector_input[0].name != "images":
            raise ValueError("bundle_detector_onnx_input_mismatch")
        if len(reader_input) != 1 or reader_input[0].name != "images":
            raise ValueError("bundle_reader_onnx_input_mismatch")

    @staticmethod
    def _fixed_image_shape(shape, task: str) -> tuple[int, int]:
        if (
            not isinstance(shape, list)
            or len(shape) != 4
            or shape[0] != 1
            or shape[1] != 3
            or type(shape[2]) is not int
            or type(shape[3]) is not int
            or shape[2] <= 0
            or shape[3] <= 0
        ):
            raise ValueError(f"bundle_{task}_input_shape_not_static")
        return shape[2], shape[3]

    def _detector_tensor(self, image):
        np = self.np
        expected_h, expected_w = self._detector_shape
        if image.size != (expected_w, expected_h):
            raise ValueError(
                f"detector_native_geometry_mismatch: expected {expected_w}x{expected_h}, "
                f"got {image.width}x{image.height}"
            )
        array = np.asarray(image, dtype=np.float32) / np.float32(255.0)
        if array.shape != (expected_h, expected_w, 3):
            raise ValueError("detector_rgb_shape_mismatch")
        return np.transpose(array, (2, 0, 1))[None, ...]

    def _reader_tensor(self, crop):
        np = self.np
        target_h, target_w = self._reader_shape
        resized = crop.resize((target_w, target_h), self.Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / np.float32(255.0)
        mean = np.asarray(
            self.reader_contract["input"]["normalization"]["mean"],
            dtype=np.float32,
        )
        std = np.asarray(
            self.reader_contract["input"]["normalization"]["std"],
            dtype=np.float32,
        )
        array = (array - mean) / std
        return np.transpose(array, (2, 0, 1))[None, ...]

    def run_image(
        self,
        image_path: Path,
        *,
        detector_threshold: float | None = None,
        reader_threshold: float | None = None,
        max_detections: int = 20,
    ) -> list[dict[str, Any]]:
        if type(max_detections) is not int or max_detections < 1:
            raise ValueError("invalid_max_detections")

        reference_thresholds = self.config.get("reference_thresholds", {})
        if detector_threshold is None:
            detector_threshold = float(reference_thresholds.get("detector_score", .25))
        if reader_threshold is None:
            reader_threshold = float(reference_thresholds.get("reader_score", 0.0))
        if not 0 <= detector_threshold <= 1 or not 0 <= reader_threshold <= 1:
            raise ValueError("invalid_confidence_threshold")

        with self.Image.open(image_path) as source:
            image = source.convert("RGB")
            detector_input = self._detector_tensor(image)
            boxes, scores, labels = self.detector.run(
                ["boxes", "scores", "labels"],
                {"images": detector_input},
            )

            detections = []
            for box, det_score, detector_label in zip(boxes, scores, labels):
                if int(detector_label) != 1 or float(det_score) < detector_threshold:
                    continue
                if len(detections) >= max_detections:
                    break
                x1, y1, x2, y2 = [float(v) for v in box]
                x1i = max(0, min(image.width - 1, int(x1)))
                y1i = max(0, min(image.height - 1, int(y1)))
                x2i = max(x1i + 1, min(image.width, int(x2 + .999)))
                y2i = max(y1i + 1, min(image.height, int(y2 + .999)))
                crop = image.crop((x1i, y1i, x2i, y2i))
                logits = self.reader.run(
                    ["logits"],
                    {"images": self._reader_tensor(crop)},
                )[0][0]
                probabilities = _softmax(self.np, logits)
                class_index = int(self.np.argmax(probabilities))
                classification_score = float(probabilities[class_index])
                label = self.reader_classes[class_index]
                decoded = decode_reader_label(label)
                if classification_score < reader_threshold:
                    decoded = {
                        "sign_family": decoded["sign_family"],
                        "value_state": "unknown",
                        "value_kph": None,
                        "supported_domain": decoded["supported_domain"],
                    }
                detections.append({
                    "bbox_xyxy": [x1i, y1i, x2i, y2i],
                    "detection_score": float(det_score),
                    "classification_score": classification_score,
                    "reader_label": label,
                    **decoded,
                })

        return detections


def reference_runtime_info(bundle_dir: Path) -> dict[str, Any]:
    """Validate a bundle and return static runtime requirements without ORT."""
    bundle_dir = Path(bundle_dir).resolve()
    report = verify_bundle(bundle_dir)
    config = json.loads((bundle_dir / "config.json").read_text(encoding="utf-8"))
    return {
        **report,
        "detector_input": config["detector"]["input"],
        "reader_input": config["reader"]["input"],
        "reference_thresholds": config["reference_thresholds"],
        "runtime": "onnxruntime_cpu_reference",
        "target_compatible": False,
    }

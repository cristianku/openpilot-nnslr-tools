"""Immutable NNSLR detector+reader model bundle packaging and verification."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


BUNDLE_SCHEMA_VERSION = 1
REQUIRED_FILES = (
    "detector.onnx",
    "reader.onnx",
    "detector.contract.json",
    "reader.contract.json",
    "config.json",
    "classes.json",
    "model_card.json",
    "capability.json",
    "sha256sums.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid_json_object: {path}")
    return payload


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _validated_contract(onnx_path: Path, contract_path: Path, task: str) -> dict[str, Any]:
    contract = _load_json(contract_path)
    if contract.get("task") != task:
        raise ValueError(f"contract_task_mismatch: expected {task}")
    if contract.get("onnx_sha256") != _sha256(onnx_path):
        raise ValueError(f"contract_onnx_hash_mismatch: {task}")
    verification = contract.get("verification")
    if not isinstance(verification, dict) or verification.get("passed") is not True:
        raise ValueError(f"onnx_not_verified: {task}")
    dataset_sha = contract.get("dataset_sha256")
    split_sha = contract.get("split_sha256")
    if not isinstance(dataset_sha, str) or len(dataset_sha) != 64:
        raise ValueError(f"contract_missing_dataset_hash: {task}")
    if not isinstance(split_sha, str) or len(split_sha) != 64:
        raise ValueError(f"contract_missing_split_hash: {task}")
    return contract


def package_model(
    detector_onnx: Path,
    detector_contract: Path,
    reader_onnx: Path,
    reader_contract: Path,
    output_dir: Path,
) -> dict[str, Any]:
    detector_onnx = Path(detector_onnx)
    reader_onnx = Path(reader_onnx)
    detector_contract = Path(detector_contract)
    reader_contract = Path(reader_contract)
    for path in (detector_onnx, reader_onnx, detector_contract, reader_contract):
        if not path.is_file():
            raise ValueError(f"bundle_input_missing: {path}")

    detector = _validated_contract(detector_onnx, detector_contract, "detector")
    reader = _validated_contract(reader_onnx, reader_contract, "reader")
    if (
        detector["dataset_sha256"] != reader["dataset_sha256"]
        or detector["split_sha256"] != reader["split_sha256"]
    ):
        raise ValueError("bundle_provenance_mismatch")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    shutil.copy2(detector_onnx, output_dir / "detector.onnx")
    shutil.copy2(reader_onnx, output_dir / "reader.onnx")
    shutil.copy2(detector_contract, output_dir / "detector.contract.json")
    shutil.copy2(reader_contract, output_dir / "reader.contract.json")

    reader_output = reader.get("output")
    classes = reader_output.get("classes") if isinstance(reader_output, dict) else None
    if not isinstance(classes, list) or not classes or not all(isinstance(v, str) for v in classes):
        raise ValueError("reader_contract_missing_classes")

    config = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "feature_id": "nnslr-speed-vision",
        "dataset_sha256": detector["dataset_sha256"],
        "split_sha256": detector["split_sha256"],
        "detector": {
            "file": "detector.onnx",
            "contract": "detector.contract.json",
            "input": detector["input"],
            "outputs": detector["outputs"],
        },
        "reader": {
            "file": "reader.onnx",
            "contract": "reader.contract.json",
            "input": reader["input"],
            "output": reader["output"],
        },
        "reference_thresholds": {
            "detector_score": 0.25,
            "reader_score": 0.0,
        },
    }
    classes_doc = {
        "schema_version": 1,
        "detector": {"0": "background", "1": "speed_sign"},
        "reader": classes,
    }
    capability = {
        "schema_version": 1,
        "mode": "offline_reference_perception",
        "supported": [
            "vertical_speed_sign_detection",
            "sign_family_and_value_reading",
        ],
        "not_supported": [
            "road_marking_runtime_detection",
            "temporal_consensus",
            "road_ownership",
            "passage_detection",
            "current_limit_inference",
            "vehicle_control",
        ],
        "advisory_only": True,
    }
    model_card = {
        "schema_version": 1,
        "name": "NNSLR detector+reader baseline bundle",
        "dataset_sha256": detector["dataset_sha256"],
        "split_sha256": detector["split_sha256"],
        "detector_checkpoint_sha256": detector.get("checkpoint_sha256"),
        "reader_checkpoint_sha256": reader.get("checkpoint_sha256"),
        "detector_onnx_sha256": detector["onnx_sha256"],
        "reader_onnx_sha256": reader["onnx_sha256"],
        "evaluation_status": "bundle packaging does not establish accuracy or target-runtime suitability",
    }

    documents = {
        "config.json": config,
        "classes.json": classes_doc,
        "capability.json": capability,
        "model_card.json": model_card,
    }
    for name, payload in documents.items():
        (output_dir / name).write_bytes(_canonical_json(payload))

    hashes = {
        name: _sha256(output_dir / name)
        for name in REQUIRED_FILES
        if name != "sha256sums.json"
    }
    sums = {
        "schema_version": 1,
        "files": dict(sorted(hashes.items())),
    }
    sums_bytes = _canonical_json(sums)
    (output_dir / "sha256sums.json").write_bytes(sums_bytes)
    bundle_digest = hashlib.sha256(sums_bytes).hexdigest()

    report = verify_bundle(output_dir)
    report["bundle_digest"] = bundle_digest
    return report


def verify_bundle(bundle_dir: Path) -> dict[str, Any]:
    bundle_dir = Path(bundle_dir)
    if not bundle_dir.is_dir():
        raise ValueError(f"bundle_not_found: {bundle_dir}")
    missing = [name for name in REQUIRED_FILES if not (bundle_dir / name).is_file()]
    if missing:
        raise ValueError(f"bundle_missing_files: {','.join(missing)}")

    sums_path = bundle_dir / "sha256sums.json"
    sums = _load_json(sums_path)
    files = sums.get("files")
    if sums.get("schema_version") != 1 or not isinstance(files, dict):
        raise ValueError("invalid_sha256_manifest")
    expected_names = set(REQUIRED_FILES) - {"sha256sums.json"}
    if set(files) != expected_names:
        raise ValueError("sha256_manifest_membership_mismatch")
    for name, expected in files.items():
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"invalid_sha256: {name}")
        actual = _sha256(bundle_dir / name)
        if actual != expected:
            raise ValueError(f"bundle_hash_mismatch: {name}")

    config = _load_json(bundle_dir / "config.json")
    detector = _load_json(bundle_dir / "detector.contract.json")
    reader = _load_json(bundle_dir / "reader.contract.json")
    capability = _load_json(bundle_dir / "capability.json")
    classes = _load_json(bundle_dir / "classes.json")

    if config.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError("bundle_schema_mismatch")
    if capability.get("advisory_only") is not True:
        raise ValueError("bundle_capability_not_advisory")
    if detector.get("task") != "detector" or reader.get("task") != "reader":
        raise ValueError("bundle_contract_task_mismatch")
    if (
        config.get("dataset_sha256") != detector.get("dataset_sha256")
        or config.get("dataset_sha256") != reader.get("dataset_sha256")
        or config.get("split_sha256") != detector.get("split_sha256")
        or config.get("split_sha256") != reader.get("split_sha256")
    ):
        raise ValueError("bundle_provenance_mismatch")
    if classes.get("reader") != reader.get("output", {}).get("classes"):
        raise ValueError("bundle_class_mapping_mismatch")
    if detector.get("onnx_sha256") != _sha256(bundle_dir / "detector.onnx"):
        raise ValueError("bundle_detector_contract_hash_mismatch")
    if reader.get("onnx_sha256") != _sha256(bundle_dir / "reader.onnx"):
        raise ValueError("bundle_reader_contract_hash_mismatch")

    sums_bytes = sums_path.read_bytes()
    return {
        "valid": True,
        "bundle": str(bundle_dir),
        "bundle_digest": hashlib.sha256(sums_bytes).hexdigest(),
        "dataset_sha256": config["dataset_sha256"],
        "split_sha256": config["split_sha256"],
        "file_count": len(REQUIRED_FILES),
        "advisory_only": True,
    }

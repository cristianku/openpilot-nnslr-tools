from __future__ import annotations

import hashlib
import json

import pytest

from nnslr_tools.bundle import package_model, verify_bundle
from test_annotations import invoke


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(root):
    detector = root / "detector.onnx"
    reader = root / "reader.onnx"
    detector.write_bytes(b"detector-onnx")
    reader.write_bytes(b"reader-onnx")
    dataset_sha = "a" * 64
    split_sha = "b" * 64

    detector_contract = root / "detector.onnx.contract.json"
    detector_contract.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": "detector",
                "onnx_sha256": _sha(detector),
                "checkpoint_sha256": "c" * 64,
                "dataset_sha256": dataset_sha,
                "split_sha256": split_sha,
                "verification": {"passed": True},
                "input": {"shape": [1, 3, 320, 320]},
                "outputs": {
                    "boxes": {"shape": ["detections", 4]},
                    "scores": {"shape": ["detections"]},
                    "labels": {"shape": ["detections"]},
                },
            }
        )
        + "\n"
    )

    reader_contract = root / "reader.onnx.contract.json"
    reader_contract.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": "reader",
                "onnx_sha256": _sha(reader),
                "checkpoint_sha256": "d" * 64,
                "dataset_sha256": dataset_sha,
                "split_sha256": split_sha,
                "verification": {"passed": True},
                "input": {"shape": [1, 3, 160, 160]},
                "output": {
                    "shape": [1, 2],
                    "classes": ["maximum_speed:30", "maximum_speed:50"],
                },
            }
        )
        + "\n"
    )
    return detector, detector_contract, reader, reader_contract


def test_package_and_verify_bundle(tmp_path) -> None:
    detector, detector_contract, reader, reader_contract = _inputs(tmp_path)
    bundle = tmp_path / "bundle"

    report = package_model(
        detector,
        detector_contract,
        reader,
        reader_contract,
        bundle,
    )

    assert report["valid"] is True
    assert report["advisory_only"] is True
    assert (bundle / "sha256sums.json").is_file()
    assert verify_bundle(bundle)["bundle_digest"] == report["bundle_digest"]


def test_bundle_detects_single_byte_tamper(tmp_path) -> None:
    detector, detector_contract, reader, reader_contract = _inputs(tmp_path)
    bundle = tmp_path / "bundle"
    package_model(detector, detector_contract, reader, reader_contract, bundle)

    with (bundle / "reader.onnx").open("ab") as handle:
        handle.write(b"x")

    with pytest.raises(ValueError, match="bundle_hash_mismatch"):
        verify_bundle(bundle)


def test_package_rejects_unverified_onnx(tmp_path) -> None:
    detector, detector_contract, reader, reader_contract = _inputs(tmp_path)
    payload = json.loads(reader_contract.read_text())
    payload["verification"]["passed"] = False
    reader_contract.write_text(json.dumps(payload) + "\n")

    with pytest.raises(ValueError, match="onnx_not_verified"):
        package_model(
            detector,
            detector_contract,
            reader,
            reader_contract,
            tmp_path / "bundle",
        )


def test_bundle_cli_roundtrip(tmp_path, capsys) -> None:
    detector, detector_contract, reader, reader_contract = _inputs(tmp_path)
    bundle = tmp_path / "bundle"

    code, out = invoke(
        capsys,
        "package-model",
        "--detector-onnx",
        str(detector),
        "--detector-contract",
        str(detector_contract),
        "--reader-onnx",
        str(reader),
        "--reader-contract",
        str(reader_contract),
        "--output",
        str(bundle),
    )
    assert code == 0, out.err + out.out

    code, out = invoke(capsys, "verify-bundle", str(bundle))
    assert code == 0, out.err + out.out
    assert json.loads(out.out)["valid"] is True

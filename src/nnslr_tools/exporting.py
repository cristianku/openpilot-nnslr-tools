"""Reference ONNX export for trained NNSLR baselines."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from nnslr_tools.training import _require_training_stack


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reader_io_contract(checkpoint: dict[str, Any]) -> dict[str, Any]:
    classes = checkpoint.get("classes")
    input_size = checkpoint.get("input_size")
    if not isinstance(classes, list) or not classes or not all(isinstance(v, str) for v in classes):
        raise ValueError("checkpoint_invalid_classes")
    if type(input_size) is not int or input_size <= 0:
        raise ValueError("checkpoint_invalid_input_size")
    return {
        "schema_version": 1,
        "task": "reader",
        "architecture": "mobilenet_v3_small",
        "input": {
            "name": "images",
            "shape": [1, 3, input_size, input_size],
            "dtype": "float32",
            "layout": "NCHW",
            "color": "RGB",
            "source": "reviewed_bbox_crop",
            "resize": [input_size, input_size],
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "output": {
            "name": "logits",
            "shape": [1, len(classes)],
            "dtype": "float32",
            "classes": classes,
            "decode": "argmax",
        },
        "dataset_sha256": checkpoint.get("dataset_sha256"),
        "split_sha256": checkpoint.get("split_sha256"),
    }


def export_reader_onnx(
    checkpoint_path: Path,
    output_path: Path,
    *,
    verify: bool = True,
) -> dict[str, Any]:
    torch, _, _, _, _, _, mobilenet_v3_small = _require_training_stack()
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("task") != "reader":
        raise ValueError("checkpoint_task_mismatch: expected reader")
    contract = reader_io_contract(checkpoint)
    classes = contract["output"]["classes"]
    input_size = contract["input"]["resize"][0]

    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(classes))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)

    try:
        program = torch.onnx.export(
            model,
            (example,),
            input_names=["images"],
            output_names=["logits"],
            dynamo=True,
            external_data=False,
        )
        program.save(str(output_path))
    except ImportError as exc:
        raise ValueError(
            "onnx_dependencies_missing: install onnx and onnxscript in the training environment"
        ) from exc

    try:
        import onnx
    except ImportError as exc:
        raise ValueError("onnx_dependencies_missing: install onnx") from exc
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    verification = {"enabled": verify, "passed": None}
    if verify:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ValueError("onnxruntime_missing") from exc
        with torch.no_grad():
            expected = model(example).detach().cpu().numpy()
        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        actual = session.run(["logits"], {"images": example.numpy()})[0]
        import numpy as np
        max_abs_error = float(np.max(np.abs(expected - actual)))
        passed = bool(np.allclose(expected, actual, rtol=1e-4, atol=1e-5))
        verification = {
            "enabled": True,
            "passed": passed,
            "max_abs_error": max_abs_error,
            "rtol": 1e-4,
            "atol": 1e-5,
            "provider": "CPUExecutionProvider",
        }
        if not passed:
            raise ValueError(f"onnx_parity_failed: max_abs_error={max_abs_error}")

    contract["onnx_sha256"] = sha256_file(output_path)
    contract["checkpoint"] = str(checkpoint_path)
    contract["checkpoint_sha256"] = sha256_file(Path(checkpoint_path))
    contract["verification"] = verification
    contract_path = output_path.with_suffix(output_path.suffix + ".contract.json")
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "valid": True,
        "task": "reader",
        "onnx": str(output_path),
        "onnx_sha256": contract["onnx_sha256"],
        "contract": str(contract_path),
        "verification": verification,
    }

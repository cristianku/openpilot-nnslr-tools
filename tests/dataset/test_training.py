from __future__ import annotations

import hashlib
import json
import sys

from nnslr_tools.annotations import load_jsonl, validate_dataset
from nnslr_tools.training import prepare_detector_training, prepare_reader_training
from test_annotations import invoke, reviewed, write


def _training_fixture(root, capsys):
    source_rows = []
    for index in range(8):
        row = reviewed(root, f"{index + 1:08x}--abcdef1234", index=index, negative=index == 0)
        if not index == 0:
            value = 30 if index % 2 else 50
            row["detections"][0]["value_kph"] = value
            row["source_proposals"][0]["value_kph"] = value
        source_rows.append(row)

    source = write(root, source_rows)
    code, out = invoke(capsys, "import-annotations", str(source), "--data-root", str(root))
    assert code == 0, out.err + out.out

    dataset = root / "annotations" / "objects.jsonl"
    canonical = load_jsonl(dataset)
    report = validate_dataset(canonical, root)
    assert report["valid"]

    negative = next(r for r in canonical if r["sign_family"] == "not_a_sign")
    thirties = [r for r in canonical if r["value_kph"] == 30]
    fifties = [r for r in canonical if r["value_kph"] == 50]
    assert len(thirties) >= 2 and len(fifties) >= 2

    assignments = {r["annotation_id"]: "test" for r in canonical}
    assignments[negative["annotation_id"]] = "hard-negative"
    assignments[thirties[0]["annotation_id"]] = "train"
    assignments[fifties[0]["annotation_id"]] = "train"
    assignments[thirties[1]["annotation_id"]] = "validation"
    assignments[fifties[1]["annotation_id"]] = "validation"
    remaining = [r for r in canonical if assignments[r["annotation_id"]] == "test"]
    if remaining:
        assignments[remaining[0]["annotation_id"]] = "route-held-out"

    payload = {
        "schema_version": 1,
        "dataset_kind": "gold",
        "dataset_sha256": report["dataset_sha256"],
        "assignments": dict(sorted(assignments.items())),
    }
    frozen = root / "splits" / "fixture.json"
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
    latest = root / "splits" / "latest.json"
    latest.write_text(
        json.dumps({"path": "splits/fixture.json", "sha256": digest}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return dataset, frozen, latest


def test_prepare_reader_training_follows_hash_bound_latest_pointer(tmp_path, capsys):
    dataset, frozen, latest = _training_fixture(tmp_path, capsys)
    assert "torch" not in sys.modules

    plan = prepare_reader_training(dataset, latest, tmp_path)

    assert plan["trainable"] is True
    assert plan["classes"] == ["maximum_speed:30", "maximum_speed:50"]
    assert plan["split_path"] == str(frozen)
    assert plan["requested_split_path"] == str(latest)
    assert plan["split_sha256"] == hashlib.sha256(frozen.read_bytes()).hexdigest()
    assert plan["counts_by_split"]["train"] == 2
    assert plan["counts_by_split"]["validation"] == 2
    assert "torch" not in sys.modules


def test_train_dry_run_uses_latest_pointer_without_gpu_stack(tmp_path, capsys):
    dataset, _, _ = _training_fixture(tmp_path, capsys)
    assert "torch" not in sys.modules

    code, out = invoke(
        capsys,
        "train",
        str(dataset),
        "--data-root",
        str(tmp_path),
        "--dry-run",
    )

    assert code == 0, out.err + out.out
    summary = json.loads(out.out)
    assert summary["trainable"] is True
    assert summary["record_count"] >= 4
    assert "records" not in summary
    assert "torch" not in sys.modules


def test_latest_pointer_hash_mismatch_is_rejected(tmp_path, capsys):
    dataset, _, latest = _training_fixture(tmp_path, capsys)
    pointer = json.loads(latest.read_text())
    pointer["sha256"] = "0" * 64
    latest.write_text(json.dumps(pointer) + "\n")

    try:
        prepare_reader_training(dataset, latest, tmp_path)
    except ValueError as exc:
        assert "split_pointer_hash_mismatch" in str(exc)
    else:
        raise AssertionError("tampered split pointer was accepted")



def test_prepare_detector_training_groups_full_frames_without_torch(tmp_path, capsys):
    dataset, frozen, latest = _training_fixture(tmp_path, capsys)
    assert "torch" not in sys.modules

    plan = prepare_detector_training(dataset, latest, tmp_path)

    assert plan["trainable"] is True
    assert plan["classes"] == ["speed_sign"]
    assert plan["split_path"] == str(frozen)
    assert plan["boxes_by_split"]["train"] >= 2
    assert plan["boxes_by_split"]["validation"] >= 2
    assert plan["negative_frames_by_split"]["hard-negative"] == 1
    assert all(frame["image_sha256"] for frame in plan["frames"])
    assert "torch" not in sys.modules


def test_detector_train_dry_run_is_cpu_only(tmp_path, capsys):
    dataset, _, _ = _training_fixture(tmp_path, capsys)
    assert "torch" not in sys.modules

    code, out = invoke(
        capsys,
        "train",
        str(dataset),
        "--task",
        "detector",
        "--data-root",
        str(tmp_path),
        "--dry-run",
    )

    assert code == 0, out.err + out.out
    summary = json.loads(out.out)
    assert summary["task"] == "detector"
    assert summary["trainable"] is True
    assert summary["record_count"] >= 5
    assert "frames" not in summary
    assert "torch" not in sys.modules

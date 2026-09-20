from __future__ import annotations

import json
import sys

import pytest

from nnslr_tools.evaluation import box_iou, match_detections
from test_training import _training_fixture
from test_annotations import invoke


def test_box_iou_exact_disjoint_and_partial() -> None:
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert box_iou([0, 0, 10, 10], [5, 5, 15, 15]) == pytest.approx(25 / 175)


def test_match_detections_is_one_to_one_and_score_ordered() -> None:
    truth = [[0, 0, 10, 10], [20, 20, 30, 30]]
    predicted = [
        {"box": [0, 0, 10, 10], "score": .95},
        {"box": [1, 1, 9, 9], "score": .8},
        {"box": [20, 20, 30, 30], "score": .7},
    ]

    result = match_detections(predicted, truth, iou_threshold=.5)

    assert result["true_positives"] == 2
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 0


@pytest.mark.parametrize("task", ["reader", "detector"])
def test_evaluate_dry_run_is_cpu_only(tmp_path, capsys, task) -> None:
    dataset, _, _ = _training_fixture(tmp_path, capsys)
    assert "torch" not in sys.modules

    code, out = invoke(
        capsys,
        "evaluate",
        str(dataset),
        "--task",
        task,
        "--partition",
        "test",
        "--data-root",
        str(tmp_path),
        "--dry-run",
    )

    assert code == 0, out.err + out.out
    report = json.loads(out.out)
    assert report["task"] == task
    assert report["partition"] == "test"
    assert report["sample_count"] > 0
    assert report["checkpoint_required_for_execution"] is True
    assert "torch" not in sys.modules


def test_evaluate_without_checkpoint_rejected_after_plan(tmp_path, capsys) -> None:
    dataset, _, _ = _training_fixture(tmp_path, capsys)

    code, out = invoke(
        capsys,
        "evaluate",
        str(dataset),
        "--data-root",
        str(tmp_path),
    )

    assert code == 2
    assert "--checkpoint is required" in out.err

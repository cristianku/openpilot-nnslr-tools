from __future__ import annotations

import json

from nnslr_tools.hard_examples import mine_hard_examples
from test_annotations import invoke


def test_reader_mining_keeps_only_errors() -> None:
    report = {
        "task": "reader",
        "partition": "test",
        "samples": [
            {
                "annotation_id": "a",
                "image_path": "a.png",
                "truth": "maximum_speed:30",
                "predicted": "maximum_speed:30",
                "confidence": .99,
                "correct": True,
            },
            {
                "annotation_id": "b",
                "image_path": "b.png",
                "truth": "maximum_speed:50",
                "predicted": "maximum_speed:30",
                "confidence": .8,
                "correct": False,
            },
        ],
    }

    mined = mine_hard_examples(report)

    assert len(mined) == 1
    assert mined[0]["annotation_id"] == "b"
    assert mined[0]["kind"] == "misclassification"


def test_detector_mining_ranks_frames_by_total_errors() -> None:
    report = {
        "task": "detector",
        "partition": "test",
        "score_threshold": .25,
        "iou_threshold": .5,
        "images": [
            {
                "image_path": "ok.png",
                "image_sha256": "a" * 64,
                "route_id": "route-a",
                "segment_index": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "truth_boxes": [],
                "predictions": [],
            },
            {
                "image_path": "hard.png",
                "image_sha256": "b" * 64,
                "route_id": "route-b",
                "segment_index": 0,
                "false_positives": 2,
                "false_negatives": 1,
                "truth_boxes": [[1, 2, 3, 4]],
                "predictions": [{"box": [1, 2, 4, 5], "score": .7}],
            },
            {
                "image_path": "miss.png",
                "image_sha256": "c" * 64,
                "route_id": "route-c",
                "segment_index": 0,
                "false_positives": 0,
                "false_negatives": 1,
                "truth_boxes": [[1, 2, 3, 4]],
                "predictions": [],
            },
        ],
    }

    mined = mine_hard_examples(report)

    assert [row["image_path"] for row in mined] == ["hard.png", "miss.png"]
    assert mined[0]["kind"] == "false_positive+false_negative"


def test_mine_hard_examples_cli_writes_jsonl(tmp_path, capsys) -> None:
    evaluation = tmp_path / "evaluation.json"
    output = tmp_path / "hard.jsonl"
    evaluation.write_text(
        json.dumps(
            {
                "task": "reader",
                "partition": "test",
                "samples": [
                    {
                        "annotation_id": "x",
                        "image_path": "x.png",
                        "truth": "maximum_speed:80",
                        "predicted": "maximum_speed:30",
                        "confidence": .7,
                        "correct": False,
                    }
                ],
            }
        )
    )

    code, out = invoke(
        capsys,
        "mine-hard-examples",
        str(evaluation),
        "--output",
        str(output),
    )

    assert code == 0, out.err + out.out
    summary = json.loads(out.out)
    assert summary["hard_example_count"] == 1
    row = json.loads(output.read_text())
    assert row["annotation_id"] == "x"

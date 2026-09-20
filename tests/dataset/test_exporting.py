from __future__ import annotations

import pytest

from nnslr_tools.exporting import detector_io_contract, reader_io_contract


def test_reader_io_contract_freezes_preprocessing_and_classes() -> None:
    contract = reader_io_contract(
        {
            "task": "reader",
            "classes": ["maximum_speed:30", "maximum_speed:50"],
            "input_size": 160,
            "dataset_sha256": "a" * 64,
            "split_sha256": "b" * 64,
        }
    )

    assert contract["input"]["shape"] == [1, 3, 160, 160]
    assert contract["input"]["layout"] == "NCHW"
    assert contract["input"]["color"] == "RGB"
    assert contract["input"]["normalization"]["mean"] == [0.485, 0.456, 0.406]
    assert contract["output"]["shape"] == [1, 2]
    assert contract["output"]["classes"] == ["maximum_speed:30", "maximum_speed:50"]


@pytest.mark.parametrize(
    "checkpoint",
    [
        {"classes": [], "input_size": 160},
        {"classes": ["maximum_speed:30"], "input_size": 0},
        {"classes": [30], "input_size": 160},
    ],
)
def test_reader_io_contract_rejects_incomplete_checkpoint_metadata(checkpoint) -> None:
    with pytest.raises(ValueError):
        reader_io_contract(checkpoint)



def test_detector_io_contract_freezes_full_frame_interface() -> None:
    contract = detector_io_contract(
        {
            "task": "detector",
            "classes": ["background", "speed_sign"],
            "dataset_sha256": "a" * 64,
            "split_sha256": "b" * 64,
        }
    )

    assert contract["input"]["shape"] == [1, 3, 760, 1344]
    assert contract["input"]["value_range"] == [0.0, 1.0]
    assert contract["input"]["external_resize"] is None
    assert contract["input"]["model_internal_resize"] == [320, 320]
    assert contract["input"]["model_internal_normalization"] == {
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    }
    assert contract["outputs"]["boxes"]["format"] == "xyxy"
    assert contract["outputs"]["labels"]["mapping"] == {"1": "speed_sign"}
    assert contract["postprocessing"] == "model_includes_score_filtering_and_nms"


def test_detector_io_contract_rejects_wrong_checkpoint() -> None:
    with pytest.raises(ValueError):
        detector_io_contract({"task": "reader", "classes": ["speed_sign"]})



def test_detector_io_contract_supports_explicit_native_dimensions() -> None:
    contract = detector_io_contract(
        {
            "task": "detector",
            "classes": ["background", "speed_sign"],
            "dataset_sha256": "a" * 64,
            "split_sha256": "b" * 64,
        },
        input_width=1920,
        input_height=1080,
    )
    assert contract["input"]["shape"] == [1, 3, 1080, 1920]


@pytest.mark.parametrize("width,height", [(0, 760), (1344, 0), (-1, 760)])
def test_detector_io_contract_rejects_invalid_native_dimensions(width, height) -> None:
    with pytest.raises(ValueError):
        detector_io_contract(
            {
                "task": "detector",
                "classes": ["background", "speed_sign"],
            },
            input_width=width,
            input_height=height,
        )

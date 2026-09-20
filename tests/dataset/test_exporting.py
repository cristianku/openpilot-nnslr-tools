from __future__ import annotations

import pytest

from nnslr_tools.exporting import reader_io_contract


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

    assert contract["input"]["shape"] == ["batch", 3, 160, 160]
    assert contract["input"]["layout"] == "NCHW"
    assert contract["input"]["color"] == "RGB"
    assert contract["input"]["normalization"]["mean"] == [0.485, 0.456, 0.406]
    assert contract["output"]["shape"] == ["batch", 2]
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

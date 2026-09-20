from __future__ import annotations

import json
import sys

import pytest

from nnslr_tools.replay import decode_reader_label, load_route_frames
from test_annotations import invoke, png


@pytest.mark.parametrize(
    "label,family,state,value,supported",
    [
        ("maximum_speed:30", "maximum_speed", "read", 30, True),
        ("zone:50", "zone", "read", 50, True),
        ("cancellation", "cancellation", "unknown", None, True),
        ("unreadable", "unreadable", "unreadable", None, True),
        ("other_sign", "other_sign", "not_applicable", None, False),
        ("maximum_speed:unknown", "maximum_speed", "unknown", None, True),
    ],
)
def test_decode_reader_label(label, family, state, value, supported) -> None:
    decoded = decode_reader_label(label)
    assert decoded == {
        "sign_family": family,
        "value_state": state,
        "value_kph": value,
        "supported_domain": supported,
    }


def test_route_replay_dry_run_reads_manifests_without_torch(tmp_path, capsys) -> None:
    route = "00000001--abcdef1234"
    image = tmp_path / "derived" / "frames" / route / "0" / "frame_00000000.png"
    png(image)
    manifest = image.parent / "frames.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "route_id": route,
                "segment_index": 0,
                "image_path": str(image.relative_to(tmp_path)),
                "output_index": 0,
                "decoded_frame_index": 0,
                "camera_stream": "narrow_road",
                "image_sha256": "placeholder",
                "media_time_s": 0.0,
                "media_time_provenance": "raw_hevc_frame_duration",
                "capture_mono_ns": None,
                "capture_time_provenance": "unknown",
                "alignment_status": "unresolved",
            }
        )
        + "\n"
    )
    assert "torch" not in sys.modules

    frames, manifests = load_route_frames(tmp_path, route)
    assert len(frames) == 1
    assert manifests == [str(manifest)]

    code, out = invoke(
        capsys,
        "replay",
        "--route",
        route,
        "--data-root",
        str(tmp_path),
        "--dry-run",
    )
    assert code == 0, out.err + out.out
    report = json.loads(out.out)
    assert report["frame_count"] == 1
    assert report["gpu_required"] is False
    assert "torch" not in sys.modules


def test_decode_reader_label_rejects_unknown_encoding() -> None:
    with pytest.raises(ValueError):
        decode_reader_label("maximum_speed:banana")

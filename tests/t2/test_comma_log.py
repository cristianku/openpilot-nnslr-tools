from __future__ import annotations

from nnslr_tools.comma_log import CameraStateEvent, EncodeIndexEvent, MapSpeedEvent, event_from_dict


def test_encode_index_event_preserves_presentation_and_encode_order() -> None:
    event = event_from_dict({
        "kind": "encode_index",
        "service": "narrowRoadEncodeIdx",
        "log_mono_time": 100,
        "frame_id": 55,
        "segment_num": 4,
        "segment_id": 7,
        "segment_id_encode": 9,
        "timestamp_sof": 1000,
        "timestamp_eof": 1100,
        "encode_type": "fullHEVC",
    })
    assert isinstance(event, EncodeIndexEvent)
    assert event.presentation_index == 7
    assert event.segment_id_encode == 9


def test_camera_state_event() -> None:
    event = event_from_dict({
        "kind": "camera_state",
        "service": "narrowRoadCameraState",
        "log_mono_time": 100,
        "frame_id": 55,
        "encode_id": 7,
        "timestamp_sof": 1000,
        "timestamp_eof": 1100,
    })
    assert isinstance(event, CameraStateEvent)
    assert event.encode_id == 7


def test_map_event_keeps_raw_hint_values() -> None:
    event = event_from_dict({
        "kind": "map_speed",
        "service": "liveMapDataSP",
        "log_mono_time": 100,
        "speed_limit": 22.2222,
        "speed_limit_ahead": None,
    })
    assert isinstance(event, MapSpeedEvent)
    assert event.speed_limit == 22.2222

# [t2-validation] - START
from pathlib import Path
import pytest
from nnslr_tools.comma_log import CommaLogError, resolve_schema_paths


def test_resolve_schemas_uses_explicit_matching_car_checkout(tmp_path):
    root = tmp_path / "runtime"
    cereal = root / "openpilot" / "cereal"
    cereal.mkdir(parents=True)
    (cereal / "log.capnp").touch()
    car_root = tmp_path / "dbc"
    car = car_root / "opendbc" / "car"
    car.mkdir(parents=True)
    (car / "car.capnp").touch()
    assert resolve_schema_paths(root, car_root) == (cereal, car)


def test_resolve_schemas_supports_embedded_layout_and_reports_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("NNSLR_OPENDBC_ROOT", raising=False)
    (tmp_path / "cereal").mkdir()
    (tmp_path / "cereal" / "log.capnp").touch()
    with pytest.raises(CommaLogError, match="car_schema_not_found"):
        resolve_schema_paths(tmp_path)
    car = tmp_path / "opendbc" / "car"
    car.mkdir(parents=True)
    (car / "car.capnp").touch()
    assert resolve_schema_paths(tmp_path) == (tmp_path / "cereal", car)
# [t2-validation] - END

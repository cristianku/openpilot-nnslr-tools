from __future__ import annotations

import pytest

from speed_vision_core import types
from speed_vision_core.types import (
    AdvisoryComparison,
    Agreement,
    ApplicabilityEvidence,
    ApplicabilityVerdict,
    DisplayDecision,
    HypothesisState,
    LimitHypothesis,
    NnslerContractError,
    PassageEvidence,
    PassageVerdict,
    PerceptionHealth,
    ReasonCode,
    RoadContext,
    SignFamily,
    SignTrack,
    SourceKind,
    SourceSnapshot,
    ValueState,
)


def test_road_context_round_trip() -> None:
    value = RoadContext(
        session_id="s1",
        context_mono_ns=1_000,
        calibration_available=True,
        ego_motion_available=True,
        road_continuity_id="road-1",
        at_junction=False,
        country_ambiguous=False,
        provenance="fixture",
    )
    assert types.road_context_from_dict(types.road_context_to_dict(value)) == value


def test_sign_track_round_trip_preserves_nested_evidence() -> None:
    value = SignTrack(
        track_id="track-1",
        sign_family=SignFamily.MAX_SPEED,
        observation_ids=("o1", "o2", "o3"),
        consensus_value_state=ValueState.VALUE,
        consensus_value_kph=80,
        first_seen_mono_ns=1_000,
        last_seen_mono_ns=1_500,
        capture_timestamps_ns=(1_000, 1_250, 1_500),
        applicability=ApplicabilityEvidence(
            verdict=ApplicabilityVerdict.OWN,
            contributing_frame_ids=("f1", "f2"),
            contributing_context_ids=("c1",),
            method="fixture",
            method_version="1",
        ),
        passage=PassageEvidence(
            verdict=PassageVerdict.PASSED,
            earliest_crossing_mono_ns=1_600,
            latest_crossing_mono_ns=1_650,
            method="fixture",
            method_version="1",
        ),
    )

    payload = types.sign_track_to_dict(value)
    assert types.sign_track_from_dict(payload) == value


def test_perception_health_round_trip() -> None:
    value = PerceptionHealth(
        session_id="s1",
        backend="fixture",
        backend_available=True,
        last_processed_capture_mono_ns=1_000,
        last_successful_completion_mono_ns=1_100,
        overflow=False,
        fault_code=None,
        dropped_frames=2,
    )
    assert types.perception_health_from_dict(types.perception_health_to_dict(value)) == value


def test_advisory_comparison_round_trip() -> None:
    car = SourceSnapshot(
        source=SourceKind.CAR,
        provenance="car",
        valid=True,
        value_mps=80 / 3.6,
        observed_mono_ns=1_000,
    )
    map_source = SourceSnapshot(
        source=SourceKind.MAP,
        provenance="map",
        valid=True,
        value_mps=80 / 3.6,
        observed_mono_ns=1_010,
    )
    vision = LimitHypothesis(
        has_value=True,
        value_kph=80,
        state=HypothesisState.CURRENT,
        track_id="track-1",
        last_context_mono_ns=1_020,
        usable_for_advisory=True,
    )
    value = AdvisoryComparison(
        car=car,
        map=map_source,
        vision=vision,
        car_age_ns=20,
        map_age_ns=10,
        vision_age_ns=5,
        agreement=Agreement.AGREE,
        display_decision=DisplayDecision.SHOW_CURRENT,
        reason="fixture",
    )

    assert types.advisory_comparison_from_dict(
        types.advisory_comparison_to_dict(value)
    ) == value


def assert_invalid_numeric(callable_) -> None:
    with pytest.raises(NnslerContractError) as exc:
        callable_()
    assert exc.value.reason_code == ReasonCode.INVALID_NUMERIC_TYPE


def test_hypothesis_deserializer_does_not_coerce_string_value() -> None:
    assert_invalid_numeric(
        lambda: types.limit_hypothesis_from_dict(
            {
                "has_value": True,
                "value_kph": "80",
                "state": "current",
                "usable_for_advisory": True,
            }
        )
    )


def test_source_deserializer_does_not_coerce_string_float_or_timestamp() -> None:
    assert_invalid_numeric(
        lambda: types.source_snapshot_from_dict(
            {
                "source": "car",
                "provenance": "fixture",
                "valid": True,
                "value_mps": "22.2",
            }
        )
    )
    assert_invalid_numeric(
        lambda: types.source_snapshot_from_dict(
            {
                "source": "car",
                "provenance": "fixture",
                "valid": True,
                "value_mps": 22.2,
                "observed_mono_ns": "1000",
            }
        )
    )


def test_track_deserializer_does_not_coerce_value_or_capture_times() -> None:
    base = {
        "track_id": "track-1",
        "sign_family": "max_speed",
        "observation_ids": ["o1"],
        "consensus_value_state": "value",
        "consensus_value_kph": "80",
        "first_seen_mono_ns": 1000,
        "last_seen_mono_ns": 1000,
        "capture_timestamps_ns": [1000],
    }
    assert_invalid_numeric(lambda: types.sign_track_from_dict(base))

    base["consensus_value_kph"] = 80
    base["capture_timestamps_ns"] = ["1000"]
    assert_invalid_numeric(lambda: types.sign_track_from_dict(base))


def test_health_deserializer_does_not_coerce_dropped_frame_count() -> None:
    assert_invalid_numeric(
        lambda: types.perception_health_from_dict(
            {
                "session_id": "s1",
                "backend": "fixture",
                "backend_available": True,
                "dropped_frames": "2",
            }
        )
    )

from __future__ import annotations

import pytest

from speed_vision_core.temporal import (
    TemporalConsensusProfile,
    TemporalObservation,
    build_consensus_track,
    hypothesis_from_track,
)
from speed_vision_core.types import (
    ApplicabilityEvidence,
    ApplicabilityVerdict,
    HypothesisState,
    PassageEvidence,
    PassageVerdict,
    SignFamily,
    ValueState,
)


def obs(index: int, timestamp_ns: int, value: int = 80) -> TemporalObservation:
    return TemporalObservation(
        observation_id=f"obs-{index}",
        capture_mono_ns=timestamp_ns,
        sign_family=SignFamily.MAX_SPEED,
        value_state=ValueState.VALUE,
        value_kph=value,
        detection_score=.9,
        classification_score=.9,
    )


def test_single_observation_never_becomes_stable_limit() -> None:
    track = build_consensus_track("track-1", [obs(1, 1_000_000_000)])

    assert track.consensus_value_state == ValueState.UNKNOWN
    assert track.consensus_value_kph is None

    hypothesis = hypothesis_from_track(track)
    assert hypothesis.state == HypothesisState.OBSERVED
    assert hypothesis.has_value is False
    assert hypothesis.usable_for_advisory is False


def test_three_consistent_observations_within_window_establish_consensus() -> None:
    track = build_consensus_track(
        "track-1",
        [
            obs(1, 1_000_000_000),
            obs(2, 1_200_000_000),
            obs(3, 1_400_000_000),
        ],
    )

    assert track.consensus_value_state == ValueState.VALUE
    assert track.consensus_value_kph == 80
    assert track.observation_ids == ("obs-1", "obs-2", "obs-3")


def test_three_of_five_consistent_readings_win_over_two_conflicts() -> None:
    track = build_consensus_track(
        "track-1",
        [
            obs(1, 1_000_000_000, 80),
            obs(2, 1_100_000_000, 30),
            obs(3, 1_200_000_000, 80),
            obs(4, 1_300_000_000, 30),
            obs(5, 1_400_000_000, 80),
        ],
    )

    assert track.consensus_value_state == ValueState.VALUE
    assert track.consensus_value_kph == 80
    assert track.observation_ids == ("obs-1", "obs-3", "obs-5")


def test_old_evidence_outside_window_cannot_create_consensus() -> None:
    profile = TemporalConsensusProfile(
        min_observations=3,
        max_observations=5,
        window_ns=800_000_000,
    )
    track = build_consensus_track(
        "track-1",
        [
            obs(1, 1_000_000_000),
            obs(2, 1_200_000_000),
            obs(3, 2_100_000_000),
        ],
        profile=profile,
    )

    assert track.consensus_value_state == ValueState.UNKNOWN
    assert track.consensus_value_kph is None


def test_more_than_five_observations_use_only_recent_bounded_window() -> None:
    track = build_consensus_track(
        "track-1",
        [
            obs(1, 1_000_000_000, 30),
            obs(2, 1_100_000_000, 30),
            obs(3, 1_200_000_000, 30),
            obs(4, 1_300_000_000, 80),
            obs(5, 1_400_000_000, 80),
            obs(6, 1_500_000_000, 80),
        ],
    )

    assert track.consensus_value_kph == 80
    assert track.observation_ids == ("obs-4", "obs-5", "obs-6")


def test_same_capture_cannot_count_twice() -> None:
    first = obs(1, 1_000_000_000, 80)
    second = obs(2, 1_000_000_000, 80)

    with pytest.raises(ValueError, match="multiple_observations_same_capture"):
        build_consensus_track("track-1", [first, second])


def stable_track(*, applicability: ApplicabilityVerdict, passage: PassageVerdict):
    return build_consensus_track(
        "track-1",
        [
            obs(1, 1_000_000_000),
            obs(2, 1_200_000_000),
            obs(3, 1_400_000_000),
        ],
        applicability=ApplicabilityEvidence(verdict=applicability),
        passage=PassageEvidence(
            verdict=passage,
            earliest_crossing_mono_ns=1_500_000_000 if passage == PassageVerdict.PASSED else None,
            latest_crossing_mono_ns=1_550_000_000 if passage == PassageVerdict.PASSED else None,
        ),
    )


def test_own_road_ahead_is_not_current() -> None:
    hypothesis = hypothesis_from_track(
        stable_track(
            applicability=ApplicabilityVerdict.OWN,
            passage=PassageVerdict.AHEAD,
        )
    )

    assert hypothesis.has_value is True
    assert hypothesis.value_kph == 80
    assert hypothesis.state == HypothesisState.AHEAD
    assert hypothesis.usable_for_advisory is False


def test_current_requires_consensus_own_road_and_verified_passage() -> None:
    hypothesis = hypothesis_from_track(
        stable_track(
            applicability=ApplicabilityVerdict.OWN,
            passage=PassageVerdict.PASSED,
        )
    )

    assert hypothesis.state == HypothesisState.CURRENT
    assert hypothesis.value_kph == 80
    assert hypothesis.usable_for_advisory is True
    assert hypothesis.activation_start_mono_ns == 1_500_000_000
    assert hypothesis.activation_end_mono_ns == 1_550_000_000


def test_other_road_can_never_be_current() -> None:
    hypothesis = hypothesis_from_track(
        stable_track(
            applicability=ApplicabilityVerdict.OTHER,
            passage=PassageVerdict.PASSED,
        )
    )

    assert hypothesis.state == HypothesisState.UNCERTAIN
    assert hypothesis.has_value is True
    assert hypothesis.usable_for_advisory is False
    assert hypothesis.unavailable_reason == "other_road"


def test_passage_unknown_stays_observed() -> None:
    hypothesis = hypothesis_from_track(
        stable_track(
            applicability=ApplicabilityVerdict.OWN,
            passage=PassageVerdict.UNKNOWN,
        )
    )

    assert hypothesis.state == HypothesisState.OBSERVED
    assert hypothesis.usable_for_advisory is False
    assert hypothesis.unavailable_reason == "passage_unresolved"

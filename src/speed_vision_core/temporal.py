"""Pure temporal consensus and hypothesis derivation for NNSLR.

This module deliberately does not associate detections into physical tracks.
Association is a separate geometry/context problem. The functions here consume
observations already assigned to one candidate track and enforce the temporal
rule before any value can become a stable SignTrack consensus.

No vehicle-control type or target-speed output exists here.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Iterable

from .types import (
    ApplicabilityEvidence,
    ApplicabilityVerdict,
    HypothesisState,
    LimitHypothesis,
    MIN_VALUE_KPH,
    MAX_VALUE_KPH,
    PassageEvidence,
    PassageVerdict,
    SignFamily,
    SignTrack,
    ValueState,
)


@dataclass(frozen=True)
class TemporalConsensusProfile:
    """Frozen parameters for one temporal-consensus experiment."""

    min_observations: int = 3
    max_observations: int = 5
    window_ns: int = 800_000_000

    def __post_init__(self) -> None:
        if type(self.min_observations) is not int or self.min_observations < 2:
            raise ValueError("min_observations must be an integer >= 2")
        if type(self.max_observations) is not int or self.max_observations < self.min_observations:
            raise ValueError("max_observations must be >= min_observations")
        if type(self.window_ns) is not int or self.window_ns <= 0:
            raise ValueError("window_ns must be a positive integer")


DEFAULT_TEMPORAL_PROFILE = TemporalConsensusProfile()


@dataclass(frozen=True)
class TemporalObservation:
    """One accepted perception observation already associated to a track."""

    observation_id: str
    capture_mono_ns: int
    sign_family: SignFamily
    value_state: ValueState
    value_kph: int | None
    detection_score: float = 1.0
    classification_score: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.observation_id, str) or not self.observation_id:
            raise ValueError("observation_id must be non-empty")
        if type(self.capture_mono_ns) is not int or self.capture_mono_ns <= 0:
            raise ValueError("capture_mono_ns must be a positive integer")
        for name, value in (
            ("detection_score", self.detection_score),
            ("classification_score", self.classification_score),
        ):
            if type(value) not in (int, float) or isinstance(value, bool):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite in [0,1]")
        if self.value_state == ValueState.VALUE:
            if type(self.value_kph) is not int or isinstance(self.value_kph, bool):
                raise ValueError("VALUE requires integer value_kph")
            if not MIN_VALUE_KPH <= self.value_kph <= MAX_VALUE_KPH:
                raise ValueError("value_kph outside supported domain")
        elif self.value_kph is not None:
            raise ValueError("non-VALUE observation requires value_kph=None")


def _consensus_key(observation: TemporalObservation) -> tuple[SignFamily, ValueState, int | None]:
    return observation.sign_family, observation.value_state, observation.value_kph


def _dedupe_observations(observations: Iterable[TemporalObservation]) -> list[TemporalObservation]:
    """Deduplicate exact observation IDs and capture timestamps."""
    by_id: dict[str, TemporalObservation] = {}
    by_capture: dict[int, TemporalObservation] = {}
    for observation in observations:
        prior_id = by_id.get(observation.observation_id)
        if prior_id is not None and prior_id != observation:
            raise ValueError(f"conflicting_observation_id: {observation.observation_id}")
        prior_capture = by_capture.get(observation.capture_mono_ns)
        if prior_capture is not None and prior_capture != observation:
            raise ValueError(f"multiple_observations_same_capture: {observation.capture_mono_ns}")
        by_id[observation.observation_id] = observation
        by_capture[observation.capture_mono_ns] = observation
    return sorted(by_id.values(), key=lambda item: (item.capture_mono_ns, item.observation_id))


def _recent_window(
    observations: list[TemporalObservation],
    profile: TemporalConsensusProfile,
) -> list[TemporalObservation]:
    if not observations:
        return []
    latest = observations[-1].capture_mono_ns
    lower = latest - profile.window_ns
    in_window = [item for item in observations if item.capture_mono_ns >= lower]
    return in_window[-profile.max_observations:]


def build_consensus_track(
    track_id: str,
    observations: Iterable[TemporalObservation],
    *,
    profile: TemporalConsensusProfile = DEFAULT_TEMPORAL_PROFILE,
    applicability: ApplicabilityEvidence | None = None,
    passage: PassageEvidence | None = None,
) -> SignTrack:
    """Build a SignTrack from already-associated observations.

    A consensus exists only when one exact family/state/value tuple reaches
    the profile minimum within the recent bounded window, with no tie at the
    winning count. Otherwise the track remains UNKNOWN and has no value.
    """
    if not isinstance(track_id, str) or not track_id:
        raise ValueError("track_id must be non-empty")

    unique = _dedupe_observations(observations)
    if not unique:
        raise ValueError("track requires at least one observation")
    recent = _recent_window(unique, profile)

    counts = Counter(_consensus_key(item) for item in recent)
    ranked = counts.most_common()
    winner = ranked[0] if ranked else None
    tied = (
        winner is not None
        and len(ranked) > 1
        and ranked[1][1] == winner[1]
    )
    stable = winner is not None and winner[1] >= profile.min_observations and not tied

    if stable:
        family, value_state, value_kph = winner[0]
        contributing = [item for item in recent if _consensus_key(item) == winner[0]]
        observation_ids = tuple(item.observation_id for item in contributing)
        captures = tuple(item.capture_mono_ns for item in contributing)
    else:
        family = recent[-1].sign_family
        value_state = ValueState.UNKNOWN
        value_kph = None
        observation_ids = tuple(item.observation_id for item in recent)
        captures = tuple(item.capture_mono_ns for item in recent)

    return SignTrack(
        track_id=track_id,
        sign_family=family,
        observation_ids=observation_ids,
        consensus_value_state=value_state,
        consensus_value_kph=value_kph,
        first_seen_mono_ns=unique[0].capture_mono_ns,
        last_seen_mono_ns=unique[-1].capture_mono_ns,
        capture_timestamps_ns=captures,
        applicability=applicability,
        passage=passage,
    )


def hypothesis_from_track(track: SignTrack) -> LimitHypothesis:
    """Derive an advisory-only lifecycle state from a stable track.

    CURRENT is possible only for a numeric consensus, OWN-road evidence and
    PASSED evidence. Disappearance alone is never passage evidence.
    """
    has_value = (
        track.consensus_value_state == ValueState.VALUE
        and track.consensus_value_kph is not None
    )
    if not has_value:
        return LimitHypothesis(
            has_value=False,
            value_kph=None,
            state=HypothesisState.OBSERVED,
            track_id=track.track_id,
            last_context_mono_ns=track.last_seen_mono_ns,
            unavailable_reason="temporal_consensus_not_established",
            usable_for_advisory=False,
        )

    applicability = (
        track.applicability.verdict
        if track.applicability is not None
        else ApplicabilityVerdict.UNKNOWN
    )
    passage = (
        track.passage.verdict
        if track.passage is not None
        else PassageVerdict.UNKNOWN
    )

    if applicability == ApplicabilityVerdict.OTHER:
        state = HypothesisState.UNCERTAIN
        reason = "other_road"
        usable = False
    elif applicability != ApplicabilityVerdict.OWN:
        state = HypothesisState.OBSERVED
        reason = "road_ownership_unresolved"
        usable = False
    elif passage == PassageVerdict.AHEAD:
        state = HypothesisState.AHEAD
        reason = None
        usable = False
    elif passage == PassageVerdict.PASSED:
        state = HypothesisState.CURRENT
        reason = None
        usable = True
    else:
        state = HypothesisState.OBSERVED
        reason = "passage_unresolved"
        usable = False

    return LimitHypothesis(
        has_value=True,
        value_kph=track.consensus_value_kph,
        state=state,
        track_id=track.track_id,
        activation_start_mono_ns=(
            track.passage.earliest_crossing_mono_ns
            if track.passage is not None and passage == PassageVerdict.PASSED
            else None
        ),
        activation_end_mono_ns=(
            track.passage.latest_crossing_mono_ns
            if track.passage is not None and passage == PassageVerdict.PASSED
            else None
        ),
        last_context_mono_ns=track.last_seen_mono_ns,
        unavailable_reason=reason,
        usable_for_advisory=usable,
    )

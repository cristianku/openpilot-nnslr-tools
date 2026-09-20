from __future__ import annotations

import pytest

from speed_vision_core.state import HealthGateProfile, gate_hypothesis_by_health
from speed_vision_core.types import HypothesisState, LimitHypothesis, PerceptionHealth


NOW = 10_000_000_000


def current() -> LimitHypothesis:
    return LimitHypothesis(
        has_value=True,
        value_kph=80,
        state=HypothesisState.CURRENT,
        track_id="track-1",
        last_context_mono_ns=NOW - 50_000_000,
        usable_for_advisory=True,
    )


def health(**overrides) -> PerceptionHealth:
    values = dict(
        session_id="session-1",
        backend="test",
        backend_available=True,
        last_processed_capture_mono_ns=NOW - 100_000_000,
        last_successful_completion_mono_ns=NOW - 50_000_000,
        overflow=False,
        fault_code=None,
        dropped_frames=0,
    )
    values.update(overrides)
    return PerceptionHealth(**values)


def test_healthy_fresh_backend_preserves_hypothesis() -> None:
    hypothesis = current()
    assert gate_hypothesis_by_health(hypothesis, health(), now_mono_ns=NOW) is hypothesis


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"backend_available": False}, "backend_unavailable"),
        ({"overflow": True}, "detection_overflow"),
        ({"fault_code": "backend_timeout"}, "backend_timeout"),
        ({"last_processed_capture_mono_ns": None}, "no_processed_capture"),
        ({"last_processed_capture_mono_ns": NOW + 1}, "processed_capture_in_future"),
        ({"last_processed_capture_mono_ns": NOW - 400_000_000}, "stale_processed_capture"),
    ],
)
def test_health_failures_invalidate_current_value(overrides, reason) -> None:
    gated = gate_hypothesis_by_health(current(), health(**overrides), now_mono_ns=NOW)

    assert gated.state == HypothesisState.UNAVAILABLE
    assert gated.has_value is False
    assert gated.value_kph is None
    assert gated.usable_for_advisory is False
    assert gated.unavailable_reason == reason


def test_backend_fault_reason_is_preserved_when_backend_unavailable() -> None:
    gated = gate_hypothesis_by_health(
        current(),
        health(backend_available=False, fault_code="model_hash_mismatch"),
        now_mono_ns=NOW,
    )

    assert gated.unavailable_reason == "model_hash_mismatch"


def test_health_age_threshold_is_explicit() -> None:
    profile = HealthGateProfile(max_processed_capture_age_ns=500_000_000)
    hypothesis = current()
    gated = gate_hypothesis_by_health(
        hypothesis,
        health(last_processed_capture_mono_ns=NOW - 400_000_000),
        now_mono_ns=NOW,
        profile=profile,
    )

    assert gated is hypothesis

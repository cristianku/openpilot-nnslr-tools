"""Perception-health gating for advisory Vision hypotheses."""
from __future__ import annotations

from dataclasses import dataclass

from .types import HypothesisState, LimitHypothesis, PerceptionHealth


@dataclass(frozen=True)
class HealthGateProfile:
    max_processed_capture_age_ns: int = 350_000_000

    def __post_init__(self) -> None:
        if type(self.max_processed_capture_age_ns) is not int or self.max_processed_capture_age_ns <= 0:
            raise ValueError("max_processed_capture_age_ns must be a positive integer")


DEFAULT_HEALTH_GATE_PROFILE = HealthGateProfile()


def unavailable_hypothesis(reason: str, *, last_context_mono_ns: int | None = None) -> LimitHypothesis:
    if not isinstance(reason, str) or not reason:
        raise ValueError("unavailable reason must be non-empty")
    return LimitHypothesis(
        has_value=False,
        value_kph=None,
        state=HypothesisState.UNAVAILABLE,
        last_context_mono_ns=last_context_mono_ns,
        unavailable_reason=reason,
        usable_for_advisory=False,
    )


def gate_hypothesis_by_health(
    hypothesis: LimitHypothesis,
    health: PerceptionHealth,
    *,
    now_mono_ns: int,
    profile: HealthGateProfile = DEFAULT_HEALTH_GATE_PROFILE,
) -> LimitHypothesis:
    """Invalidate a Vision hypothesis when perception health is not usable.

    This function never attempts recovery and never reuses a stale value.
    """
    if type(now_mono_ns) is not int or now_mono_ns <= 0:
        raise ValueError("now_mono_ns must be a positive integer")

    last_capture = health.last_processed_capture_mono_ns
    if not health.backend_available:
        return unavailable_hypothesis(
            health.fault_code or "backend_unavailable",
            last_context_mono_ns=hypothesis.last_context_mono_ns,
        )
    if health.overflow:
        return unavailable_hypothesis(
            "detection_overflow",
            last_context_mono_ns=hypothesis.last_context_mono_ns,
        )
    if health.fault_code:
        return unavailable_hypothesis(
            health.fault_code,
            last_context_mono_ns=hypothesis.last_context_mono_ns,
        )
    if last_capture is None:
        return unavailable_hypothesis(
            "no_processed_capture",
            last_context_mono_ns=hypothesis.last_context_mono_ns,
        )
    age = now_mono_ns - last_capture
    if age < 0:
        return unavailable_hypothesis(
            "processed_capture_in_future",
            last_context_mono_ns=hypothesis.last_context_mono_ns,
        )
    if age > profile.max_processed_capture_age_ns:
        return unavailable_hypothesis(
            "stale_processed_capture",
            last_context_mono_ns=hypothesis.last_context_mono_ns,
        )
    return hypothesis

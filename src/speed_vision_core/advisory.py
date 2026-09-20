"""Read-only CAR / MAP / VISION advisory comparison.

This module never selects a target speed and never mutates operational source
state. It only reports diagnostic agreement/conflict and a separate display
decision for the Vision hypothesis.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .types import (
    AdvisoryComparison,
    Agreement,
    DisplayDecision,
    HypothesisState,
    LimitHypothesis,
    SourceSnapshot,
)


@dataclass(frozen=True)
class AdvisoryProfile:
    """Explicit comparator thresholds; freeze before an evaluation run."""

    agreement_tolerance_kph: float = 1.0
    max_operational_source_age_ns: int = 2_000_000_000
    max_vision_age_ns: int = 350_000_000

    def __post_init__(self) -> None:
        if (
            type(self.agreement_tolerance_kph) not in (int, float)
            or isinstance(self.agreement_tolerance_kph, bool)
            or not math.isfinite(self.agreement_tolerance_kph)
            or self.agreement_tolerance_kph < 0
        ):
            raise ValueError("agreement_tolerance_kph must be finite and >= 0")
        for name, value in (
            ("max_operational_source_age_ns", self.max_operational_source_age_ns),
            ("max_vision_age_ns", self.max_vision_age_ns),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


DEFAULT_ADVISORY_PROFILE = AdvisoryProfile()


def _source_age(snapshot: SourceSnapshot | None, now_mono_ns: int) -> int | None:
    if snapshot is None or snapshot.observed_mono_ns is None or snapshot.age_unknown:
        return None
    age = now_mono_ns - snapshot.observed_mono_ns
    return age if age >= 0 else None


def _vision_age(vision: LimitHypothesis | None, now_mono_ns: int) -> int | None:
    if vision is None or vision.last_context_mono_ns is None:
        return None
    age = now_mono_ns - vision.last_context_mono_ns
    return age if age >= 0 else None


def _usable_source_value_kph(
    snapshot: SourceSnapshot | None,
    age_ns: int | None,
    profile: AdvisoryProfile,
) -> float | None:
    if snapshot is None or not snapshot.valid or snapshot.value_mps is None:
        return None
    if type(snapshot.value_mps) not in (int, float) or isinstance(snapshot.value_mps, bool):
        return None
    if not math.isfinite(snapshot.value_mps) or snapshot.value_mps <= 0:
        return None
    if age_ns is None or age_ns > profile.max_operational_source_age_ns:
        return None
    return float(snapshot.value_mps) * 3.6


def compare_advisory_sources(
    car: SourceSnapshot | None,
    map_source: SourceSnapshot | None,
    vision: LimitHypothesis | None,
    *,
    now_mono_ns: int,
    profile: AdvisoryProfile = DEFAULT_ADVISORY_PROFILE,
) -> AdvisoryComparison:
    """Compare independent source snapshots for advisory display only."""
    if type(now_mono_ns) is not int or now_mono_ns <= 0:
        raise ValueError("now_mono_ns must be a positive integer")

    car_age = _source_age(car, now_mono_ns)
    map_age = _source_age(map_source, now_mono_ns)
    vision_age = _vision_age(vision, now_mono_ns)

    if vision is None:
        return AdvisoryComparison(
            car=car,
            map=map_source,
            vision=None,
            car_age_ns=car_age,
            map_age_ns=map_age,
            vision_age_ns=None,
            agreement=Agreement.INSUFFICIENT_DATA,
            display_decision=DisplayDecision.HIDE,
            reason="vision_absent",
        )

    if vision.state == HypothesisState.UNAVAILABLE:
        return AdvisoryComparison(
            car=car,
            map=map_source,
            vision=vision,
            car_age_ns=car_age,
            map_age_ns=map_age,
            vision_age_ns=vision_age,
            agreement=Agreement.INSUFFICIENT_DATA,
            display_decision=DisplayDecision.SHOW_UNAVAILABLE,
            reason=vision.unavailable_reason or "vision_unavailable",
        )

    vision_fresh = (
        vision_age is not None
        and vision_age <= profile.max_vision_age_ns
    )
    if not vision_fresh:
        return AdvisoryComparison(
            car=car,
            map=map_source,
            vision=vision,
            car_age_ns=car_age,
            map_age_ns=map_age,
            vision_age_ns=vision_age,
            agreement=Agreement.INSUFFICIENT_DATA,
            display_decision=DisplayDecision.SHOW_UNAVAILABLE,
            reason="vision_stale_or_age_unknown",
        )

    if vision.state == HypothesisState.AHEAD and vision.has_value:
        return AdvisoryComparison(
            car=car,
            map=map_source,
            vision=vision,
            car_age_ns=car_age,
            map_age_ns=map_age,
            vision_age_ns=vision_age,
            agreement=Agreement.INSUFFICIENT_DATA,
            display_decision=DisplayDecision.SHOW_AHEAD,
            reason="vision_ahead",
        )

    if vision.state in (HypothesisState.OBSERVED, HypothesisState.UNCERTAIN):
        return AdvisoryComparison(
            car=car,
            map=map_source,
            vision=vision,
            car_age_ns=car_age,
            map_age_ns=map_age,
            vision_age_ns=vision_age,
            agreement=Agreement.INSUFFICIENT_DATA,
            display_decision=DisplayDecision.SHOW_OBSERVED,
            reason=vision.unavailable_reason or "vision_not_current",
        )

    if (
        vision.state != HypothesisState.CURRENT
        or not vision.usable_for_advisory
        or not vision.has_value
        or vision.value_kph is None
    ):
        return AdvisoryComparison(
            car=car,
            map=map_source,
            vision=vision,
            car_age_ns=car_age,
            map_age_ns=map_age,
            vision_age_ns=vision_age,
            agreement=Agreement.INSUFFICIENT_DATA,
            display_decision=DisplayDecision.HIDE,
            reason="vision_not_usable_current",
        )

    car_kph = _usable_source_value_kph(car, car_age, profile)
    map_kph = _usable_source_value_kph(map_source, map_age, profile)
    available = [
        ("car", car_kph),
        ("map", map_kph),
    ]
    available = [(name, value) for name, value in available if value is not None]

    if not available:
        return AdvisoryComparison(
            car=car,
            map=map_source,
            vision=vision,
            car_age_ns=car_age,
            map_age_ns=map_age,
            vision_age_ns=vision_age,
            agreement=Agreement.INSUFFICIENT_DATA,
            display_decision=DisplayDecision.SHOW_CURRENT,
            reason="operational_sources_missing_or_stale",
        )

    disagree = [
        name
        for name, value in available
        if abs(value - vision.value_kph) > profile.agreement_tolerance_kph
    ]
    if disagree:
        return AdvisoryComparison(
            car=car,
            map=map_source,
            vision=vision,
            car_age_ns=car_age,
            map_age_ns=map_age,
            vision_age_ns=vision_age,
            agreement=Agreement.CONFLICT,
            display_decision=DisplayDecision.SHOW_CONFLICT,
            reason="conflict_with_" + "_and_".join(disagree),
        )

    return AdvisoryComparison(
        car=car,
        map=map_source,
        vision=vision,
        car_age_ns=car_age,
        map_age_ns=map_age,
        vision_age_ns=vision_age,
        agreement=Agreement.AGREE,
        display_decision=DisplayDecision.SHOW_CURRENT,
        reason="available_operational_sources_agree",
    )

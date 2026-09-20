from __future__ import annotations

from speed_vision_core.advisory import AdvisoryProfile, compare_advisory_sources
from speed_vision_core.types import (
    Agreement,
    DisplayDecision,
    HypothesisState,
    LimitHypothesis,
    SourceKind,
    SourceSnapshot,
)


NOW = 10_000_000_000


def source(kind: SourceKind, kph: float, *, age_ns: int = 100_000_000, valid: bool = True):
    return SourceSnapshot(
        source=kind,
        provenance=f"{kind.value}-test",
        valid=valid,
        value_mps=kph / 3.6,
        observed_mono_ns=NOW - age_ns,
    )


def vision(
    state: HypothesisState,
    *,
    value: int | None = 80,
    age_ns: int = 100_000_000,
    usable: bool | None = None,
):
    has_value = value is not None
    if usable is None:
        usable = state == HypothesisState.CURRENT
    return LimitHypothesis(
        has_value=has_value,
        value_kph=value,
        state=state,
        track_id="track-1",
        last_context_mono_ns=NOW - age_ns,
        usable_for_advisory=usable,
    )


def test_absent_vision_never_changes_operational_display() -> None:
    result = compare_advisory_sources(
        source(SourceKind.CAR, 80),
        source(SourceKind.MAP, 80),
        None,
        now_mono_ns=NOW,
    )

    assert result.agreement == Agreement.INSUFFICIENT_DATA
    assert result.display_decision == DisplayDecision.HIDE
    assert result.reason == "vision_absent"


def test_observed_value_is_not_compared_as_current() -> None:
    result = compare_advisory_sources(
        source(SourceKind.CAR, 30),
        source(SourceKind.MAP, 50),
        vision(HypothesisState.OBSERVED, value=80, usable=False),
        now_mono_ns=NOW,
    )

    assert result.agreement == Agreement.INSUFFICIENT_DATA
    assert result.display_decision == DisplayDecision.SHOW_OBSERVED


def test_ahead_value_is_separate_not_conflict() -> None:
    result = compare_advisory_sources(
        source(SourceKind.CAR, 50),
        source(SourceKind.MAP, 50),
        vision(HypothesisState.AHEAD, value=80, usable=False),
        now_mono_ns=NOW,
    )

    assert result.agreement == Agreement.INSUFFICIENT_DATA
    assert result.display_decision == DisplayDecision.SHOW_AHEAD


def test_stale_vision_is_unavailable_even_if_value_exists() -> None:
    result = compare_advisory_sources(
        source(SourceKind.CAR, 80),
        source(SourceKind.MAP, 80),
        vision(HypothesisState.CURRENT, age_ns=500_000_000),
        now_mono_ns=NOW,
    )

    assert result.agreement == Agreement.INSUFFICIENT_DATA
    assert result.display_decision == DisplayDecision.SHOW_UNAVAILABLE
    assert result.reason == "vision_stale_or_age_unknown"


def test_current_vision_with_no_operational_sources_is_shown_without_agreement_claim() -> None:
    result = compare_advisory_sources(
        None,
        None,
        vision(HypothesisState.CURRENT),
        now_mono_ns=NOW,
    )

    assert result.agreement == Agreement.INSUFFICIENT_DATA
    assert result.display_decision == DisplayDecision.SHOW_CURRENT
    assert result.reason == "operational_sources_missing_or_stale"


def test_current_vision_agrees_with_fresh_car_and_map() -> None:
    result = compare_advisory_sources(
        source(SourceKind.CAR, 80),
        source(SourceKind.MAP, 80),
        vision(HypothesisState.CURRENT),
        now_mono_ns=NOW,
    )

    assert result.agreement == Agreement.AGREE
    assert result.display_decision == DisplayDecision.SHOW_CURRENT


def test_any_fresh_operational_disagreement_is_explicit_conflict() -> None:
    result = compare_advisory_sources(
        source(SourceKind.CAR, 80),
        source(SourceKind.MAP, 100),
        vision(HypothesisState.CURRENT),
        now_mono_ns=NOW,
    )

    assert result.agreement == Agreement.CONFLICT
    assert result.display_decision == DisplayDecision.SHOW_CONFLICT
    assert result.reason == "conflict_with_map"


def test_stale_map_is_ignored_while_fresh_car_can_agree() -> None:
    profile = AdvisoryProfile(
        agreement_tolerance_kph=1.0,
        max_operational_source_age_ns=2_000_000_000,
        max_vision_age_ns=350_000_000,
    )
    result = compare_advisory_sources(
        source(SourceKind.CAR, 80),
        source(SourceKind.MAP, 100, age_ns=3_000_000_000),
        vision(HypothesisState.CURRENT),
        now_mono_ns=NOW,
        profile=profile,
    )

    assert result.agreement == Agreement.AGREE
    assert result.map_age_ns == 3_000_000_000


def test_tolerance_is_explicit_profile_configuration() -> None:
    strict = AdvisoryProfile(agreement_tolerance_kph=0.1)
    relaxed = AdvisoryProfile(agreement_tolerance_kph=2.0)
    car = source(SourceKind.CAR, 81)
    current = vision(HypothesisState.CURRENT, value=80)

    strict_result = compare_advisory_sources(car, None, current, now_mono_ns=NOW, profile=strict)
    relaxed_result = compare_advisory_sources(car, None, current, now_mono_ns=NOW, profile=relaxed)

    assert strict_result.agreement == Agreement.CONFLICT
    assert relaxed_result.agreement == Agreement.AGREE


def test_current_flag_without_advisory_usability_is_hidden() -> None:
    result = compare_advisory_sources(
        source(SourceKind.CAR, 80),
        None,
        vision(HypothesisState.CURRENT, usable=False),
        now_mono_ns=NOW,
    )

    assert result.agreement == Agreement.INSUFFICIENT_DATA
    assert result.display_decision == DisplayDecision.HIDE
    assert result.reason == "vision_not_usable_current"

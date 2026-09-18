# [nnslr-t1] - START
"""Contract tests for ``speed_vision_core.types`` (plan §7).

Pure CPU, stdlib-only. No model, no openpilot, no device, no real data.
These tests pin the binding invariants:

- ``0`` is never a speed-limit value; absence stays absent;
- value/None consistency is enforced at construction and at deserialization;
- monotonic timestamps are comparable only inside the same session;
- validation is deterministic, pure, and reports stable reason codes;
- nothing in the contract carries a target speed / actuation request.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from speed_vision_core import types
from speed_vision_core.types import (
    DEFAULT_MAX_EVIDENCE_AGE_NS,
    MAX_DETECTIONS_PER_BATCH,
    MIN_VALUE_KPH,
    MAX_VALUE_KPH,
    REASON_CODE_REGISTRY,
    ReasonCode,
    NnslerContractError,
    Agreement,
    ApplicabilityVerdict,
    CaptureReference,
    DisplayDecision,
    FrameRef,
    FrameIdentity,
    HypothesisState,
    LimitHypothesis,
    NnslerContractError as _ECE,  # noqa: F401  (alias kept for clarity)
    ObservationBatch,
    PassageVerdict,
    SignFamily,
    SignTrack,
    SourceKind,
    SourceSnapshot,
    StreamId,
    ValueState,
    WallClockMapping,
    AdvisoryComparison,
    Detection,
    same_clock_domain,
    to_session_mono_ns,
)

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

CAPTURE_NS = 1_000_000_000_000  # 1000 s of session time, ns
NOW_NS = CAPTURE_NS + 10_000_000  # +10 ms


def make_frame(session: str = "s1", stream: str = "narrow_road",
               frame_id: int = 1, capture_ns: int = CAPTURE_NS,
               width: int = 1920, height: int = 1080) -> FrameRef:
    return FrameRef(
        session_id=session,
        stream=StreamId(stream),
        frame_id=frame_id,
        capture_mono_ns=capture_ns,
        capture_reference=CaptureReference.SOF,
        native_width=width,
        native_height=height,
        preprocessing_identity="none",
    )


def make_det(frame: FrameRef, value_kph: int | None = 50,
             value_state: ValueState = ValueState.VALUE,
             score: float = 0.9) -> Detection:
    return Detection(
        frame=frame,
        bbox_xyxy=(100.0, 100.0, 300.0, 380.0),
        sign_family=SignFamily.MAX_SPEED,
        value_state=value_state,
        value_kph=value_kph,
        detection_score=score,
        classification_score=score,
        supported_domain=True,
        observation_id="",
    )


def make_batch(frame: FrameRef, detections: tuple[Detection, ...] = (
    make_det(frame),
),
               now_ns: int = NOW_NS,
               processed_ns: int | None = None) -> ObservationBatch:
    return ObservationBatch(
        frame=frame,
        model_hash="m",
        config_hash="c",
        rulepack_hash="r",
        processed_mono_ns=processed_ns if processed_ns is not None else now_ns,
        backend="synthetic",
        backend_status="ok",
        detections=detections,
    )


# ---------------------------------------------------------------------------
# Value invariants: zero is never a limit; absence stays absent
# ---------------------------------------------------------------------------

def test_zero_is_never_a_value_in_validation() -> None:
    frame = make_frame()
    det = make_det(frame, value_kph=0)  # VALUE with 0
    batch = make_batch(frame, (det,))
    result = types.validate_batch(batch, now_mono_ns=NOW_NS, expected_session_id="s1")
    assert not result.accepted
    assert ReasonCode.INVALID_VALUE_ZERO in result.reason_codes


def test_zero_with_non_value_state_is_a_hard_error() -> None:
    frame = make_frame()
    with pytest.raises(NnslerContractError) as exc:
        make_det(frame, value_kph=0, value_state=ValueState.UNKNOWN)
    assert exc.value.reason_code == ReasonCode.INVALID_VALUE_CONSISTENCY


def test_value_without_number_is_a_hard_error() -> None:
    frame = make_frame()
    with pytest.raises(NnslerContractError) as exc:
        make_det(frame, value_kph=None, value_state=ValueState.VALUE)
    assert exc.value.reason_code == ReasonCode.INVALID_VALUE_CONSISTENCY


def test_non_value_state_must_not_carry_a_number() -> None:
    frame = make_frame()
    with pytest.raises(NnslerContractError) as exc:
        make_det(frame, value_kph=50, value_state=ValueState.UNREADABLE)
    assert exc.value.reason_code == ReasonCode.INVALID_VALUE_CONSISTENCY


def test_value_out_of_domain_is_rejected() -> None:
    frame = make_frame()
    for bad in (0, MIN_VALUE_KPH - 1, MAX_VALUE_KPH + 1, 300, 1000):
        det = make_det(frame, value_kph=bad if bad > 0 else 0)
        batch = make_batch(frame, (det,))
        result = types.validate_batch(batch, now_mono_ns=NOW_NS, expected_session_id="s1")
        if bad == 0:
            assert ReasonCode.INVALID_VALUE_ZERO in result.reason_codes
        else:
            assert ReasonCode.INVALID_VALUE_DOMAIN in result.reason_codes


def test_boundary_values_are_accepted() -> None:
    frame = make_frame()
    for ok in (MIN_VALUE_KPH, MAX_VALUE_KPH, 50, 130):
        det = make_det(frame, value_kph=ok)
        batch = make_batch(frame, (det,))
        result = types.validate_batch(batch, now_mono_ns=NOW_NS, expected_session_id="s1")
        assert result.accepted, f"value {ok} should be accepted"


# ---------------------------------------------------------------------------
# Clock domains
# ---------------------------------------------------------------------------

def test_wall_clock_mapping_requires_verification() -> None:
    mapping = WallClockMapping(session_id="s1", offset_ns=42, verified=False)
    with pytest.raises(NnslerContractError) as exc:
        to_session_mono_ns(1_000_000, mapping, "s1")
    assert exc.value.reason_code == ReasonCode.INVALID_CLOCK_DOMAIN


def test_wall_clock_mapping_session_mismatch() -> None:
    mapping = WallClockMapping(session_id="s1", offset_ns=42, verified=True)
    with pytest.raises(NnslerContractError) as exc:
        to_session_mono_ns(1_000_000, mapping, "s2")
    assert exc.value.reason_code == ReasonCode.INVALID_CLOCK_DOMAIN


def test_wall_clock_mapping_verified_converts() -> None:
    mapping = WallClockMapping(session_id="s1", offset_ns=42, verified=True)
    assert to_session_mono_ns(1_000_000, mapping, "s1") == 1_000_042


def test_same_clock_domain_is_session_bound() -> None:
    a = make_frame(session="s1")
    b = make_frame(session="s1")
    c = make_frame(session="s2")
    assert same_clock_domain(a, b)
    assert not same_clock_domain(a, c)


def test_future_capture_is_rejected() -> None:
    frame = make_frame(capture_ns=CAPTURE_NS + 100_000_000)  # after NOW_NS
    batch = make_batch(frame, now_ns=NOW_NS)
    result = types.validate_batch(batch, now_mono_ns=NOW_NS, expected_session_id="s1")
    assert not result.accepted
    assert ReasonCode.INVALID_CAPTURE_TIMESTAMP_FUTURE in result.reason_codes


def test_stale_evidence_is_rejected() -> None:
    frame = make_frame(capture_ns=CAPTURE_NS - 2 * DEFAULT_MAX_EVIDENCE_AGE_NS)
    batch = make_batch(frame, now_ns=NOW_NS)
    result = types.validate_batch(batch, now_mono_ns=NOW_NS, expected_session_id="s1")
    assert not result.accepted
    assert ReasonCode.STALE_EVIDENCE in result.reason_codes


def test_fresh_capture_within_window_is_accepted() -> None:
    frame = make_frame(capture_ns=NOW_NS - DEFAULT_MAX_EVIDENCE_AGE_NS // 2)
    batch = make_batch(frame, now_ns=NOW_NS)
    result = types.validate_batch(batch, now_mono_ns=NOW_NS, expected_session_id="s1")
    assert result.accepted


# ---------------------------------------------------------------------------
# Determinism, purity, and reason-code stability
# ---------------------------------------------------------------------------

def test_validation_is_deterministic() -> None:
    frame = make_frame()
    det = make_det(frame, value_kph=0)
    batch = make_batch(frame, (det,))
    r1 = types.validate_batch(batch, now_mono_ns=NOW_NS, expected_session_id="s1")
    r2 = types.validate_batch(batch, now_mono_ns=NOW_NS, expected_session_id="s1")
    assert r1.reason_codes == r2.reason_codes
    assert r1.rejected_indices == r2.rejected_indices
    assert r1.accepted == r2.accepted


def test_validation_does_not_mutate_input() -> None:
    frame = make_frame()
    det = make_det(frame, value_kph=0)
    batch = make_batch(frame, (det,))
    before = copy.deepcopy(batch)
    types.validate_batch(batch, now_mono_ns=NOW_NS, expected_session_id="s1")
    assert batch == before


def test_all_reported_codes_are_in_registry() -> None:
    # Every code the validator can emit must be a documented, stable code.
    frame = make_frame()
    bad = make_det(frame, value_kph=0)
    batch = make_batch(frame, (bad,))
    result = types.validate_batch(batch, now_mono_ns=NOW_NS, expected_session_id="s1")
    assert set(result.reason_codes) <= REASON_CODE_REGISTRY


def test_reason_code_registry_is_exhaustive() -> None:
    # Registry must match the class attributes exactly (no typos, no strays).
    class_codes = frozenset(
        v for k, v in vars(ReasonCode).items()
        if not k.startswith("_") and isinstance(v, str)
    )
    assert class_codes == REASON_CODE_REGISTRY


def test_duplicate_frame_is_rejected() -> None:
    frame = make_frame()
    batch = make_batch(frame)
    recent = frozenset({frame.identity()})
    result = types.validate_batch(
        batch, now_mono_ns=NOW_NS, expected_session_id="s1", recent_frame_ids=recent
    )
    assert not result.accepted
    assert result.reason_codes == (ReasonCode.DUPLICATE_FRAME,)


def test_detection_overflow_is_rejected() -> None:
    frame = make_frame()
    dets = tuple(make_det(frame) for _ in range(MAX_DETECTIONS_PER_BATCH + 1))
    batch = make_batch(frame, dets)
    result = types.validate_batch(batch, now_mono_ns=NOW_NS, expected_session_id="s1")
    assert not result.accepted
    assert result.reason_codes == (ReasonCode.DETECTION_OVERFLOW,)


def test_empty_batch_is_accepted_as_unavailable() -> None:
    frame = make_frame()
    batch = make_batch(frame, ())
    result = types.validate_batch(batch, now_mono_ns=NOW_NS, expected_session_id="s1")
    assert result.accepted
    assert result.reason_codes == ()


def test_partial_rejection_keeps_batch_acceptable() -> None:
    frame = make_frame()
    good = make_det(frame, value_kph=50)
    bad = make_det(frame, value_kph=0)
    batch = make_batch(frame, (good, bad))
    result = types.validate_batch(batch, now_mono_ns=NOW_NS, expected_session_id="s1")
    assert result.accepted
    assert result.rejected_indices == (1,)
    assert ReasonCode.INVALID_VALUE_ZERO in result.reason_codes


def test_session_mismatch_is_rejected() -> None:
    frame = make_frame(session="s1")
    batch = make_batch(frame)
    result = types.validate_batch(batch, now_mono_ns=NOW_NS, expected_session_id="s2")
    assert not result.accepted
    assert ReasonCode.INVALID_SESSION_MISMATCH in result.reason_codes


def test_schema_version_mismatch_is_rejected() -> None:
    frame = make_frame()
    batch = make_batch(frame)
    bad = ObservationBatch(
        frame=batch.frame, model_hash=batch.model_hash, config_hash=batch.config_hash,
        rulepack_hash=batch.rulepack_hash, processed_mono_ns=batch.processed_mono_ns,
        backend=batch.backend, backend_status=batch.backend_status,
        detections=batch.detections, schema_version=99,
    )
    result = types.validate_batch(bad, now_mono_ns=NOW_NS, expected_session_id="s1")
    assert not result.accepted
    assert ReasonCode.INVALID_SCHEMA_VERSION in result.reason_codes


def test_negative_capture_is_rejected() -> None:
    frame = make_frame(capture_ns=-1)
    batch = make_batch(frame, now_ns=NOW_NS)
    result = types.validate_batch(batch, now_mono_ns=NOW_NS, expected_session_id="s1")
    assert not result.accepted
    assert ReasonCode.INVALID_CAPTURE_TIMESTAMP_NEGATIVE in result.reason_codes


def test_negative_latency_is_rejected() -> None:
    frame = make_frame()
    batch = make_batch(frame, processed_ns=CAPTURE_NS - 5)  # processed before capture
    result = types.validate_batch(batch, now_mono_ns=NOW_NS, expected_session_id="s1")
    assert not result.accepted
    assert ReasonCode.INVALID_NEGATIVE_LATENCY in result.reason_codes


def test_bad_bbox_is_rejected() -> None:
    frame = make_frame()
    oob = Detection(frame=frame, bbox_xyxy=(0.0, 0.0, frame.native_width + 1, 100.0),
                    sign_family=SignFamily.MAX_SPEED, value_state=ValueState.VALUE,
                    value_kph=50, detection_score=0.9, classification_score=0.9)
    degenerate = Detection(frame=frame, bbox_xyxy=(100.0, 100.0, 100.0, 380.0),
                           sign_family=SignFamily.MAX_SPEED, value_state=ValueState.VALUE,
                           value_kph=50, detection_score=0.9, classification_score=0.9)
    for det in (oob, degenerate):
        batch = make_batch(frame, (det,))
        result = types.validate_batch(batch, now_mono_ns=NOW_NS, expected_session_id="s1")
        assert not result.accepted
        assert set(result.reason_codes) & {
            ReasonCode.INVALID_BOX_OUT_OF_BOUNDS,
            ReasonCode.INVALID_BOX_DEGENERATE,
        }


# ---------------------------------------------------------------------------
# Serialization: presence-preserving round-trip; zero never reinterpreted
# ---------------------------------------------------------------------------

def test_serialization_round_trip_preserves_value() -> None:
    frame = make_frame()
    det = make_det(frame, value_kph=50)
    batch = make_batch(frame, (det,))
    again = types.batch_from_dict(types.batch_to_dict(batch))
    assert again == batch
    assert again.detections[0].value_kph == 50


def test_serialization_round_trip_preserves_absence() -> None:
    frame = make_frame()
    det = make_det(frame, value_kph=None, value_state=ValueState.UNKNOWN)
    batch = make_batch(frame, (det,))
    payload = types.batch_to_dict(batch)
    assert payload["detections"][0]["value_kph"] is None  # absent stays absent
    again = types.batch_from_dict(payload)
    assert again == batch
    assert again.detections[0].value_kph is None
    assert again.detections[0].value_state == ValueState.UNKNOWN


def test_deserializer_rejects_number_without_value_state() -> None:
    frame = make_frame()
    det = make_det(frame, value_kph=50)
    batch = make_batch(frame, (det,))
    payload = types.batch_to_dict(batch)
    # Corrupt: keep the number but claim the value is not present.
    payload["detections"][0]["value_state"] = ValueState.UNKNOWN.value
    with pytest.raises(NnslerContractError) as exc:
        types.batch_from_dict(payload)
    assert exc.value.reason_code == ReasonCode.INVALID_VALUE_CONSISTENCY


def test_valid_fixture_round_trip_is_stable() -> None:
    fixtures = (
        Path(__file__).resolve().parents[2]
        / "src" / "nnslr_tools" / "fixtures" / "valid_batch.json"
    )
    payload = json.loads(fixtures.read_text(encoding="utf-8"))
    batch = types.batch_from_dict(payload)
    again = types.batch_to_dict(batch)
    assert json.dumps(again, sort_keys=True) == json.dumps(payload, sort_keys=True)


# ---------------------------------------------------------------------------
# Advisory-only: no target speed, no actuation anywhere in the contract
# ---------------------------------------------------------------------------

def test_limit_hypothesis_invariants() -> None:
    with pytest.raises(NnslerContractError):
        LimitHypothesis(has_value=True, value_kph=None, state=HypothesisState.OBSERVED)
    with pytest.raises(NnslerContractError):
        LimitHypothesis(has_value=False, value_kph=50, state=HypothesisState.OBSERVED)
    with pytest.raises(NnslerContractError):
        LimitHypothesis(has_value=True, value_kph=300, state=HypothesisState.CURRENT)
    ok = LimitHypothesis(has_value=True, value_kph=50, state=HypothesisState.CURRENT,
                         usable_for_advisory=True)
    assert ok.value_kph == 50


def test_advisory_comparison_has_no_target_speed() -> None:
    import dataclasses

    fields = {f.name for f in dataclasses.fields(AdvisoryComparison)}
    banned = {"target_speed", "target_speed_mps", "set_speed", "actuation", "command"}
    assert not (fields & banned)


def test_no_type_carries_a_target_speed() -> None:
    import dataclasses

    for cls in (
        FrameRef, Detection, ObservationBatch, SignTrack, SourceSnapshot,
        LimitHypothesis, AdvisoryComparison,
    ):
        names = {f.name for f in dataclasses.fields(cls)}
        banned = {"target_speed", "target_speed_mps", "set_speed", "actuation_request"}
        assert not (names & banned), f"{cls.__name__} carries a control field"


def test_source_snapshot_uses_meters_per_second() -> None:
    import dataclasses

    names = {f.name for f in dataclasses.fields(SourceSnapshot)}
    assert "value_mps" in names
    assert "value_kph" not in names  # operational unit is m/s; kph is display-only


def test_vision_is_not_an_operational_source_kind() -> None:
    assert set(SourceKind) == {SourceKind.CAR, SourceKind.MAP}

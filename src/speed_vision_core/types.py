"""NNSLR domain types and validity invariants (plan section 7).

This module is the hardware-independent reference contract for the NN Vision
Speed Limit project. It is deliberately pure Python 3.11+ stdlib: it must
import and run without CUDA, PyTorch, tinygrad, openpilot, cereal or any
vehicle hardware. Serialization adapters (Cap'n Proto, JSON) are thin layers
around these types; the invariants live here.

Binding rules implemented (plan §7.1/§7.3/§7.4):

- ``value_kph: int | None`` — an absent value stays absent. ``0`` is never a
  speed-limit value; ``unknown``/``unreadable``/``not_applicable``/
  ``unavailable`` remain distinct outcomes (see :class:`ValueState`).
- Monotonic capture timestamps are comparable only inside the same
  session/clock domain. Wall-clock/GPS time is a separate domain and is
  mapped only through a verified :class:`WallClockMapping`.
- Every validation result carries a finite, documented reason code from
  :class:`ReasonCode`; validation is deterministic and does not mutate the
  input.
- No type in this subsystem carries a target speed, an actuation request or a
  cruise set-speed instruction (plan §7.3 last bullet, §9).
"""

# [nnslr-t1] - START
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Contract version for :class:`ObservationBatch`. Bump deliberately; an
#: unknown version invalidates the whole batch (ReasonCode.INVALID_SCHEMA_VERSION).
SUPPORTED_SCHEMA_VERSION = 1

#: Maximum detections per batch (plan §7.2: cap eight per publication; report
#: overflow instead of silently selecting the first sign).
MAX_DETECTIONS_PER_BATCH = 8

#: Observation freshness at consumption, 0.35 s (plan §8.1 initial profile).
DEFAULT_MAX_EVIDENCE_AGE_NS = 350_000_000

#: Plausible numeric domain for Swiss speed-limit signs (km/h). ``0`` is
#: outside the domain by definition — it is "no value", not a limit.
MIN_VALUE_KPH = 5
MAX_VALUE_KPH = 250


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class StreamId(str, Enum):
    """Camera stream identity. Mirrors the runtime narrow/wide/cabin naming."""

    NARROW_ROAD = "narrow_road"
    WIDE_ROAD = "wide_road"
    CABIN = "cabin"
    #: Synthetic frames produced by fixtures/tests; never vehicle data.
    SYNTHETIC = "synthetic"


class CaptureReference(str, Enum):
    """Which edge of the exposure the capture timestamp refers to."""

    SOF = "sof"
    EOF = "eof"
    UNKNOWN = "unknown"


class SignFamily(str, Enum):
    """Swiss sign families the recognizer targets (plan §5 vocabulary).

    These are perception classes only: recognition and legal/rule
    interpretation remain separate (plan §5/§10). A conditional panel
    (e.g. time restriction) is carried by ``Detection.linked_panel_boxes``,
    not as a family. ``OTHER_SIGN``/``NOT_A_SIGN``/``UNREADABLE`` exist so
    excluded semantics can be *surfaced* (e.g. as unknown) rather than
    silently dropped (plan §5 challenge-group requirement)."""

    MAX_SPEED = "max_speed"
    CANCELLATION = "cancellation"
    ZONE = "zone"
    VARIABLE_DISPLAY = "variable_display"
    OTHER_SIGN = "other_sign"
    NOT_A_SIGN = "not_a_sign"
    UNREADABLE = "unreadable"


class ValueState(str, Enum):
    """Distinct outcomes for a sign's numeric value (plan §7.3).

    None of these means zero speed or unrestricted speed.
    """

    #: A numeric value was accepted into ``value_kph``.
    VALUE = "value"
    #: A sign is present but its value cannot be determined from the evidence.
    UNKNOWN = "unknown"
    #: A sign is present but the value is illegible (blur, occlusion, glare).
    UNREADABLE = "unreadable"
    #: The value is conditional and not applicable in the current context.
    NOT_APPLICABLE = "not_applicable"
    #: No value is obtainable at all (no sign in frame / backend offline).
    UNAVAILABLE = "unavailable"


class SourceKind(str, Enum):
    """Operational speed-limit sources kept conceptually independent (D8).

    Vision is deliberately NOT a member: it is carried by
    :class:`LimitHypothesis`, never merged into this enum.
    """

    CAR = "car"
    MAP = "map"


class HypothesisState(str, Enum):
    """Lifecycle of the current Vision limit hypothesis (plan §8/§9)."""

    NONE = "none"
    #: Seen, but road ownership/passage not yet established.
    OBSERVED = "observed"
    #: Owned by our road, ahead of the vehicle.
    AHEAD = "ahead"
    #: All of observed + owned + passage verified: the applicable limit.
    CURRENT = "current"
    #: Supporting evidence degraded; must not be displayed as current.
    UNCERTAIN = "uncertain"
    UNAVAILABLE = "unavailable"


class ApplicabilityVerdict(str, Enum):
    OWN = "own"
    OTHER = "other"
    UNKNOWN = "unknown"


class PassageVerdict(str, Enum):
    AHEAD = "ahead"
    PASSED = "passed"
    UNKNOWN = "unknown"


class Agreement(str, Enum):
    AGREE = "agree"
    CONFLICT = "conflict"
    INSUFFICIENT_DATA = "insufficient_data"


class DisplayDecision(str, Enum):
    SHOW_CURRENT = "show_current"
    SHOW_AHEAD = "show_ahead"
    SHOW_OBSERVED = "show_observed"
    SHOW_UNAVAILABLE = "show_unavailable"
    SHOW_CONFLICT = "show_conflict"
    HIDE = "hide"


# ---------------------------------------------------------------------------
# Reason codes (stable, finite, documented — plan §7.4)
# ---------------------------------------------------------------------------

class ReasonCode:
    """Stable reason codes. Values are part of the contract: never renumber
    or rename; add new codes as needed."""

    OK = "ok"
    # Batch-level
    INVALID_SCHEMA_VERSION = "invalid_schema_version"
    INVALID_SESSION_MISMATCH = "invalid_session_mismatch"
    INVALID_CLOCK_DOMAIN = "invalid_clock_domain"
    INVALID_CAPTURE_TIMESTAMP_NEGATIVE = "invalid_capture_timestamp_negative"
    INVALID_CAPTURE_TIMESTAMP_FUTURE = "invalid_capture_timestamp_future"
    INVALID_NEGATIVE_LATENCY = "invalid_negative_latency"
    STALE_EVIDENCE = "stale_evidence"
    DUPLICATE_FRAME = "duplicate_frame"
    DETECTION_OVERFLOW = "detection_overflow"
    # Detection-level
    INVALID_NONFINITE_SCORE = "invalid_nonfinite_score"
    INVALID_SCORE_RANGE = "invalid_score_range"
    INVALID_BOX_OUT_OF_BOUNDS = "invalid_box_out_of_bounds"
    INVALID_BOX_DEGENERATE = "invalid_box_degenerate"
    INVALID_VALUE_ZERO = "invalid_value_zero"
    INVALID_VALUE_DOMAIN = "invalid_value_domain"
    INVALID_VALUE_CONSISTENCY = "invalid_value_consistency"
    INVALID_NATIVE_DIMENSIONS = "invalid_native_dimensions"
    # [nnslr-t1] - START  (strict numeric deserialization: wrong type, no coercion)
    INVALID_NUMERIC_TYPE = "invalid_numeric_type"
    # [nnslr-t1] - END


#: All documented reason codes; tests assert exhaustiveness of results against
#: this set so a typo in a code cannot slip through.
REASON_CODE_REGISTRY: frozenset[str] = frozenset(
    {
        ReasonCode.OK,
        ReasonCode.INVALID_SCHEMA_VERSION,
        ReasonCode.INVALID_SESSION_MISMATCH,
        ReasonCode.INVALID_CLOCK_DOMAIN,
        ReasonCode.INVALID_CAPTURE_TIMESTAMP_NEGATIVE,
        ReasonCode.INVALID_CAPTURE_TIMESTAMP_FUTURE,
        ReasonCode.INVALID_NEGATIVE_LATENCY,
        ReasonCode.STALE_EVIDENCE,
        ReasonCode.DUPLICATE_FRAME,
        ReasonCode.DETECTION_OVERFLOW,
        ReasonCode.INVALID_NONFINITE_SCORE,
        ReasonCode.INVALID_SCORE_RANGE,
        ReasonCode.INVALID_BOX_OUT_OF_BOUNDS,
        ReasonCode.INVALID_BOX_DEGENERATE,
        ReasonCode.INVALID_VALUE_ZERO,
        ReasonCode.INVALID_VALUE_DOMAIN,
        ReasonCode.INVALID_VALUE_CONSISTENCY,
        ReasonCode.INVALID_NATIVE_DIMENSIONS,
        # [nnslr-t1] - START  (strict numeric deserialization)
        ReasonCode.INVALID_NUMERIC_TYPE,
        # [nnslr-t1] - END
    }
)


class NnslerContractError(ValueError):
    """Raised when a payload violates a hard contract (e.g. deserializing a
    value that claims ``has_value=false`` alongside a non-None ``value_kph``).

    Carries a deterministic :class:`ReasonCode`."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(f"{reason_code}: {message}")
        self.reason_code = reason_code
        self.message = message


# ---------------------------------------------------------------------------
# Clock domains
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrameIdentity:
    """Identity of a frame inside one session/clock domain.

    Two frames with the same identity are duplicates; comparisons across
    sessions are meaningless and must not be made."""

    session_id: str
    stream: StreamId
    frame_id: int

    def key(self) -> str:
        return f"{self.session_id}:{self.stream.value}:{self.frame_id}"


@dataclass(frozen=True)
class WallClockMapping:
    """Verified conversion between wall-clock (or GPS) time and the session
    monotonic domain. ``verified`` must be True for conversions to be used;
    an unverified mapping is rejected rather than silently trusted (plan
    §7.3: map wall time only through a verified conversion)."""

    session_id: str
    #: session_mono_ns = wall_ns + offset_ns  (when verified)
    offset_ns: int
    verified: bool = False


def to_session_mono_ns(wall_ns: int, mapping: WallClockMapping, session_id: str) -> int:
    """Convert an absolute wall-clock nanosecond timestamp into the session
    monotonic domain. Raises :class:`NnslerContractError` if the mapping is
    unverified or belongs to a different session."""
    if not mapping.verified:
        raise NnslerContractError(
            ReasonCode.INVALID_CLOCK_DOMAIN,
            "wall-clock mapping is not verified; refusing conversion",
        )
    if mapping.session_id != session_id:
        raise NnslerContractError(
            ReasonCode.INVALID_CLOCK_DOMAIN,
            f"wall-clock mapping session {mapping.session_id!r} != {session_id!r}",
        )
    return wall_ns + mapping.offset_ns


def same_clock_domain(ref_a: "FrameRef", ref_b: "FrameRef") -> bool:
    """Monotonic timestamps are comparable only inside the same session
    (plan §7.3)."""
    return ref_a.session_id == ref_b.session_id


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrameRef:
    """Identity and timing of one captured frame (plan §7.1)."""

    session_id: str
    stream: StreamId
    frame_id: int
    #: Capture timestamp in the session monotonic domain, nanoseconds.
    capture_mono_ns: int
    capture_reference: CaptureReference = CaptureReference.UNKNOWN
    native_width: int = 0
    native_height: int = 0
    #: Identity of the preprocessing that produced the pixels this frame's
    #: detections refer to ("none" for raw frames).
    preprocessing_identity: str = "none"

    def identity(self) -> FrameIdentity:
        return FrameIdentity(self.session_id, self.stream, self.frame_id)

    def key(self) -> str:
        return self.identity().key()


@dataclass(frozen=True)
class Detection:
    """One detected sign in one frame (plan §7.1).

    ``value_kph`` is meaningful only when ``value_state == ValueState.VALUE``;
    in every other state it is ``None``. ``0`` is never a value."""

    frame: FrameRef
    bbox_xyxy: tuple[float, float, float, float]
    sign_family: SignFamily
    value_state: ValueState
    value_kph: int | None = None
    #: Boxes of panels linked to this sign (original pixels), if any.
    linked_panel_boxes: tuple[tuple[float, float, float, float], ...] = ()
    detection_score: float = 0.0
    classification_score: float = 0.0
    #: Whether this scene/sign family is inside the model's supported domain.
    supported_domain: bool = True
    #: Stable observation id; unique within a session.
    observation_id: str = ""

    def __post_init__(self) -> None:
        if self.value_state == ValueState.VALUE and self.value_kph is None:
            raise NnslerContractError(
                ReasonCode.INVALID_VALUE_CONSISTENCY,
                "value_state=VALUE requires an int value_kph",
            )
        if self.value_state != ValueState.VALUE and self.value_kph is not None:
            raise NnslerContractError(
                ReasonCode.INVALID_VALUE_CONSISTENCY,
                f"value_state={self.value_state.value} requires value_kph=None",
            )
        if self.observation_id == "":
            object.__setattr__(
                self, "observation_id",
                f"{self.frame.key()}#{self.bbox_xyxy}#{self.sign_family.value}",
            )


@dataclass(frozen=True)
class ObservationBatch:
    """Model output for one frame, with identity and backend status
    (plan §7.1/§7.2). Detections are bounded; overflow is reported, not
    truncated silently."""

    frame: FrameRef
    model_hash: str
    config_hash: str
    rulepack_hash: str
    #: Processing completion timestamp, same session monotonic domain.
    processed_mono_ns: int
    backend: str
    backend_status: str = "ok"
    detections: tuple[Detection, ...] = ()
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    # NOTE: overflow (> MAX_DETECTIONS_PER_BATCH) is deliberately NOT rejected
    # at construction: the batch must stay inspectable so validate_batch can
    # report DETECTION_OVERFLOW instead of silently truncating (plan §7.2).


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of :func:`validate_batch` (plan §7.1). Never mutates the
    input; deterministic reason ordering."""

    accepted: bool
    batch_identity: str
    reason_codes: tuple[str, ...] = ()
    #: Indices into ``batch.detections`` invalidated at detection level.
    rejected_indices: tuple[int, ...] = ()

    def has_reason(self, code: str) -> bool:
        return code in self.reason_codes


@dataclass(frozen=True)
class RoadContext:
    """Timestamped context for road ownership (plan §7.1). Carries NO truth
    labels — only availability, continuity identity and provenance."""

    session_id: str
    context_mono_ns: int
    calibration_available: bool = False
    ego_motion_available: bool = False
    road_continuity_id: str | None = None
    at_junction: bool = False
    country_ambiguous: bool = False
    provenance: str = "unknown"


@dataclass(frozen=True)
class ApplicabilityEvidence:
    verdict: ApplicabilityVerdict
    contributing_frame_ids: tuple[str, ...] = ()
    contributing_context_ids: tuple[str, ...] = ()
    method: str = ""
    method_version: str = ""
    #: Rejection/uncertainty reason; "" when the verdict is fully supported.
    reason: str = ""


@dataclass(frozen=True)
class PassageEvidence:
    verdict: PassageVerdict
    #: Same-session monotonic crossing bounds, when established.
    earliest_crossing_mono_ns: int | None = None
    latest_crossing_mono_ns: int | None = None
    method: str = ""
    method_version: str = ""
    reason: str = ""


@dataclass(frozen=True)
class SignTrack:
    """A stable track of one physical sign (plan §7.1)."""

    track_id: str
    sign_family: SignFamily
    observation_ids: tuple[str, ...]
    consensus_value_state: ValueState
    consensus_value_kph: int | None = None
    first_seen_mono_ns: int = 0
    last_seen_mono_ns: int = 0
    #: Unique capture timestamps backing the track (deduped).
    capture_timestamps_ns: tuple[int, ...] = ()
    applicability: ApplicabilityEvidence | None = None
    passage: PassageEvidence | None = None

    def __post_init__(self) -> None:
        if self.consensus_value_state == ValueState.VALUE and self.consensus_value_kph is None:
            raise NnslerContractError(
                ReasonCode.INVALID_VALUE_CONSISTENCY,
                "SignTrack consensus VALUE requires an int consensus_value_kph",
            )
        if self.consensus_value_state != ValueState.VALUE and self.consensus_value_kph is not None:
            raise NnslerContractError(
                ReasonCode.INVALID_VALUE_CONSISTENCY,
                "SignTrack consensus non-VALUE requires consensus_value_kph=None",
            )


@dataclass(frozen=True)
class PerceptionHealth:
    """Backend/health snapshot (plan §7.1)."""

    session_id: str
    backend: str
    backend_available: bool
    last_processed_capture_mono_ns: int | None = None
    last_successful_completion_mono_ns: int | None = None
    overflow: bool = False
    fault_code: str | None = None
    dropped_frames: int = 0


@dataclass(frozen=True)
class SourceSnapshot:
    """Point-in-time value of an operational source (plan §7.1).

    ``value_mps`` is the operational unit for car/map; conversion to km/h for
    display happens exactly once at the advisory boundary and never mutates
    this snapshot."""

    source: SourceKind
    provenance: str
    valid: bool
    value_mps: float | None = None
    #: Observation time in the session monotonic domain, when known.
    observed_mono_ns: int | None = None
    #: True when the source does not expose a usable observation time.
    age_unknown: bool = False
    road_context_id: str | None = None


@dataclass(frozen=True)
class LimitHypothesis:
    """The current Vision hypothesis (plan §7.1). Advisory only."""

    has_value: bool
    value_kph: int | None
    state: HypothesisState
    track_id: str | None = None
    event_id: str | None = None
    activation_start_mono_ns: int | None = None
    activation_end_mono_ns: int | None = None
    last_context_mono_ns: int | None = None
    unavailable_reason: str | None = None
    usable_for_advisory: bool = False

    def __post_init__(self) -> None:
        if self.has_value and self.value_kph is None:
            raise NnslerContractError(
                ReasonCode.INVALID_VALUE_CONSISTENCY,
                "LimitHypothesis has_value requires an int value_kph",
            )
        if not self.has_value and self.value_kph is not None:
            raise NnslerContractError(
                ReasonCode.INVALID_VALUE_CONSISTENCY,
                "LimitHypothesis value_kph requires has_value=True",
            )
        if self.has_value and not (MIN_VALUE_KPH <= self.value_kph <= MAX_VALUE_KPH):
            raise NnslerContractError(
                ReasonCode.INVALID_VALUE_DOMAIN,
                f"LimitHypothesis value_kph={self.value_kph} outside "
                f"[{MIN_VALUE_KPH}, {MAX_VALUE_KPH}]",
            )


@dataclass(frozen=True)
class AdvisoryComparison:
    """Car / Map / Vision comparison for advisory display (plan §7.1/§9).

    There is deliberately NO target-speed field anywhere in this type."""

    car: SourceSnapshot | None
    map: SourceSnapshot | None
    vision: LimitHypothesis | None
    car_age_ns: int | None = None
    map_age_ns: int | None = None
    vision_age_ns: int | None = None
    agreement: Agreement = Agreement.INSUFFICIENT_DATA
    display_decision: DisplayDecision = DisplayDecision.HIDE
    reason: str = ""


# ---------------------------------------------------------------------------
# Validation (deterministic API contract, plan §7.4)
# ---------------------------------------------------------------------------

def _is_finite_in_unit_interval(x: float) -> bool:
    return math.isfinite(x) and 0.0 <= x <= 1.0


def _validate_bbox(bbox: Sequence[float], frame: FrameRef) -> tuple[str, ...]:
    if len(bbox) != 4:
        return (ReasonCode.INVALID_BOX_DEGENERATE,)
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return (ReasonCode.INVALID_BOX_DEGENERATE,)
    if not (x1 < x2 and y1 < y2):
        return (ReasonCode.INVALID_BOX_DEGENERATE,)
    if frame.native_width <= 0 or frame.native_height <= 0:
        # Native dimensions are invalid at the batch level; geometry bounds
        # cannot be evaluated here — the batch code reports it.
        return ()
    if x1 < 0 or y1 < 0 or x2 > frame.native_width or y2 > frame.native_height:
        return (ReasonCode.INVALID_BOX_OUT_OF_BOUNDS,)
    return ()


def _validate_value_state(value_state: ValueState, value_kph: int | None) -> tuple[str, ...]:
    if value_state == ValueState.VALUE:
        if value_kph is None:
            return (ReasonCode.INVALID_VALUE_CONSISTENCY,)
        if not isinstance(value_kph, int) or isinstance(value_kph, bool):
            return (ReasonCode.INVALID_VALUE_CONSISTENCY,)
        if value_kph == 0:
            return (ReasonCode.INVALID_VALUE_ZERO,)
        if not (MIN_VALUE_KPH <= value_kph <= MAX_VALUE_KPH):
            return (ReasonCode.INVALID_VALUE_DOMAIN,)
        return ()
    if value_kph is not None:
        return (ReasonCode.INVALID_VALUE_CONSISTENCY,)
    return ()


def _validate_frame_ref(det_frame: FrameRef, frame: FrameRef) -> tuple[str, ...]:
    """A detection must refer to the *exact* frame of its batch: identity and
    every timing/native field must match (plan §7.1). A mismatched field
    (capture time, reference edge, native size, preprocessing) is a
    clock-domain/consistency violation, not a silent merge."""
    if det_frame.identity() != frame.identity():
        return (ReasonCode.INVALID_CLOCK_DOMAIN,)
    for name in (
        "capture_mono_ns",
        "capture_reference",
        "native_width",
        "native_height",
        "preprocessing_identity",
    ):
        if getattr(det_frame, name) != getattr(frame, name):
            return (ReasonCode.INVALID_CLOCK_DOMAIN,)
    return ()


def _validate_detection(det: Detection, frame: FrameRef) -> tuple[str, ...]:
    codes: list[str] = []
    codes.extend(_validate_frame_ref(det.frame, frame))
    if not _is_finite_in_unit_interval(det.detection_score):
        if not math.isfinite(det.detection_score):
            codes.append(ReasonCode.INVALID_NONFINITE_SCORE)
        else:
            codes.append(ReasonCode.INVALID_SCORE_RANGE)
    if not _is_finite_in_unit_interval(det.classification_score):
        if not math.isfinite(det.classification_score):
            codes.append(ReasonCode.INVALID_NONFINITE_SCORE)
        else:
            codes.append(ReasonCode.INVALID_SCORE_RANGE)
    codes.extend(_validate_bbox(det.bbox_xyxy, frame))
    # [nnslr-t1] - START  (linked panels: same geometric rules as the bbox)
    for panel in det.linked_panel_boxes:
        codes.extend(_validate_bbox(panel, frame))
    # [nnslr-t1] - END
    codes.extend(_validate_value_state(det.value_state, det.value_kph))
    return tuple(dict.fromkeys(codes))


def validate_batch(
    batch: ObservationBatch,
    now_mono_ns: int,
    expected_session_id: str,
    max_evidence_age_ns: int = DEFAULT_MAX_EVIDENCE_AGE_NS,
    recent_frame_ids: frozenset[FrameIdentity] | None = None,
) -> ValidationResult:
    """Validate one :class:`ObservationBatch` (plan §7.4).

    Pure and deterministic: injects time via ``now_mono_ns``; never calls the
    live clock; never mutates the input.

    ``recent_frame_ids`` (optional) lets callers (e.g. the T6 tracker) reject
    duplicate frames: a batch whose frame identity is already present is
    rejected with ``DUPLICATE_FRAME``. The three-argument form from the
    §7.4 contract is preserved by the default.
    """
    batch_codes: list[str] = []

    if batch.schema_version != SUPPORTED_SCHEMA_VERSION:
        batch_codes.append(ReasonCode.INVALID_SCHEMA_VERSION)

    frame = batch.frame
    if frame.session_id != expected_session_id:
        batch_codes.append(ReasonCode.INVALID_SESSION_MISMATCH)

    if frame.native_width <= 0 or frame.native_height <= 0:
        batch_codes.append(ReasonCode.INVALID_NATIVE_DIMENSIONS)

    if frame.capture_mono_ns < 0:
        batch_codes.append(ReasonCode.INVALID_CAPTURE_TIMESTAMP_NEGATIVE)
    elif frame.session_id == expected_session_id and frame.capture_mono_ns > now_mono_ns:
        batch_codes.append(ReasonCode.INVALID_CAPTURE_TIMESTAMP_FUTURE)

    if batch.processed_mono_ns < frame.capture_mono_ns:
        batch_codes.append(ReasonCode.INVALID_NEGATIVE_LATENCY)

    # Staleness: only evaluated when the capture timestamp is in-domain and sane.
    if (
        not batch_codes
        and frame.session_id == expected_session_id
        and 0 <= frame.capture_mono_ns <= now_mono_ns
        and (now_mono_ns - frame.capture_mono_ns) > max_evidence_age_ns
    ):
        batch_codes.append(ReasonCode.STALE_EVIDENCE)

    if batch_codes:
        return ValidationResult(
            accepted=False,
            batch_identity=frame.key(),
            reason_codes=tuple(dict.fromkeys(batch_codes)),
            rejected_indices=tuple(range(len(batch.detections))),
        )

    if recent_frame_ids is not None and frame.identity() in recent_frame_ids:
        return ValidationResult(
            accepted=False,
            batch_identity=frame.key(),
            reason_codes=(ReasonCode.DUPLICATE_FRAME,),
            rejected_indices=tuple(range(len(batch.detections))),
        )

    if len(batch.detections) > MAX_DETECTIONS_PER_BATCH:
        return ValidationResult(
            accepted=False,
            batch_identity=frame.key(),
            reason_codes=(ReasonCode.DETECTION_OVERFLOW,),
            rejected_indices=tuple(range(len(batch.detections))),
        )

    rejected: dict[int, tuple[str, ...]] = {}
    for idx, det in enumerate(batch.detections):
        codes = _validate_detection(det, frame)
        if codes:
            rejected[idx] = codes

    all_codes = list(rejected.values())
    if len(rejected) == len(batch.detections) and batch.detections:
        # Every detection dead → batch not accepted.
        return ValidationResult(
            accepted=False,
            batch_identity=frame.key(),
            reason_codes=tuple(dict.fromkeys(c for codes in all_codes for c in codes)),
            rejected_indices=tuple(sorted(rejected)),
        )
    if not batch.detections:
        # An empty batch is structurally valid (no sign seen) but carries no
        # evidence; accept it so "unavailable" stays representable.
        return ValidationResult(
            accepted=True,
            batch_identity=frame.key(),
            reason_codes=(),
            rejected_indices=(),
        )
    return ValidationResult(
        accepted=True,
        batch_identity=frame.key(),
        reason_codes=tuple(dict.fromkeys(c for codes in all_codes for c in codes)),
        rejected_indices=tuple(sorted(rejected)),
    )


# ---------------------------------------------------------------------------
# Serialization (pure JSON-safe round-trip; plan §7.2 presence rules)
# ---------------------------------------------------------------------------

def _enum_value(v: Enum) -> str:
    return v.value


# [nnslr-t1] - START  (strict numeric deserialization: no silent coercion)
def _strict_int(value: Any, field: str) -> int:
    """Deserialize an integer contract field with NO silent normalization.

    Only a real ``int`` is accepted. ``bool`` is a subclass of ``int`` but is
    rejected explicitly; integral floats (``50.0``), non-integral floats
    (``50.9``), strings (``"50"``) and every other type are rejected with a
    deterministic :class:`NnslerContractError` carrying
    :attr:`ReasonCode.INVALID_NUMERIC_TYPE` (plan §7.2: external payloads are
    validated, never coerced).
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise NnslerContractError(
            ReasonCode.INVALID_NUMERIC_TYPE,
            f"{field} must be an integer, got {type(value).__name__}: {value!r}",
        )
    return value


def _strict_float(value: Any, field: str) -> float:
    """Deserialize a float contract field with NO silent normalization.

    Accepts ``int`` (safe widening) and ``float`` (including NaN/inf, which
    downstream validation must then reject as non-finite). Rejects ``bool``,
    strings and every other type deterministically.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NnslerContractError(
            ReasonCode.INVALID_NUMERIC_TYPE,
            f"{field} must be a number, got {type(value).__name__}: {value!r}",
        )
    return float(value)
# [nnslr-t1] - END


def frame_ref_to_dict(ref: FrameRef) -> dict[str, Any]:
    return {
        "session_id": ref.session_id,
        "stream": _enum_value(ref.stream),
        "frame_id": ref.frame_id,
        "capture_mono_ns": ref.capture_mono_ns,
        "capture_reference": _enum_value(ref.capture_reference),
        "native_width": ref.native_width,
        "native_height": ref.native_height,
        "preprocessing_identity": ref.preprocessing_identity,
    }


def frame_ref_from_dict(data: Mapping[str, Any]) -> FrameRef:
    return FrameRef(
        session_id=str(data["session_id"]),
        stream=StreamId(str(data["stream"])),
        frame_id=int(data["frame_id"]),
        capture_mono_ns=int(data["capture_mono_ns"]),
        capture_reference=CaptureReference(str(data.get("capture_reference", "unknown"))),
        native_width=int(data.get("native_width", 0)),
        native_height=int(data.get("native_height", 0)),
        preprocessing_identity=str(data.get("preprocessing_identity", "none")),
    )


def detection_to_dict(det: Detection) -> dict[str, Any]:
    return {
        "frame": frame_ref_to_dict(det.frame),
        "bbox_xyxy": [float(v) for v in det.bbox_xyxy],
        "sign_family": _enum_value(det.sign_family),
        "value_state": _enum_value(det.value_state),
        "value_kph": det.value_kph,  # None stays None — absence is preserved
        "linked_panel_boxes": [[float(v) for v in b] for b in det.linked_panel_boxes],
        "detection_score": float(det.detection_score),
        "classification_score": float(det.classification_score),
        "supported_domain": bool(det.supported_domain),
        "observation_id": det.observation_id,
    }


def detection_from_dict(data: Mapping[str, Any]) -> Detection:
    value_kph = data.get("value_kph", None)
    value_state = ValueState(str(data["value_state"]))
    if value_state != ValueState.VALUE and value_kph is not None:
        raise NnslerContractError(
            ReasonCode.INVALID_VALUE_CONSISTENCY,
            "payload carries value_kph without has_value; zero/other numbers "
            "are never reinterpreted as an accepted limit",
        )
    return Detection(
        frame=frame_ref_from_dict(data["frame"]),
        bbox_xyxy=tuple(float(v) for v in data["bbox_xyxy"]),
        sign_family=SignFamily(str(data["sign_family"])),
        value_state=value_state,
        value_kph=None if value_state != ValueState.VALUE else int(value_kph),
        linked_panel_boxes=tuple(
            tuple(float(v) for v in b) for b in data.get("linked_panel_boxes", ())
        ),
        detection_score=float(data.get("detection_score", 0.0)),
        classification_score=float(data.get("classification_score", 0.0)),
        supported_domain=bool(data.get("supported_domain", True)),
        observation_id=str(data.get("observation_id", "")),
    )


def batch_to_dict(batch: ObservationBatch) -> dict[str, Any]:
    return {
        "schema_version": batch.schema_version,
        "frame": frame_ref_to_dict(batch.frame),
        "model_hash": batch.model_hash,
        "config_hash": batch.config_hash,
        "rulepack_hash": batch.rulepack_hash,
        "processed_mono_ns": batch.processed_mono_ns,
        "backend": batch.backend,
        "backend_status": batch.backend_status,
        "detections": [detection_to_dict(d) for d in batch.detections],
    }


def batch_from_dict(data: Mapping[str, Any]) -> ObservationBatch:
    return ObservationBatch(
        schema_version=int(data.get("schema_version", SUPPORTED_SCHEMA_VERSION)),
        frame=frame_ref_from_dict(data["frame"]),
        model_hash=str(data["model_hash"]),
        config_hash=str(data["config_hash"]),
        rulepack_hash=str(data.get("rulepack_hash", "")),
        processed_mono_ns=int(data["processed_mono_ns"]),
        backend=str(data["backend"]),
        backend_status=str(data.get("backend_status", "ok")),
        detections=tuple(detection_from_dict(d) for d in data.get("detections", ())),
    )


def validation_result_to_dict(result: ValidationResult) -> dict[str, Any]:
    return {
        "accepted": result.accepted,
        "batch_identity": result.batch_identity,
        "reason_codes": list(result.reason_codes),
        "rejected_indices": list(result.rejected_indices),
    }


def validation_result_from_dict(data: Mapping[str, Any]) -> ValidationResult:
    return ValidationResult(
        accepted=bool(data["accepted"]),
        batch_identity=str(data["batch_identity"]),
        reason_codes=tuple(str(c) for c in data.get("reason_codes", ())),
        rejected_indices=tuple(int(i) for i in data.get("rejected_indices", ())),
    )


def limit_hypothesis_to_dict(h: LimitHypothesis) -> dict[str, Any]:
    return {
        "has_value": h.has_value,
        "value_kph": h.value_kph,
        "state": _enum_value(h.state),
        "track_id": h.track_id,
        "event_id": h.event_id,
        "activation_start_mono_ns": h.activation_start_mono_ns,
        "activation_end_mono_ns": h.activation_end_mono_ns,
        "last_context_mono_ns": h.last_context_mono_ns,
        "unavailable_reason": h.unavailable_reason,
        "usable_for_advisory": h.usable_for_advisory,
    }


def limit_hypothesis_from_dict(data: Mapping[str, Any]) -> LimitHypothesis:
    value_kph = data.get("value_kph", None)
    has_value = bool(data["has_value"])
    if not has_value and value_kph is not None:
        raise NnslerContractError(
            ReasonCode.INVALID_VALUE_CONSISTENCY,
            "hypothesis payload carries value_kph without has_value",
        )
    return LimitHypothesis(
        has_value=has_value,
        value_kph=None if not has_value else int(value_kph),
        state=HypothesisState(str(data["state"])),
        track_id=data.get("track_id"),
        event_id=data.get("event_id"),
        activation_start_mono_ns=data.get("activation_start_mono_ns"),
        activation_end_mono_ns=data.get("activation_end_mono_ns"),
        last_context_mono_ns=data.get("last_context_mono_ns"),
        unavailable_reason=data.get("unavailable_reason"),
        usable_for_advisory=bool(data.get("usable_for_advisory", False)),
    )


def source_snapshot_to_dict(s: SourceSnapshot) -> dict[str, Any]:
    return {
        "source": _enum_value(s.source),
        "provenance": s.provenance,
        "valid": s.valid,
        "value_mps": s.value_mps,
        "observed_mono_ns": s.observed_mono_ns,
        "age_unknown": s.age_unknown,
        "road_context_id": s.road_context_id,
    }


def source_snapshot_from_dict(data: Mapping[str, Any]) -> SourceSnapshot:
    return SourceSnapshot(
        source=SourceKind(str(data["source"])),
        provenance=str(data["provenance"]),
        valid=bool(data["valid"]),
        value_mps=float(data["value_mps"]) if data.get("value_mps") is not None else None,
        observed_mono_ns=data.get("observed_mono_ns"),
        age_unknown=bool(data.get("age_unknown", False)),
        road_context_id=data.get("road_context_id"),
    )


__all__ = [
    "SUPPORTED_SCHEMA_VERSION",
    "MAX_DETECTIONS_PER_BATCH",
    "DEFAULT_MAX_EVIDENCE_AGE_NS",
    "MIN_VALUE_KPH",
    "MAX_VALUE_KPH",
    "StreamId",
    "CaptureReference",
    "SignFamily",
    "ValueState",
    "SourceKind",
    "HypothesisState",
    "ApplicabilityVerdict",
    "PassageVerdict",
    "Agreement",
    "DisplayDecision",
    "ReasonCode",
    "REASON_CODE_REGISTRY",
    "NnslerContractError",
    "FrameIdentity",
    "WallClockMapping",
    "to_session_mono_ns",
    "same_clock_domain",
    "FrameRef",
    "Detection",
    "ObservationBatch",
    "ValidationResult",
    "RoadContext",
    "ApplicabilityEvidence",
    "PassageEvidence",
    "SignTrack",
    "PerceptionHealth",
    "SourceSnapshot",
    "LimitHypothesis",
    "AdvisoryComparison",
    "validate_batch",
    "frame_ref_to_dict",
    "frame_ref_from_dict",
    "detection_to_dict",
    "detection_from_dict",
    "batch_to_dict",
    "batch_from_dict",
    "validation_result_to_dict",
    "validation_result_from_dict",
    "limit_hypothesis_to_dict",
    "limit_hypothesis_from_dict",
    "source_snapshot_to_dict",
    "source_snapshot_from_dict",
]
# [nnslr-t1] - END

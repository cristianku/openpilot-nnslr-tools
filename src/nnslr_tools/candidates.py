# [nnslr-t2] - START
"""Candidate discovery for human review.

Map-derived values are search hints only. They are never copied into annotation
ground truth. Candidate timestamps are projected to video time only through an
already-established frame/log alignment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from nnslr_tools.alignment import CommaAlignedFrame
from nnslr_tools.comma_log import MapSpeedEvent


@dataclass(frozen=True)
class CandidateEvent:
    candidate_id: str
    log_mono_time: int
    candidate_reason: str
    source_service: str
    field: str
    old_value: float | None
    new_value: float | None
    video_time_s: float | None = None
    projection_error_ns: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "log_mono_time": self.log_mono_time,
            "candidate_reason": self.candidate_reason,
            "source_service": self.source_service,
            "field": self.field,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "video_time_s": self.video_time_s,
            "projection_error_ns": self.projection_error_ns,
            "ground_truth": False,
        }


def _different(a: float | None, b: float | None) -> bool:
    if a is None and b is None:
        return False
    if a is None or b is None:
        return True
    return abs(a - b) > 1e-9


def find_speed_candidates(events: Sequence[MapSpeedEvent]) -> list[CandidateEvent]:
    """Return transitions in map speed fields as review candidates."""
    ordered = sorted(events, key=lambda e: e.log_mono_time)
    out: list[CandidateEvent] = []
    previous: MapSpeedEvent | None = None
    seq = 0
    for event in ordered:
        if previous is None:
            previous = event
            continue
        for field in ("speed_limit", "speed_limit_ahead"):
            old = getattr(previous, field)
            new = getattr(event, field)
            if not _different(old, new):
                continue
            seq += 1
            out.append(
                CandidateEvent(
                    candidate_id=f"map-{event.log_mono_time}-{seq}",
                    log_mono_time=event.log_mono_time,
                    candidate_reason=f"{event.service}.{field}_transition",
                    source_service=event.service,
                    field=field,
                    old_value=old,
                    new_value=new,
                )
            )
        previous = event
    return out


def project_candidates_to_video(
    candidates: Sequence[CandidateEvent],
    aligned_frames: Sequence[CommaAlignedFrame],
    *,
    max_error_ns: int = 250_000_000,
) -> list[CandidateEvent]:
    """Attach video time using the nearest exact aligned capture timestamp.

    Projection only establishes where to inspect video. It does not establish a
    sign, value or legal applicability.
    """
    anchors = [
        f for f in aligned_frames
        if f.alignment_status == "exact"
        and f.capture_mono_ns is not None
        and (f.video_best_effort_time_s is not None or f.video_pts_time_s is not None)
    ]
    anchors.sort(key=lambda f: int(f.capture_mono_ns))
    out: list[CandidateEvent] = []
    for candidate in candidates:
        if not anchors:
            out.append(candidate)
            continue
        nearest = min(anchors, key=lambda f: abs(int(f.capture_mono_ns) - candidate.log_mono_time))
        error = abs(int(nearest.capture_mono_ns) - candidate.log_mono_time)
        video_time = (
            nearest.video_best_effort_time_s
            if nearest.video_best_effort_time_s is not None
            else nearest.video_pts_time_s
        )
        if error > max_error_ns:
            out.append(candidate)
            continue
        out.append(
            CandidateEvent(
                candidate_id=candidate.candidate_id,
                log_mono_time=candidate.log_mono_time,
                candidate_reason=candidate.candidate_reason,
                source_service=candidate.source_service,
                field=candidate.field,
                old_value=candidate.old_value,
                new_value=candidate.new_value,
                video_time_s=video_time,
                projection_error_ns=error,
            )
        )
    return out
# [nnslr-t2] - END

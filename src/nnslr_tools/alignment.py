# [nnslr-t2] - START
"""Real local comma video/log alignment.

Unlike the synthetic align_frames() helper in manifest.py, this adapter does
not assume an arbitrary encoded_frame_id equals the decoded index. It uses the
openpilot EncodeIndex contract directly:

  segmentId       = index into the camera file in presentation order
  segmentIdEncode = index into the camera file in encode order

Therefore decoded presentation index is joined to EncodeIndex.segmentId.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from nnslr_tools.comma_log import EncodeIndexEvent
from nnslr_tools.media import VideoFrameTiming


@dataclass(frozen=True)
class CommaAlignedFrame:
    decoded_frame_index: int
    video_pts_time_s: float | None
    video_best_effort_time_s: float | None
    frame_id: int | None
    segment_num: int | None
    segment_id: int | None
    segment_id_encode: int | None
    timestamp_sof: int | None
    timestamp_eof: int | None
    capture_mono_ns: int | None
    capture_reference: str
    alignment_status: str
    alignment_reason: str | None
    # [media-clock] - START
    video_media_time_s: float | None = None
    video_media_time_source: str = "unknown"
    # [media-clock] - END

    def to_dict(self) -> dict[str, Any]:
        return {
            "decoded_frame_index": self.decoded_frame_index,
            "video_pts_time_s": self.video_pts_time_s,
            "video_best_effort_time_s": self.video_best_effort_time_s,
            "frame_id": self.frame_id,
            "segment_num": self.segment_num,
            "segment_id": self.segment_id,
            "segment_id_encode": self.segment_id_encode,
            "timestamp_sof": self.timestamp_sof,
            "timestamp_eof": self.timestamp_eof,
            "capture_mono_ns": self.capture_mono_ns,
            "capture_reference": self.capture_reference,
            "alignment_status": self.alignment_status,
            "alignment_reason": self.alignment_reason,
            "video_media_time_s": self.video_media_time_s,
            "video_media_time_source": self.video_media_time_source,
        }


def comma_aligned_frame_from_dict(data: dict[str, Any]) -> CommaAlignedFrame:
    """Strict-enough loader for alignment JSONL produced by this package."""
    def opt_int(name: str) -> int | None:
        value = data.get(name)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be int|null")
        return value

    def opt_float(name: str) -> float | None:
        value = data.get(name)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be number|null")
        return float(value)

    idx = data.get("decoded_frame_index")
    if isinstance(idx, bool) or not isinstance(idx, int):
        raise ValueError("decoded_frame_index must be int")
    capture_reference = data.get("capture_reference")
    alignment_status = data.get("alignment_status")
    if not isinstance(capture_reference, str) or not isinstance(alignment_status, str):
        raise ValueError("capture_reference/alignment_status must be strings")
    reason = data.get("alignment_reason")
    if reason is not None and not isinstance(reason, str):
        raise ValueError("alignment_reason must be string|null")

    return CommaAlignedFrame(
        decoded_frame_index=idx,
        video_pts_time_s=opt_float("video_pts_time_s"),
        video_best_effort_time_s=opt_float("video_best_effort_time_s"),
        frame_id=opt_int("frame_id"),
        segment_num=opt_int("segment_num"),
        segment_id=opt_int("segment_id"),
        segment_id_encode=opt_int("segment_id_encode"),
        timestamp_sof=opt_int("timestamp_sof"),
        timestamp_eof=opt_int("timestamp_eof"),
        capture_mono_ns=opt_int("capture_mono_ns"),
        capture_reference=capture_reference,
        alignment_status=alignment_status,
        alignment_reason=reason,
        video_media_time_s=opt_float("video_media_time_s"),
        video_media_time_source=data.get("video_media_time_source", "unknown"),
    )


@dataclass(frozen=True)
class CommaAlignmentReport:
    frame_count: int
    metadata_count: int
    matched: int
    unresolved: int
    duplicate_segment_ids: tuple[int, ...]
    unmatched_video_indices: tuple[int, ...]
    unmatched_metadata_segment_ids: tuple[int, ...]
    frame_id_duplicates: tuple[int, ...]
    presentation_discontinuities: tuple[dict[str, int], ...]
    capture_discontinuities: tuple[dict[str, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "metadata_count": self.metadata_count,
            "matched": self.matched,
            "unresolved": self.unresolved,
            "duplicate_segment_ids": list(self.duplicate_segment_ids),
            "unmatched_video_indices": list(self.unmatched_video_indices),
            "unmatched_metadata_segment_ids": list(self.unmatched_metadata_segment_ids),
            "frame_id_duplicates": list(self.frame_id_duplicates),
            "presentation_discontinuities": list(self.presentation_discontinuities),
            "capture_discontinuities": list(self.capture_discontinuities),
        }


def _capture_from_event(event: EncodeIndexEvent) -> tuple[int | None, str]:
    if event.timestamp_sof > 0:
        return event.timestamp_sof, "sof"
    if event.timestamp_eof > 0:
        return event.timestamp_eof, "eof"
    return None, "unknown"


def align_comma_segment(
    video_frames: Sequence[VideoFrameTiming],
    encode_index: Sequence[EncodeIndexEvent],
    *,
    segment_num: int,
) -> tuple[list[CommaAlignedFrame], CommaAlignmentReport]:
    """Join ffprobe presentation-order frames to openpilot EncodeIndex.

    The only production join key here is EncodeIndex.segmentId, whose schema
    comment defines it as presentation order within the segment.
    """
    metadata = [e for e in encode_index if e.segment_num == segment_num]
    by_segment_id: dict[int, list[EncodeIndexEvent]] = {}
    by_frame_id: dict[int, int] = {}
    for event in metadata:
        by_segment_id.setdefault(event.segment_id, []).append(event)
        by_frame_id[event.frame_id] = by_frame_id.get(event.frame_id, 0) + 1

    duplicate_segment_ids = tuple(sorted(k for k, v in by_segment_id.items() if len(v) > 1))
    frame_id_duplicates = tuple(sorted(k for k, count in by_frame_id.items() if count > 1))

    video_indices = {f.decoded_frame_index for f in video_frames}
    # [t2-validation] - START
    # Logger startup can discard frames up to an I-frame without resetting
    # segmentId. Missing/truncated inputs are also ambiguous: overlapping ids
    # alone cannot prove their correspondence to decoded positions.
    domain_matches = video_indices == set(by_segment_id) and video_indices == set(range(len(video_frames)))
    # [t2-validation] - END
    unmatched_metadata = tuple(sorted(k for k in by_segment_id if k not in video_indices))
    unmatched_video: list[int] = []
    aligned: list[CommaAlignedFrame] = []
    matched = 0

    for frame in sorted(video_frames, key=lambda f: f.decoded_frame_index):
        candidates = by_segment_id.get(frame.decoded_frame_index, [])
        if not domain_matches or len(candidates) != 1:
            reason = ("presentation_index_domain_mismatch" if not domain_matches
                      else "missing_encode_index" if not candidates else "duplicate_encode_index")
            unmatched_video.append(frame.decoded_frame_index)
            aligned.append(
                CommaAlignedFrame(
                    decoded_frame_index=frame.decoded_frame_index,
                    video_pts_time_s=frame.pts_time_s,
                    video_best_effort_time_s=frame.best_effort_timestamp_s,
                    video_media_time_s=frame.media_time_s,
                    video_media_time_source=frame.media_time_source,
                    frame_id=None,
                    segment_num=None,
                    segment_id=None,
                    segment_id_encode=None,
                    timestamp_sof=None,
                    timestamp_eof=None,
                    capture_mono_ns=None,
                    capture_reference="unknown",
                    alignment_status="unresolved",
                    alignment_reason=reason,
                )
            )
            continue

        event = candidates[0]
        capture, reference = _capture_from_event(event)
        status = "exact" if capture is not None else "unresolved"
        reason = None if capture is not None else "capture_timestamp_missing"
        if capture is not None:
            matched += 1
        aligned.append(
            CommaAlignedFrame(
                decoded_frame_index=frame.decoded_frame_index,
                video_pts_time_s=frame.pts_time_s,
                video_best_effort_time_s=frame.best_effort_timestamp_s,
                video_media_time_s=frame.media_time_s,
                video_media_time_source=frame.media_time_source,
                frame_id=event.frame_id,
                segment_num=event.segment_num,
                segment_id=event.segment_id,
                segment_id_encode=event.segment_id_encode,
                timestamp_sof=event.timestamp_sof or None,
                timestamp_eof=event.timestamp_eof or None,
                capture_mono_ns=capture,
                capture_reference=reference,
                alignment_status=status,
                alignment_reason=reason,
            )
        )

    presentation_discontinuities: list[dict[str, int]] = []
    previous_segment_id: int | None = None
    for event in sorted(metadata, key=lambda e: e.log_mono_time):
        if previous_segment_id is not None and event.segment_id < previous_segment_id:
            presentation_discontinuities.append(
                {"previous_segment_id": previous_segment_id, "current_segment_id": event.segment_id}
            )
        previous_segment_id = event.segment_id

    capture_discontinuities: list[dict[str, int]] = []
    previous_capture: int | None = None
    previous_index: int | None = None
    for item in aligned:
        if item.capture_mono_ns is None:
            previous_capture = None
            previous_index = None
            continue
        if previous_capture is not None and item.capture_mono_ns < previous_capture:
            capture_discontinuities.append(
                {
                    "previous_frame_index": int(previous_index),
                    "current_frame_index": item.decoded_frame_index,
                    "previous_capture_mono_ns": previous_capture,
                    "current_capture_mono_ns": item.capture_mono_ns,
                }
            )
        previous_capture = item.capture_mono_ns
        previous_index = item.decoded_frame_index

    report = CommaAlignmentReport(
        frame_count=len(video_frames),
        metadata_count=len(metadata),
        matched=matched,
        unresolved=sum(1 for x in aligned if x.alignment_status == "unresolved"),
        duplicate_segment_ids=duplicate_segment_ids,
        unmatched_video_indices=tuple(unmatched_video),
        unmatched_metadata_segment_ids=unmatched_metadata,
        frame_id_duplicates=frame_id_duplicates,
        presentation_discontinuities=tuple(presentation_discontinuities),
        capture_discontinuities=tuple(capture_discontinuities),
    )
    return aligned, report
# [nnslr-t2] - END

from __future__ import annotations

from nnslr_tools.alignment import align_comma_segment
from nnslr_tools.comma_log import EncodeIndexEvent
from nnslr_tools.media import VideoFrameTiming


def _frame(idx: int) -> VideoFrameTiming:
    return VideoFrameTiming(
        decoded_frame_index=idx,
        pts_time_s=idx / 20.0,
        dts_time_s=idx / 20.0,
        best_effort_timestamp_s=idx / 20.0,
        key_frame=idx == 0,
        pict_type="I" if idx == 0 else "P",
    )


def _enc(segment_id: int, frame_id: int, *, segment_num: int = 4, sof: int | None = None) -> EncodeIndexEvent:
    return EncodeIndexEvent(
        service="narrowRoadEncodeIdx",
        log_mono_time=1_000_000_000 + segment_id * 50_000_000,
        frame_id=frame_id,
        segment_num=segment_num,
        segment_id=segment_id,
        segment_id_encode=segment_id,
        timestamp_sof=sof if sof is not None else 2_000_000_000 + segment_id * 50_000_000,
        timestamp_eof=2_010_000_000 + segment_id * 50_000_000,
        encode_type="fullHEVC",
    )


def test_real_alignment_uses_segment_id_presentation_order() -> None:
    frames = [_frame(0), _frame(1), _frame(2)]
    metadata = [_enc(2, 102), _enc(0, 100), _enc(1, 101)]
    aligned, report = align_comma_segment(frames, metadata, segment_num=4)
    assert [x.frame_id for x in aligned] == [100, 101, 102]
    assert report.matched == 3
    assert report.unresolved == 0


def test_duplicate_segment_id_is_unresolved() -> None:
    frames = [_frame(0), _frame(1)]
    metadata = [_enc(0, 100), _enc(1, 101), _enc(1, 999)]
    aligned, report = align_comma_segment(frames, metadata, segment_num=4)
    assert aligned[1].alignment_status == "unresolved"
    assert aligned[1].alignment_reason == "duplicate_encode_index"
    assert report.duplicate_segment_ids == (1,)


def test_missing_capture_timestamp_stays_unresolved() -> None:
    frames = [_frame(0)]
    event = EncodeIndexEvent(
        service="narrowRoadEncodeIdx",
        log_mono_time=1_000_000_000,
        frame_id=100,
        segment_num=4,
        segment_id=0,
        segment_id_encode=0,
        timestamp_sof=0,
        timestamp_eof=0,
        encode_type="fullHEVC",
    )
    aligned, report = align_comma_segment(frames, [event], segment_num=4)
    assert aligned[0].capture_mono_ns is None
    assert aligned[0].alignment_status == "unresolved"
    assert report.matched == 0

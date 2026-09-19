from __future__ import annotations

from nnslr_tools.alignment import CommaAlignedFrame
from nnslr_tools.candidates import find_speed_candidates, project_candidates_to_video
from nnslr_tools.comma_log import MapSpeedEvent


def test_map_transition_is_candidate_not_ground_truth() -> None:
    events = [
        MapSpeedEvent("liveMapDataSP", 1_000, 50.0, None),
        MapSpeedEvent("liveMapDataSP", 2_000, 80.0, None),
    ]
    candidates = find_speed_candidates(events)
    assert len(candidates) == 1
    payload = candidates[0].to_dict()
    assert payload["candidate_reason"] == "liveMapDataSP.speed_limit_transition"
    assert payload["old_value"] == 50.0
    assert payload["new_value"] == 80.0
    assert payload["ground_truth"] is False


def test_candidate_projects_through_exact_alignment() -> None:
    candidate = find_speed_candidates([
        MapSpeedEvent("liveMapDataSP", 1_000_000_000, 50.0, None),
        MapSpeedEvent("liveMapDataSP", 2_000_000_000, 80.0, None),
    ])[0]
    frames = [
        CommaAlignedFrame(
            decoded_frame_index=10,
            video_pts_time_s=3.0,
            video_best_effort_time_s=3.0,
            frame_id=100,
            segment_num=1,
            segment_id=10,
            segment_id_encode=10,
            timestamp_sof=2_010_000_000,
            timestamp_eof=2_020_000_000,
            capture_mono_ns=2_010_000_000,
            capture_reference="sof",
            alignment_status="exact",
            alignment_reason=None,
        )
    ]
    projected = project_candidates_to_video([candidate], frames, max_error_ns=20_000_000)
    assert projected[0].video_time_s == 3.0
    assert projected[0].projection_error_ns == 10_000_000

# [media-clock] - START
from dataclasses import replace
from nnslr_tools.alignment import align_comma_segment, comma_aligned_frame_from_dict
from nnslr_tools.comma_log import EncodeIndexEvent
from nnslr_tools.media import VideoFrameTiming


def test_candidate_projects_generated_media_clock_but_not_unresolved_capture():
    frame = VideoFrameTiming(0, None, None, None, True, "I", .5, "raw_hevc_frame_duration")
    event = EncodeIndexEvent("narrowRoadEncodeIdx", 2_000, 42, 0, 0, 0, 2_000, 2_100)
    aligned, _ = align_comma_segment([frame], [event], segment_num=0)
    restored = comma_aligned_frame_from_dict(aligned[0].to_dict())
    assert restored.video_media_time_source == "raw_hevc_frame_duration"
    candidate = find_speed_candidates([MapSpeedEvent("liveMapDataSP", 1_000, 50, None), MapSpeedEvent("liveMapDataSP", 2_000, 80, None)])[0]
    assert project_candidates_to_video([candidate], [restored])[0].video_time_s == .5
    assert project_candidates_to_video([candidate], [replace(restored, alignment_status="unresolved")])[0].video_time_s is None
# [media-clock] - END

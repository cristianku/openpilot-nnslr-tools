from __future__ import annotations

import json
from pathlib import Path

from nnslr_tools.media import extract_frames, infer_camera_stream, probe_video


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def test_infer_camera_stream() -> None:
    assert infer_camera_stream(Path("fcamera.hevc")) == "narrow_road"
    assert infer_camera_stream(Path("ecamera.hevc")) == "wide_road"
    assert infer_camera_stream(Path("qcamera.ts")) == "q_narrow_road"
    assert infer_camera_stream(Path("other.mp4")) is None


def test_probe_video_parses_ffprobe_json(tmp_path) -> None:
    video = tmp_path / "fcamera.hevc"
    video.write_bytes(b"x")
    ffprobe = tmp_path / "ffprobe"
    payload = {
        "streams": [{
            "codec_type": "video",
            "codec_name": "hevc",
            "codec_long_name": "HEVC",
            "width": 1928,
            "height": 1208,
            "pix_fmt": "yuv420p",
            "r_frame_rate": "20/1",
            "avg_frame_rate": "20/1",
            "time_base": "1/1200000",
            "nb_read_frames": "1200",
        }],
        "format": {"format_name": "hevc", "duration": "60.0", "start_time": "0.0"},
    }
    _write_executable(
        ffprobe,
        "#!/bin/sh\n"
        "if [ \"$1\" = \"-version\" ]; then echo 'ffprobe fake 1.0'; exit 0; fi\n"
        f"cat <<'EOF'\n{json.dumps(payload)}\nEOF\n",
    )
    result = probe_video(video, ffprobe=str(ffprobe))
    assert result.camera_stream == "narrow_road"
    assert result.codec == "hevc"
    assert result.width == 1928
    assert result.height == 1208
    assert result.average_fps == 20.0
    assert result.duration_s == 60.0
    assert result.frame_count == 1200


def test_extract_frames_uses_deterministic_names(tmp_path) -> None:
    video = tmp_path / "fcamera.hevc"
    video.write_bytes(b"x")
    output = tmp_path / "frames"
    ffmpeg = tmp_path / "ffmpeg"
    _write_executable(
        ffmpeg,
        "#!/bin/sh\n"
        "last=''\n"
        "for arg in \"$@\"; do last=\"$arg\"; done\n"
        "out=$(printf '%s' \"$last\" | sed 's/%08d/00000000/')\n"
        "mkdir -p \"$(dirname \"$out\")\"\n"
        "printf x > \"$out\"\n",
    )
    rows = extract_frames(video, output, fps=1.0, ffmpeg=str(ffmpeg))
    assert len(rows) == 1
    assert Path(rows[0].image_path).name == "frame_00000000.png"
    assert rows[0].requested_sample_fps == 1.0

# [t2-validation] - START
from nnslr_tools.media import probe_frame_timestamps
import pytest


def _frame_probe(tmp_path, frames, container="hevc"):
    video = tmp_path / "video.hevc"
    video.write_bytes(b"x")
    ffprobe = tmp_path / "ffprobe"
    payload = {"frames": frames, "format": {"format_name": container}}
    _write_executable(ffprobe, "#!/bin/sh\ncat <<'EOF'\n" + json.dumps(payload) + "\nEOF\n")
    return probe_frame_timestamps(video, ffprobe=str(ffprobe))


def test_raw_hevc_duration_clock_is_explicit_and_not_capture_or_pts(tmp_path):
    frames = _frame_probe(tmp_path, [{"duration_time": "0.05"}, {"duration_time": "0.1"}, {"duration_time": "0.05"}])
    assert [x.media_time_s for x in frames] == pytest.approx([0, .05, .15])
    assert all(x.media_time_source == "raw_hevc_frame_duration" for x in frames)
    assert all(x.pts_time_s is None and x.best_effort_timestamp_s is None for x in frames)


@pytest.mark.parametrize("durations", [[".05", None], [".05", "0"], [".05", "NaN"]])
def test_missing_or_invalid_duration_does_not_fabricate_media_clock(tmp_path, durations):
    frames = _frame_probe(tmp_path, [{"duration_time": d} for d in durations])
    assert all(x.media_time_s is None for x in frames)


def test_container_and_mixed_timestamps_are_not_reconstructed(tmp_path):
    frames = _frame_probe(tmp_path, [{"duration_time": ".05"}, {"duration_time": ".05", "pts_time": "2"}])
    assert frames[0].media_time_s is None
    assert frames[1].media_time_s == 2
    frames = _frame_probe(tmp_path, [{"duration_time": ".05"}], container="mov,mp4")
    assert frames[0].media_time_s is None
# [t2-validation] - END

# [t2-validation] - START
import shutil
import subprocess
from nnslr_tools.media import make_clip


@pytest.fixture
def raw_hevc(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe required for local media integration")
    video = tmp_path / "fcamera.hevc"
    p = subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=red:s=64x48:r=20:d=1",
        "-f", "lavfi", "-i", "color=blue:s=64x48:r=20:d=1",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]",
        "-c:v", "libx265", "-x265-params", "pools=1:frame-threads=1:log-level=error:bframes=0",
        str(video),
    ], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return video


def test_raw_hevc_clip_seeks_to_real_frames_and_default_is_usable(raw_hevc, tmp_path):
    clip = make_clip(raw_hevc, tmp_path / "clip.mp4", start_s=1.25, end_s=1.75)
    probe = probe_video(clip)
    assert probe.frame_count == 10
    pixel = subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(clip), "-vf", "scale=1:1", "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ], capture_output=True, check=True).stdout
    assert len(pixel) == 3 and pixel[2] > 150 and pixel[0] < 80


def test_raw_hevc_frame_extraction_can_start_after_zero(raw_hevc, tmp_path):
    frames = extract_frames(raw_hevc, tmp_path / "frames", start_s=1.25, end_s=1.75)
    assert len(frames) == 10
# [t2-validation] - END

# [hevc-seek] - START
from nnslr_tools.media import MediaToolError


def test_empty_clip_is_an_error(raw_hevc, tmp_path):
    with pytest.raises(MediaToolError, match="clip_extraction_failed"):
        make_clip(raw_hevc, tmp_path / "empty.mp4", start_s=4, end_s=5)


def test_empty_extraction_is_an_error(raw_hevc, tmp_path):
    with pytest.raises(MediaToolError, match="frame_extraction_failed"):
        extract_frames(raw_hevc, tmp_path / "empty", start_s=4, end_s=5)
# [hevc-seek] - END

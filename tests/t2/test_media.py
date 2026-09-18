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

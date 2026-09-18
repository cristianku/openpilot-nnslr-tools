# [nnslr-t2] - START
"""Local video probing and frame extraction for comma camera recordings.

This module never talks to a comma device. It only operates on local files and
uses ffprobe/ffmpeg through subprocess with shell=False.

The contract intentionally keeps video presentation timing separate from camera
capture timing. ffprobe can expose media PTS/DTS; capture_mono_ns is established
later from qlog/rlog alignment and is never fabricated here.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence


class MediaToolError(RuntimeError):
    """Stable local-media failure with a machine-readable reason."""

    def __init__(self, reason: str, detail: str = "", *, stderr: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail
        self.stderr = stderr


def _resolve_executable(name: str, explicit: str | None = None) -> str:
    candidate = explicit or os.environ.get(f"NNSLR_{name.upper()}", "").strip() or name
    resolved = shutil.which(candidate)
    if resolved is None:
        raise MediaToolError("media_tool_not_found", f"{name}: {candidate}")
    return resolved


def _run_json(cmd: Sequence[str]) -> dict[str, Any]:
    proc = subprocess.run(list(cmd), capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise MediaToolError("media_tool_failed", " ".join(cmd), stderr=proc.stderr.strip())
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise MediaToolError("invalid_media_tool_json", str(exc), stderr=proc.stderr.strip()) from exc
    if not isinstance(payload, dict):
        raise MediaToolError("invalid_media_tool_json", "top-level payload is not an object")
    return payload


def _ratio_to_float(value: Any) -> float | None:
    if value in (None, "", "0/0", "N/A"):
        return None
    try:
        if isinstance(value, str) and "/" in value:
            f = Fraction(value)
            if f.denominator == 0:
                return None
            return float(f)
        out = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return out if math.isfinite(out) else None


def _number_or_none(value: Any) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int_or_none(value: Any) -> int | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def infer_camera_stream(path: Path) -> str | None:
    name = path.name.lower()
    if name == "fcamera.hevc" or name.startswith("fcamera."):
        return "narrow_road"
    if name == "ecamera.hevc" or name.startswith("ecamera."):
        return "wide_road"
    if name == "qcamera.ts" or name.startswith("qcamera."):
        return "q_narrow_road"
    if name == "dcamera.hevc" or name.startswith("dcamera."):
        return "cabin"
    return None


@dataclass(frozen=True)
class VideoProbe:
    path: str
    camera_stream: str | None
    codec: str | None
    codec_long_name: str | None
    container: str | None
    width: int | None
    height: int | None
    pixel_format: str | None
    color_range: str | None
    color_space: str | None
    color_transfer: str | None
    color_primaries: str | None
    nominal_fps: float | None
    average_fps: float | None
    time_base: str | None
    start_time_s: float | None
    duration_s: float | None
    frame_count: int | None
    ffprobe_version: str | None
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "camera_stream": self.camera_stream,
            "codec": self.codec,
            "codec_long_name": self.codec_long_name,
            "container": self.container,
            "width": self.width,
            "height": self.height,
            "pixel_format": self.pixel_format,
            "color_range": self.color_range,
            "color_space": self.color_space,
            "color_transfer": self.color_transfer,
            "color_primaries": self.color_primaries,
            "nominal_fps": self.nominal_fps,
            "average_fps": self.average_fps,
            "time_base": self.time_base,
            "start_time_s": self.start_time_s,
            "duration_s": self.duration_s,
            "frame_count": self.frame_count,
            "ffprobe_version": self.ffprobe_version,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class VideoFrameTiming:
    decoded_frame_index: int
    pts_time_s: float | None
    dts_time_s: float | None
    best_effort_timestamp_s: float | None
    key_frame: bool
    pict_type: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decoded_frame_index": self.decoded_frame_index,
            "pts_time_s": self.pts_time_s,
            "dts_time_s": self.dts_time_s,
            "best_effort_timestamp_s": self.best_effort_timestamp_s,
            "key_frame": self.key_frame,
            "pict_type": self.pict_type,
        }


@dataclass(frozen=True)
class ExtractedFrame:
    image_path: str
    output_index: int
    source_video: str
    requested_sample_fps: float | None
    start_s: float | None
    end_s: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "output_index": self.output_index,
            "source_video": self.source_video,
            "requested_sample_fps": self.requested_sample_fps,
            "start_s": self.start_s,
            "end_s": self.end_s,
        }


def ffprobe_version(*, ffprobe: str | None = None) -> str | None:
    exe = _resolve_executable("ffprobe", ffprobe)
    proc = subprocess.run([exe, "-version"], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return None
    first = proc.stdout.splitlines()
    return first[0].strip() if first else None


def probe_video(path: Path, *, ffprobe: str | None = None) -> VideoProbe:
    if not path.is_file():
        raise MediaToolError("video_not_found", str(path))
    exe = _resolve_executable("ffprobe", ffprobe)
    payload = _run_json([
        exe, "-v", "warning", "-count_frames", "-show_streams",
        "-show_format", "-print_format", "json", str(path),
    ])

    streams = payload.get("streams", [])
    if not isinstance(streams, list):
        streams = []
    video = next((s for s in streams if isinstance(s, dict) and s.get("codec_type") == "video"), None)
    fmt = payload.get("format") if isinstance(payload.get("format"), dict) else {}

    warnings: list[str] = []
    if video is None:
        warnings.append("no_video_stream")
        video = {}

    frame_count = _int_or_none(video.get("nb_read_frames"))
    if frame_count is None:
        frame_count = _int_or_none(video.get("nb_frames"))
        if frame_count is not None:
            warnings.append("frame_count_from_metadata_not_counted")

    duration_s = _number_or_none(video.get("duration"))
    if duration_s is None:
        duration_s = _number_or_none(fmt.get("duration"))
    if duration_s is None:
        warnings.append("duration_unknown")

    nominal = _ratio_to_float(video.get("r_frame_rate"))
    average = _ratio_to_float(video.get("avg_frame_rate"))
    if average is None and nominal is None:
        warnings.append("fps_unknown")

    start_video = _number_or_none(video.get("start_time"))
    return VideoProbe(
        path=str(path),
        camera_stream=infer_camera_stream(path),
        codec=video.get("codec_name") if isinstance(video.get("codec_name"), str) else None,
        codec_long_name=video.get("codec_long_name") if isinstance(video.get("codec_long_name"), str) else None,
        container=fmt.get("format_name") if isinstance(fmt.get("format_name"), str) else None,
        width=_int_or_none(video.get("width")),
        height=_int_or_none(video.get("height")),
        pixel_format=video.get("pix_fmt") if isinstance(video.get("pix_fmt"), str) else None,
        color_range=video.get("color_range") if isinstance(video.get("color_range"), str) else None,
        color_space=video.get("color_space") if isinstance(video.get("color_space"), str) else None,
        color_transfer=video.get("color_transfer") if isinstance(video.get("color_transfer"), str) else None,
        color_primaries=video.get("color_primaries") if isinstance(video.get("color_primaries"), str) else None,
        nominal_fps=nominal,
        average_fps=average,
        time_base=video.get("time_base") if isinstance(video.get("time_base"), str) else None,
        start_time_s=start_video if start_video is not None else _number_or_none(fmt.get("start_time")),
        duration_s=duration_s,
        frame_count=frame_count,
        ffprobe_version=ffprobe_version(ffprobe=exe),
        warnings=tuple(warnings),
    )


def probe_frame_timestamps(path: Path, *, ffprobe: str | None = None) -> list[VideoFrameTiming]:
    """Return media presentation/decode timing; never camera capture timing."""
    if not path.is_file():
        raise MediaToolError("video_not_found", str(path))
    exe = _resolve_executable("ffprobe", ffprobe)
    payload = _run_json([
        exe, "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries",
        "frame=pts_time,pkt_dts_time,best_effort_timestamp_time,key_frame,pict_type",
        "-print_format", "json", str(path),
    ])
    frames = payload.get("frames", [])
    if not isinstance(frames, list):
        raise MediaToolError("invalid_frame_probe", "frames is not a list")

    out: list[VideoFrameTiming] = []
    for idx, raw in enumerate(frames):
        if not isinstance(raw, dict):
            continue
        out.append(VideoFrameTiming(
            decoded_frame_index=idx,
            pts_time_s=_number_or_none(raw.get("pts_time")),
            dts_time_s=_number_or_none(raw.get("pkt_dts_time")),
            best_effort_timestamp_s=_number_or_none(raw.get("best_effort_timestamp_time")),
            key_frame=bool(raw.get("key_frame") == 1),
            pict_type=raw.get("pict_type") if isinstance(raw.get("pict_type"), str) else None,
        ))
    return out


def extract_frames(
    video: Path,
    output_dir: Path,
    *,
    fps: float | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
    ffmpeg: str | None = None,
    overwrite: bool = False,
) -> list[ExtractedFrame]:
    """Decode local video frames to PNG without resizing."""
    if not video.is_file():
        raise MediaToolError("video_not_found", str(video))
    if fps is not None and (not math.isfinite(fps) or fps <= 0):
        raise MediaToolError("invalid_fps", repr(fps))
    if start_s is not None and (not math.isfinite(start_s) or start_s < 0):
        raise MediaToolError("invalid_start", repr(start_s))
    if end_s is not None and (not math.isfinite(end_s) or end_s < 0):
        raise MediaToolError("invalid_end", repr(end_s))
    if start_s is not None and end_s is not None and end_s <= start_s:
        raise MediaToolError("invalid_range", f"{start_s}..{end_s}")

    exe = _resolve_executable("ffmpeg", ffmpeg)
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "frame_%08d.png"

    existing = sorted(output_dir.glob("frame_*.png"))
    if existing and not overwrite:
        raise MediaToolError("output_not_empty", str(output_dir))

    cmd: list[str] = [exe, "-hide_banner", "-loglevel", "warning"]
    cmd.append("-y" if overwrite else "-n")
    if start_s is not None:
        cmd += ["-ss", f"{start_s:.9f}"]
    cmd += ["-i", str(video)]
    if end_s is not None:
        duration = end_s - (start_s or 0.0)
        cmd += ["-t", f"{duration:.9f}"]
    if fps is not None:
        cmd += ["-vf", f"fps={fps:.12g}"]
    cmd += ["-map", "0:v:0", "-vsync", "0", "-start_number", "0", str(pattern)]

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise MediaToolError("frame_extraction_failed", " ".join(cmd), stderr=proc.stderr.strip())

    files = sorted(output_dir.glob("frame_*.png"))
    return [
        ExtractedFrame(
            image_path=str(path),
            output_index=i,
            source_video=str(video),
            requested_sample_fps=fps,
            start_s=start_s,
            end_s=end_s,
        )
        for i, path in enumerate(files)
    ]


def make_clip(
    video: Path,
    output: Path,
    *,
    start_s: float,
    end_s: float,
    ffmpeg: str | None = None,
    reencode: bool = False,
) -> Path:
    """Create a local clip for review/candidate labeling."""
    if not video.is_file():
        raise MediaToolError("video_not_found", str(video))
    if not (math.isfinite(start_s) and math.isfinite(end_s)) or start_s < 0 or end_s <= start_s:
        raise MediaToolError("invalid_range", f"{start_s}..{end_s}")
    exe = _resolve_executable("ffmpeg", ffmpeg)
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        exe, "-hide_banner", "-loglevel", "warning", "-y", "-ss",
        f"{start_s:.9f}", "-i", str(video), "-t", f"{end_s - start_s:.9f}",
        "-map", "0:v:0",
    ]
    if reencode:
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an"]
    else:
        cmd += ["-c", "copy"]
    cmd.append(str(output))

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise MediaToolError("clip_extraction_failed", " ".join(cmd), stderr=proc.stderr.strip())
    return output
# [nnslr-t2] - END

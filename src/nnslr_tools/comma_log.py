# [nnslr-t2] - START
"""Read LOCAL qlog/rlog metadata through a local openpilot checkout.

No network or device access occurs here. A child interpreter loads only the
Cap'n Proto schemas from a local checkout, with optional pycapnp/zstandard in
that interpreter. It does not import the openpilot runtime or its dependency
stack. NNSLR_OPENDBC_ROOT can select the compatible car.capnp checkout.

The adapter preserves the fields needed to join camera video presentation order
to camera capture timestamps. openpilot's EncodeIndex schema defines segmentId
as the index into the camera file in presentation order; segmentIdEncode is the
index in encode order. Those are kept distinct.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class CommaLogError(RuntimeError):
    def __init__(self, reason: str, detail: str = "", *, stderr: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail
        self.stderr = stderr


@dataclass(frozen=True)
class EncodeIndexEvent:
    service: str
    log_mono_time: int
    frame_id: int
    segment_num: int
    segment_id: int
    segment_id_encode: int
    timestamp_sof: int
    timestamp_eof: int
    encode_type: str | None = None

    @property
    def presentation_index(self) -> int:
        return self.segment_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "encode_index",
            "service": self.service,
            "log_mono_time": self.log_mono_time,
            "frame_id": self.frame_id,
            "segment_num": self.segment_num,
            "segment_id": self.segment_id,
            "segment_id_encode": self.segment_id_encode,
            "timestamp_sof": self.timestamp_sof,
            "timestamp_eof": self.timestamp_eof,
            "encode_type": self.encode_type,
        }


@dataclass(frozen=True)
class CameraStateEvent:
    service: str
    log_mono_time: int
    frame_id: int
    encode_id: int
    timestamp_sof: int
    timestamp_eof: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "camera_state",
            "service": self.service,
            "log_mono_time": self.log_mono_time,
            "frame_id": self.frame_id,
            "encode_id": self.encode_id,
            "timestamp_sof": self.timestamp_sof,
            "timestamp_eof": self.timestamp_eof,
        }


@dataclass(frozen=True)
class MapSpeedEvent:
    service: str
    log_mono_time: int
    speed_limit: float | None
    speed_limit_ahead: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "map_speed",
            "service": self.service,
            "log_mono_time": self.log_mono_time,
            "speed_limit": self.speed_limit,
            "speed_limit_ahead": self.speed_limit_ahead,
        }


LogEvent = EncodeIndexEvent | CameraStateEvent | MapSpeedEvent


_CHILD_SCRIPT = r"""
import json
import sys

# [schema-reader] - START
import bz2
from pathlib import Path
import capnp

cereal_dir, car_dir, log_path = sys.argv[1:4]
capnp.remove_import_hook()
schema = capnp.load(str(Path(cereal_dir) / "log.capnp"), imports=[cereal_dir, car_dir])
path = Path(log_path)
if path.suffix == ".zst":
    import zstandard
    compressed = path.read_bytes()
    if not compressed:
        raise ValueError("empty zstd log")
    chunks = []
    while compressed:
        decoder = zstandard.ZstdDecompressor().decompressobj()
        chunks.append(decoder.decompress(compressed))
        if not decoder.eof:
            raise ValueError("truncated zstd frame")
        compressed = decoder.unused_data
    data = b"".join(chunks)
elif path.suffix == ".bz2":
    data = bz2.decompress(path.read_bytes())
else:
    data = path.read_bytes()
# read_multiple_bytes raises for malformed/truncated messages; never return a
# successful partial metadata stream when decoding fails.
messages = sorted(schema.Event.read_multiple_bytes(data), key=lambda msg: msg.logMonoTime)
# [schema-reader] - END

ENCODE = {
    "narrowRoadEncodeIdx",
    "wideRoadEncodeIdx",
    "qNarrowRoadEncodeIdx",
}
CAM = {
    "narrowRoadCameraState",
    "wideRoadCameraState",
}
MAP = {
    "liveMapDataSP",
    "liveMapData",
}

def sval(v):
    try:
        return str(v)
    except Exception:
        return None

def fval(obj, name):
    try:
        v = getattr(obj, name)
    except Exception:
        return None
    try:
        return float(v)
    except Exception:
        return None

for msg in messages:
    try:
        which = msg.which()
    except Exception:
        continue
    if which not in ENCODE and which not in CAM and which not in MAP:
        continue

    payload = getattr(msg, which)
    base = {"service": which, "log_mono_time": int(msg.logMonoTime)}

    if which in ENCODE:
        out = {
            **base,
            "kind": "encode_index",
            "frame_id": int(payload.frameId),
            "segment_num": int(payload.segmentNum),
            "segment_id": int(payload.segmentId),
            "segment_id_encode": int(payload.segmentIdEncode),
            "timestamp_sof": int(payload.timestampSof),
            "timestamp_eof": int(payload.timestampEof),
            "encode_type": sval(payload.type),
        }
    elif which in CAM:
        out = {
            **base,
            "kind": "camera_state",
            "frame_id": int(payload.frameId),
            "encode_id": int(payload.encodeId),
            "timestamp_sof": int(payload.timestampSof),
            "timestamp_eof": int(payload.timestampEof),
        }
    else:
        out = {
            **base,
            "kind": "map_speed",
            "speed_limit": fval(payload, "speedLimit"),
            "speed_limit_ahead": fval(payload, "speedLimitAhead"),
        }
    print(json.dumps(out, sort_keys=True))
"""


def resolve_openpilot_root(explicit: Path | None = None) -> Path:
    raw = explicit or (Path(os.environ["NNSLR_OPENPILOT_ROOT"]) if os.environ.get("NNSLR_OPENPILOT_ROOT") else None)
    if raw is None:
        raise CommaLogError(
            "openpilot_root_not_configured",
            "set NNSLR_OPENPILOT_ROOT or pass --openpilot-root",
        )
    root = Path(raw).expanduser().resolve()
    if not any((root / prefix / "log.capnp").is_file() for prefix in ("openpilot/cereal", "cereal")):
        raise CommaLogError("invalid_openpilot_root", "missing openpilot/cereal/log.capnp or cereal/log.capnp")
    return root


# [schema-reader] - START
def resolve_schema_paths(openpilot_root: Path, opendbc_root: Path | None = None) -> tuple[Path, Path]:
    root = resolve_openpilot_root(openpilot_root)
    cereal = next(root / prefix for prefix in ("openpilot/cereal", "cereal")
                  if (root / prefix / "log.capnp").is_file())
    raw = opendbc_root or os.environ.get("NNSLR_OPENDBC_ROOT")
    if raw:
        dbc = Path(raw).expanduser().resolve()
        candidates = [dbc / "opendbc/car", dbc / "car", dbc]
    else:
        candidates = [root / "opendbc/opendbc/car", root / "opendbc/car",
                      root / "openpilot/opendbc/opendbc/car", root / "openpilot/opendbc/car"]
    car = next((p for p in candidates if (p / "car.capnp").is_file()), None)
    if car is None:
        raise CommaLogError("car_schema_not_found", "set NNSLR_OPENDBC_ROOT or --opendbc-root to a compatible checkout")
    return cereal, car
# [schema-reader] - END


def _strict_int(raw: dict[str, Any], name: str) -> int:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CommaLogError("invalid_log_metadata", f"{name}={value!r}")
    return value


def _optional_float(raw: dict[str, Any], name: str) -> float | None:
    value = raw.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CommaLogError("invalid_log_metadata", f"{name}={value!r}")
    return float(value)


def event_from_dict(raw: dict[str, Any]) -> LogEvent:
    kind = raw.get("kind")
    service = raw.get("service")
    if not isinstance(service, str) or not service:
        raise CommaLogError("invalid_log_metadata", "missing service")
    mono = _strict_int(raw, "log_mono_time")

    if kind == "encode_index":
        return EncodeIndexEvent(
            service=service,
            log_mono_time=mono,
            frame_id=_strict_int(raw, "frame_id"),
            segment_num=_strict_int(raw, "segment_num"),
            segment_id=_strict_int(raw, "segment_id"),
            segment_id_encode=_strict_int(raw, "segment_id_encode"),
            timestamp_sof=_strict_int(raw, "timestamp_sof"),
            timestamp_eof=_strict_int(raw, "timestamp_eof"),
            encode_type=raw.get("encode_type") if isinstance(raw.get("encode_type"), str) else None,
        )
    if kind == "camera_state":
        return CameraStateEvent(
            service=service,
            log_mono_time=mono,
            frame_id=_strict_int(raw, "frame_id"),
            encode_id=_strict_int(raw, "encode_id"),
            timestamp_sof=_strict_int(raw, "timestamp_sof"),
            timestamp_eof=_strict_int(raw, "timestamp_eof"),
        )
    if kind == "map_speed":
        return MapSpeedEvent(
            service=service,
            log_mono_time=mono,
            speed_limit=_optional_float(raw, "speed_limit"),
            speed_limit_ahead=_optional_float(raw, "speed_limit_ahead"),
        )
    raise CommaLogError("invalid_log_metadata", f"unknown kind {kind!r}")


def read_log_metadata(
    log_path: Path,
    *,
    openpilot_root: Path | None = None,
    python_executable: str | None = None,
    opendbc_root: Path | None = None,
) -> list[LogEvent]:
    """Read supported metadata from a local qlog/rlog file."""
    if not log_path.is_file():
        raise CommaLogError("log_not_found", str(log_path))
    root = resolve_openpilot_root(openpilot_root)
    cereal, car = resolve_schema_paths(root, opendbc_root)
    exe = python_executable or os.environ.get("NNSLR_OPENPILOT_PYTHON") or sys.executable

    proc = subprocess.run(
        [exe, "-c", _CHILD_SCRIPT, str(cereal), str(car), str(log_path.resolve())],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise CommaLogError(
            "log_schema_reader_failed",
            f"{log_path} (exit {proc.returncode})",
            stderr=proc.stderr.strip(),
        )

    events: list[LogEvent] = []
    for line_no, line in enumerate(proc.stdout.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CommaLogError("invalid_logreader_output", f"line {line_no}: {exc}") from exc
        if not isinstance(raw, dict):
            raise CommaLogError("invalid_logreader_output", f"line {line_no}: not an object")
        events.append(event_from_dict(raw))
    return events


def encode_events(
    events: Iterable[LogEvent],
    *,
    service: str,
    segment_num: int | None = None,
) -> list[EncodeIndexEvent]:
    out = [
        e for e in events
        if isinstance(e, EncodeIndexEvent)
        and e.service == service
        and (segment_num is None or e.segment_num == segment_num)
    ]
    return sorted(out, key=lambda e: (e.segment_num, e.segment_id, e.log_mono_time))


def camera_state_events(events: Iterable[LogEvent], *, service: str) -> list[CameraStateEvent]:
    return sorted(
        [e for e in events if isinstance(e, CameraStateEvent) and e.service == service],
        key=lambda e: (e.frame_id, e.log_mono_time),
    )


def map_speed_events(events: Iterable[LogEvent]) -> list[MapSpeedEvent]:
    return sorted(
        [e for e in events if isinstance(e, MapSpeedEvent)],
        key=lambda e: e.log_mono_time,
    )
# [nnslr-t2] - END

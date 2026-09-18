# [nnslr-t2] - START
"""Route/segment identity, video/log manifest and frame-to-log alignment (T2).

This module implements the *contract* and the *deterministic join logic* of
plan §5 (dataset, annotation and reproducibility) and §T2 (local manifest and
frame alignment). It is deliberately pure, stdlib-only and operates on
**explicit input records** (never on a live device, never on a remote host,
never on a video decoder). That keeps the CPU/synthetic quickstart honest: the
alignment *procedure* is implemented and exercised over synthetic fixtures,
while **real frame/log alignment is NOT established** until actual comma video
is available and a decoder is authorized (plan §5.3, §T2 acceptance).

What lives here:
- route/segment identity (:class:`RouteIdentity`) and segment availability
  (:class:`SegmentStatus`);
- a safe, content-addressed file inventory (:class:`RouteFile`,
  :class:`RouteManifest`) that preserves genuine gaps (a missing segment is a
  first-class state, not an error to paper over);
- the per-frame provenance record with the exact §5.2 field set
  (:class:`FrameManifestRecord`);
- a deterministic, pure :func:`align_frames` join that reports matched /
  unmatched / duplicate / discontinuous records and an estimated error bound,
  without ever silently remapping a duplicate frame id or interpolating across
  a missing segment;
- path safety (:func:`normalize_relpath`) and content hashing
  (:func:`sha256_of`) that reject remote URLs, absolute paths and ``..``
  traversal.

Nothing here reads ``NNSLR_DATA_ROOT`` implicitly: the data root is always an
explicit argument, so the module stays importable and testable in any
environment. The CLI is the only layer that resolves the environment variable.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

MANIFEST_SCHEMA_VERSION = 1

# A comma route id is ``<counter>--<hex>`` (e.g. ``00000050--eabf0e8324``).
# A full segment dir appends ``--<segment_index>``. We validate the shape so a
# typo or a path fragment cannot masquerade as a route identity.
_ROUTE_ID_RE = re.compile(r"^(0*[1-9][0-9]*)--([0-9a-f]{6,32})$")
_SEGMENT_DIR_RE = re.compile(r"^(0*[1-9][0-9]*)--([0-9a-f]{6,32})--(0*[0-9]+)$")

# Schemes that are never acceptable as a local file reference (plan §5.2:
# "reject traversal or remote URLs").
_REMOTE_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:")


class SegmentStatus(str, Enum):
    """Availability of a route segment (plan §5.2: preserve genuine gaps)."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    MISSING = "missing"


class FileKind(str, Enum):
    """What a raw file under ``raw/`` is. ``video`` carries a stream; the log
    kinds do not (they are the metadata side of the join)."""

    VIDEO = "video"
    RLOG = "rlog"
    QLOG = "qlog"
    OTHER = "other"


class AlignmentStatus(str, Enum):
    """How a frame's timing was resolved (plan §5.2 ``alignment_status``).

    A ``classification_only`` frame is a legitimate *crop* source but is
    excluded from any precise passage/latency claim (plan §5.2 last rule)."""

    EXACT = "exact"
    BOUNDED_ESTIMATE = "bounded_estimate"
    UNRESOLVED = "unresolved"


class NnslerManifestError(ValueError):
    """Raised for unsafe paths, malformed identities or malformed records."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def parse_route_id(value: str) -> tuple[int, str]:
    """Split ``<counter>--<hex>`` into ``(counter, hex)``.

    Raises :class:`NnslerManifestError` on anything that is not a well-formed
    route id. A segment dir (with a trailing ``--<n>``) is *not* a route id.
    """
    if not isinstance(value, str):
        raise NnslerManifestError("invalid_route_id", f"not a string: {value!r}")
    m = _ROUTE_ID_RE.match(value)
    if not m:
        raise NnslerManifestError("invalid_route_id", value)
    return int(m.group(1)), m.group(2)


def parse_segment_dir(value: str) -> tuple[int, str, int]:
    """Split ``<counter>--<hex>--<seg>`` into ``(counter, hex, seg)``."""
    if not isinstance(value, str):
        raise NnslerManifestError("invalid_segment_dir", f"not a string: {value!r}")
    m = _SEGMENT_DIR_RE.match(value)
    if not m:
        raise NnslerManifestError("invalid_segment_dir", value)
    return int(m.group(1)), m.group(2), int(m.group(3))


@dataclass(frozen=True)
class RouteIdentity:
    """A single route/segment. The counter + hex is the route; the segment
    index orders it monotonically within the route (plan §5.2)."""

    route_counter: int
    route_hex: str
    segment_index: int

    @classmethod
    def from_route(cls, route_id: str, segment_index: int) -> "RouteIdentity":
        counter, hexid = parse_route_id(route_id)
        if not isinstance(segment_index, int) or isinstance(segment_index, bool) or segment_index < 0:
            raise NnslerManifestError("invalid_segment_index", str(segment_index))
        return cls(counter, hexid, segment_index)

    @classmethod
    def from_segment_dir(cls, seg_dir: str) -> "RouteIdentity":
        counter, hexid, seg = parse_segment_dir(seg_dir)
        return cls(counter, hexid, seg)

    @property
    def route_id(self) -> str:
        return f"{self.route_counter:08d}--{self.route_hex}"

    @property
    def segment_dir(self) -> str:
        return f"{self.route_id}--{self.segment_index}"

    @property
    def key(self) -> str:
        """Stable sort/join key within a route (monotonic segment order)."""
        return self.segment_dir

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_counter": self.route_counter,
            "route_hex": self.route_hex,
            "segment_index": self.segment_index,
            "route_id": self.route_id,
            "segment_dir": self.segment_dir,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RouteIdentity":
        if not isinstance(data, Mapping):
            raise NnslerManifestError("invalid_route_identity", repr(type(data)))
        counter = _strict_int(data.get("route_counter"), "route_counter")
        hexid = _strict_str(data.get("route_hex"), "route_hex")
        seg = _strict_int(data.get("segment_index"), "segment_index")
        return cls(counter, hexid, seg)


# ---------------------------------------------------------------------------
# Path safety + content hash
# ---------------------------------------------------------------------------

def _strict_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise NnslerManifestError("invalid_numeric_type", f"{field} must be an int, got {value!r}")
    return value


def _strict_str(value: Any, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise NnslerManifestError("invalid_type", f"{field} must be a string, got {value!r}")
    if not allow_empty and not value:
        raise NnslerManifestError("invalid_empty", f"{field} must be non-empty")
    return value


def _strict_optional_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    return _strict_int(value, field)


def _strict_optional_str(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _strict_str(value, field)


def normalize_relpath(data_root: Path, target: Path) -> str:
    """Return ``target`` as a POSIX relpath under ``data_root``.

    Rejects (plan §5.2): remote URLs (any ``scheme:`` prefix), absolute paths,
    and ``..`` traversal that escapes the data root. A path that equals the
    data root or is a proper ancestor of it is also rejected, because a raw
    file must be *inside* the root.
    """
    if _REMOTE_SCHEME_RE.match(str(target)):
        raise NnslerManifestError("remote_path", str(target))
    if target.is_absolute():
        raise NnslerManifestError("absolute_path", str(target))
    root = data_root.resolve()
    resolved = (root / target).resolve()
    try:
        rel = resolved.relative_to(root)
    except ValueError as exc:
        raise NnslerManifestError("path_traversal", str(target)) from exc
    if rel == Path("."):
        raise NnslerManifestError("path_is_root", str(target))
    return rel.as_posix()


def sha256_of(path: Path) -> str:
    """Content hash of a local file. Streaming: memory-bounded on large video.

    Refuses to hash anything that is not an existing regular file. This is the
    *single* place a file is opened for hashing; a whole video is hashed once
    and referenced by id from each frame record (plan §5.1).
    """
    if not path.is_file():
        raise NnslerManifestError("not_a_file", str(path))
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def classify_filekind(name: str) -> FileKind:
    """Classify a raw file by its suffix. Unknown → OTHER (never a video)."""
    lower = name.lower()
    if lower.endswith((".mp4", ".mov", ".mkv", ".h264", ".hevc", ".265")):
        return FileKind.VIDEO
    if lower.endswith((".rlog.zst", ".rlog")):
        return FileKind.RLOG
    if lower.endswith((".qlog.zst", ".qlog")):
        return FileKind.QLOG
    return FileKind.OTHER


# ---------------------------------------------------------------------------
# Route manifest (file inventory + gaps)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RouteFile:
    """One raw file under ``raw/``. ``sha256`` is the content identity; the
    same hash is referenced by every frame record that came from it, so a
    video is never re-hashed per frame (plan §5.1)."""

    route: RouteIdentity
    relpath: str
    kind: FileKind
    stream: str | None
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route.to_dict(),
            "relpath": self.relpath,
            "kind": self.kind.value,
            "stream": self.stream,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RouteFile":
        if not isinstance(data, Mapping):
            raise NnslerManifestError("invalid_route_file", repr(type(data)))
        route = RouteIdentity.from_dict(data.get("route"))
        relpath = _strict_str(data.get("relpath"), "relpath")
        kind = FileKind(_strict_str(data.get("kind"), "kind"))
        stream = _strict_optional_str(data.get("stream"), "stream")
        sha256 = _strict_str(data.get("sha256"), "sha256")
        size = _strict_int(data.get("size_bytes"), "size_bytes")
        if size < 0:
            raise NnslerManifestError("invalid_size", str(size))
        if kind == FileKind.VIDEO and stream is None:
            raise NnslerManifestError("video_missing_stream", relpath)
        return cls(route, relpath, kind, stream, sha256, size)


@dataclass
class RouteManifest:
    """The §5.1 ``manifests/routes.jsonl`` contract: file identity,
    availability, hashes and gaps. ``segment_status`` records, for every
    *declared* segment, whether it is complete, partial or missing — a missing
    segment stays visible (plan §5.2: preserve genuine gaps)."""

    schema_version: int
    data_root: str
    declared_segments: tuple[int, ...]
    files: tuple[RouteFile, ...]
    segment_status: dict[int, SegmentStatus]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "data_root": self.data_root,
            "declared_segments": list(self.declared_segments),
            "segment_status": {str(k): v.value for k, v in sorted(self.segment_status.items())},
            "files": [f.to_dict() for f in self.files],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RouteManifest":
        if not isinstance(data, Mapping):
            raise NnslerManifestError("invalid_route_manifest", repr(type(data)))
        schema = _strict_int(data.get("schema_version"), "schema_version")
        if schema != MANIFEST_SCHEMA_VERSION:
            raise NnslerManifestError("unsupported_schema_version", str(schema))
        data_root = _strict_str(data.get("data_root"), "data_root", allow_empty=True)
        declared = tuple(_strict_int(v, f"declared_segments[{i}]") for i, v in enumerate(data.get("declared_segments", [])))
        status_raw = data.get("segment_status", {})
        if not isinstance(status_raw, Mapping):
            raise NnslerManifestError("invalid_segment_status", repr(type(status_raw)))
        segment_status = {
            _strict_int(int(k), f"segment_status[{k}]"): SegmentStatus(_strict_str(v, f"segment_status[{k}]"))
            for k, v in status_raw.items()
        }
        files = tuple(RouteFile.from_dict(f) for f in data.get("files", []))
        return cls(schema, data_root, declared, files, segment_status)

    def to_jsonl(self) -> str:
        """One JSON object per line: header first, then one object per file.
        Deterministic: files are emitted in (segment_index, stream, relpath)
        order and dict keys are sorted."""
        header = {
            "schema_version": self.schema_version,
            "data_root": self.data_root,
            "declared_segments": list(self.declared_segments),
            "segment_status": {str(k): v.value for k, v in sorted(self.segment_status.items())},
            "file_count": len(self.files),
        }
        ordered = sorted(self.files, key=lambda f: (f.route.segment_index, f.stream or "", f.relpath))
        lines = [json.dumps(header, sort_keys=True)]
        for f in ordered:
            lines.append(json.dumps(f.to_dict(), sort_keys=True))
        return "\n".join(lines) + "\n"

    @classmethod
    def from_jsonl(cls, text: str) -> "RouteManifest":
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            raise NnslerManifestError("empty_manifest", "no lines")
        header = json.loads(lines[0])
        files = [RouteFile.from_dict(json.loads(ln)) for ln in lines[1:]]
        declared = tuple(_strict_int(v, "declared_segments") for v in header.get("declared_segments", []))
        status_raw = header.get("segment_status", {})
        segment_status = {
            _strict_int(int(k), f"segment_status[{k}]"): SegmentStatus(_strict_str(v, f"segment_status[{k}]"))
            for k, v in status_raw.items()
        }
        return cls(header["schema_version"], header.get("data_root", ""), declared, tuple(files), segment_status)

    def gap_report(self) -> dict[str, Any]:
        """Deterministic view of what is missing/partial (feeds the alignment
        report). Missing segments are listed; complete ones are not."""
        missing = sorted(seg for seg, st in self.segment_status.items() if st == SegmentStatus.MISSING)
        partial = sorted(seg for seg, st in self.segment_status.items() if st == SegmentStatus.PARTIAL)
        return {
            "missing_segments": missing,
            "partial_segments": partial,
            "complete_segments": sorted(seg for seg, st in self.segment_status.items() if st == SegmentStatus.COMPLETE),
        }


# ---------------------------------------------------------------------------
# Frame manifest record (§5.2 field set)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrameManifestRecord:
    """One decoded frame with its provenance and timing resolution.

    Absent fields are represented by ``None`` (JSON ``null``), never by a
    fabricated default. ``calibration_id`` is explicitly ``None`` when
    calibration was not recorded/validated (plan §5.2)."""

    schema_version: int
    route_id: str
    session_id: str
    segment_index: int
    stream: str
    video_relpath: str
    video_sha256: str
    decoded_frame_index: int | None
    encoded_frame_id: int | None
    capture_mono_ns: int | None
    capture_reference: str
    alignment_status: str
    estimated_error_ns: int | None
    width: int
    height: int
    pixel_format: str
    color_metadata: str | None
    calibration_id: str | None
    route_group: str
    site_group: str
    encounter_id: str
    frame_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "route_id": self.route_id,
            "session_id": self.session_id,
            "segment_index": self.segment_index,
            "stream": self.stream,
            "video_relpath": self.video_relpath,
            "video_sha256": self.video_sha256,
            "decoded_frame_index": self.decoded_frame_index,
            "encoded_frame_id": self.encoded_frame_id,
            "capture_mono_ns": self.capture_mono_ns,
            "capture_reference": self.capture_reference,
            "alignment_status": self.alignment_status,
            "estimated_error_ns": self.estimated_error_ns,
            "width": self.width,
            "height": self.height,
            "pixel_format": self.pixel_format,
            "color_metadata": self.color_metadata,
            "calibration_id": self.calibration_id,
            "route_group": self.route_group,
            "site_group": self.site_group,
            "encounter_id": self.encounter_id,
            "frame_sha256": self.frame_sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FrameManifestRecord":
        if not isinstance(data, Mapping):
            raise NnslerManifestError("invalid_frame_record", repr(type(data)))
        schema = _strict_int(data.get("schema_version"), "schema_version")
        if schema != MANIFEST_SCHEMA_VERSION:
            raise NnslerManifestError("unsupported_schema_version", str(schema))
        width = _strict_int(data.get("width"), "width")
        height = _strict_int(data.get("height"), "height")
        if width <= 0 or height <= 0:
            raise NnslerManifestError("invalid_dimensions", f"{width}x{height}")
        est = _strict_optional_int(data.get("estimated_error_ns"), "estimated_error_ns")
        if est is not None and est < 0:
            raise NnslerManifestError("invalid_estimated_error", str(est))
        return cls(
            schema,
            _strict_str(data.get("route_id"), "route_id"),
            _strict_str(data.get("session_id"), "session_id"),
            _strict_int(data.get("segment_index"), "segment_index"),
            _strict_str(data.get("stream"), "stream"),
            _strict_str(data.get("video_relpath"), "video_relpath"),
            _strict_str(data.get("video_sha256"), "video_sha256"),
            _strict_optional_int(data.get("decoded_frame_index"), "decoded_frame_index"),
            _strict_optional_int(data.get("encoded_frame_id"), "encoded_frame_id"),
            _strict_optional_int(data.get("capture_mono_ns"), "capture_mono_ns"),
            _strict_str(data.get("capture_reference"), "capture_reference"),
            _strict_str(data.get("alignment_status"), "alignment_status"),
            est,
            width,
            height,
            _strict_str(data.get("pixel_format"), "pixel_format"),
            _strict_optional_str(data.get("color_metadata"), "color_metadata"),
            _strict_optional_str(data.get("calibration_id"), "calibration_id"),
            _strict_str(data.get("route_group"), "route_group"),
            _strict_str(data.get("site_group"), "site_group"),
            _strict_str(data.get("encounter_id"), "encounter_id"),
            _strict_optional_str(data.get("frame_sha256"), "frame_sha256"),
        )


# ---------------------------------------------------------------------------
# Alignment (pure, deterministic join)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EncodeIndexRecord:
    """One metadata record from the pinned encode-index (the *metadata side*
    of the join). ``encoded_frame_id`` is the join key; the capture timestamp
    and reference say what the index points at (plan §5.3 step 2)."""

    encoded_frame_id: int
    capture_mono_ns: int | None
    capture_reference: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EncodeIndexRecord":
        if not isinstance(data, Mapping):
            raise NnslerManifestError("invalid_encode_index", repr(type(data)))
        return cls(
            _strict_int(data.get("encoded_frame_id"), "encoded_frame_id"),
            _strict_optional_int(data.get("capture_mono_ns"), "capture_mono_ns"),
            _strict_str(data.get("capture_reference"), "capture_reference"),
        )


@dataclass(frozen=True)
class VideoFrameRecord:
    """One decoded video frame (the *media side* of the join). The capture
    timestamp may be absent (``None``) — that frame can still be cropped for
    classification but cannot support a timing claim (plan §5.2 last rule)."""

    decoded_frame_index: int
    capture_mono_ns: int | None
    width: int
    height: int
    pixel_format: str
    color_metadata: str | None
    frame_sha256: str | None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VideoFrameRecord":
        if not isinstance(data, Mapping):
            raise NnslerManifestError("invalid_video_frame", repr(type(data)))
        width = _strict_int(data.get("width"), "width")
        height = _strict_int(data.get("height"), "height")
        if width <= 0 or height <= 0:
            raise NnslerManifestError("invalid_dimensions", f"{width}x{height}")
        return cls(
            _strict_int(data.get("decoded_frame_index"), "decoded_frame_index"),
            _strict_optional_int(data.get("capture_mono_ns"), "capture_mono_ns"),
            width,
            height,
            _strict_str(data.get("pixel_format"), "pixel_format"),
            _strict_optional_str(data.get("color_metadata"), "color_metadata"),
            _strict_optional_str(data.get("frame_sha256"), "frame_sha256"),
        )


@dataclass
class AlignmentReport:
    """Deterministic output of :func:`align_frames`. It is a *report*, not a
    correction: nothing here is silently fixed. Duplicates, gaps and
    unresolved timing all remain visible (plan §5.3 step 5, §T2 acceptance)."""

    schema_version: int
    stream: str
    session_id: str
    matched: int
    unmatched_frames: tuple[dict[str, Any], ...]
    unmatched_metadata: tuple[dict[str, Any], ...]
    duplicate_frame_ids: tuple[dict[str, Any], ...]
    discontinuities: tuple[dict[str, Any], ...]
    missing_segments: tuple[int, ...]
    classification_only_frames: int
    max_estimated_error_ns: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stream": self.stream,
            "session_id": self.session_id,
            "matched": self.matched,
            "unmatched_frames": list(self.unmatched_frames),
            "unmatched_metadata": list(self.unmatched_metadata),
            "duplicate_frame_ids": list(self.duplicate_frame_ids),
            "discontinuities": list(self.discontinuities),
            "missing_segments": list(self.missing_segments),
            "classification_only_frames": self.classification_only_frames,
            "max_estimated_error_ns": self.max_estimated_error_ns,
            # The ten-clip visual report (plan §5.3 step 6) needs real video +
            # human inspection. It is always false here until produced.
            "visual_alignment_done": False,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AlignmentReport":
        if not isinstance(data, Mapping):
            raise NnslerManifestError("invalid_alignment_report", repr(type(data)))
        return cls(
            _strict_int(data.get("schema_version"), "schema_version"),
            _strict_str(data.get("stream"), "stream"),
            _strict_str(data.get("session_id"), "session_id"),
            _strict_int(data.get("matched"), "matched"),
            tuple(data.get("unmatched_frames", [])),
            tuple(data.get("unmatched_metadata", [])),
            tuple(data.get("duplicate_frame_ids", [])),
            tuple(data.get("discontinuities", [])),
            tuple(_strict_int(v, "missing_segments") for v in data.get("missing_segments", [])),
            _strict_int(data.get("classification_only_frames"), "classification_only_frames"),
            _strict_optional_int(data.get("max_estimated_error_ns"), "max_estimated_error_ns"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)


def align_frames(
    stream: str,
    session_id: str,
    video_frames: Sequence[VideoFrameRecord],
    encode_index: Sequence[EncodeIndexRecord],
    *,
    missing_segments: Sequence[int] = (),
    route_group: str = "unknown",
    site_group: str = "unknown",
    encounter_id: str = "unknown",
    video_relpath: str = "unknown",
    video_sha256: str = "unknown",
    route_id: str = "unknown",
    segment_index: int = 0,
    estimated_error_ns: int | None = None,
) -> tuple[list[FrameManifestRecord], AlignmentReport]:
    """Deterministically join decoded frames with encode-index metadata.

    The join key is ``encoded_frame_id`` == ``decoded_frame_index`` (the
    presentation-order frame number). A frame whose id is not in the index — or
    whose id appears more than once in the index — is *unresolved*, never
    remapped. A metadata record with no matching frame is *unmatched
    metadata*. A non-monotonic capture sequence within the stream is a
    *discontinuity*. Missing segments are carried through from the route
    manifest. The result is fully deterministic for a given input (no RNG, no
    wall clock, no filesystem).
    """
    # Index the metadata side.
    by_id: dict[int, list[EncodeIndexRecord]] = {}
    for rec in encode_index:
        by_id.setdefault(rec.encoded_frame_id, []).append(rec)

    duplicates = [
        {"encoded_frame_id": fid, "count": len(recs)}
        for fid, recs in sorted(by_id.items())
        if len(recs) > 1
    ]

    matched = 0
    unmatched_frames: list[dict[str, Any]] = []
    unmatched_metadata: list[dict[str, Any]] = []
    classification_only = 0
    records: list[FrameManifestRecord] = []
    max_error: int | None = None

    matched_ids = set()
    for frame in video_frames:
        recs = by_id.get(frame.decoded_frame_index)
        if recs is None or len(recs) != 1:
            status = AlignmentStatus.UNRESOLVED
            cap = frame.capture_mono_ns
            ref = "unknown"
            matched_ids.add(frame.decoded_frame_index)
            if recs is None:
                unmatched_frames.append({"decoded_frame_index": frame.decoded_frame_index})
        else:
            rec = recs[0]
            matched += 1
            matched_ids.add(frame.decoded_frame_index)
            cap = rec.capture_mono_ns
            ref = rec.capture_reference
            if cap is None:
                status = AlignmentStatus.UNRESOLVED
            elif estimated_error_ns is not None and estimated_error_ns > 0:
                status = AlignmentStatus.BOUNDED_ESTIMATE
            else:
                status = AlignmentStatus.EXACT
        if status is AlignmentStatus.UNRESOLVED or cap is None:
            classification_only += 1
        records.append(
            FrameManifestRecord(
                schema_version=MANIFEST_SCHEMA_VERSION,
                route_id=route_id,
                session_id=session_id,
                segment_index=segment_index,
                stream=stream,
                video_relpath=video_relpath,
                video_sha256=video_sha256,
                decoded_frame_index=frame.decoded_frame_index,
                encoded_frame_id=frame.decoded_frame_index if recs is not None and len(recs) == 1 else None,
                capture_mono_ns=cap,
                capture_reference=ref,
                alignment_status=status.value,
                estimated_error_ns=estimated_error_ns if status is not AlignmentStatus.EXACT else None,
                width=frame.width,
                height=frame.height,
                pixel_format=frame.pixel_format,
                color_metadata=frame.color_metadata,
                calibration_id=None,
                route_group=route_group,
                site_group=site_group,
                encounter_id=encounter_id,
                frame_sha256=frame.frame_sha256,
            )
        )

    # Metadata that never matched a frame.
    for fid in sorted(by_id):
        if fid not in matched_ids:
            unmatched_metadata.append({"encoded_frame_id": fid})

    # Discontinuities: capture must be non-decreasing in presentation order.
    discontinuities: list[dict[str, Any]] = []
    prev_ts: int | None = None
    prev_idx: int | None = None
    for frame in sorted(video_frames, key=lambda f: f.decoded_frame_index):
        recs = by_id.get(frame.decoded_frame_index)
        if recs is None or len(recs) != 1 or recs[0].capture_mono_ns is None:
            prev_ts = None
            continue
        ts = recs[0].capture_mono_ns
        if prev_ts is not None and ts < prev_ts:
            discontinuities.append(
                {
                    "after_frame_index": prev_idx,
                    "before_frame_index": frame.decoded_frame_index,
                    "previous_capture_mono_ns": prev_ts,
                    "current_capture_mono_ns": ts,
                }
            )
        prev_ts = ts
        prev_idx = frame.decoded_frame_index

    # Estimated error bound: only meaningful if some frame is bounded.
    if max_error is None:
        max_error = estimated_error_ns if any(r.alignment_status == AlignmentStatus.BOUNDED_ESTIMATE.value for r in records) else None

    report = AlignmentReport(
        schema_version=MANIFEST_SCHEMA_VERSION,
        stream=stream,
        session_id=session_id,
        matched=matched,
        unmatched_frames=tuple(unmatched_frames),
        unmatched_metadata=tuple(unmatched_metadata),
        duplicate_frame_ids=tuple(duplicates),
        discontinuities=tuple(discontinuities),
        missing_segments=tuple(sorted(missing_segments)),
        classification_only_frames=classification_only,
        max_estimated_error_ns=max_error,
    )
    return records, report


def frames_to_jsonl(records: Sequence[FrameManifestRecord]) -> str:
    """Deterministic ``manifests/frames.jsonl``: one record per line, sorted by
    (segment_index, decoded_frame_index), keys sorted."""
    ordered = sorted(records, key=lambda r: (r.segment_index, r.decoded_frame_index if r.decoded_frame_index is not None else -1))
    lines = [json.dumps(r.to_dict(), sort_keys=True) for r in ordered]
    return "\n".join(lines) + "\n" if lines else ""


def frames_from_jsonl(text: str) -> list[FrameManifestRecord]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return [FrameManifestRecord.from_dict(json.loads(ln)) for ln in lines]
# [nnslr-t2] - END

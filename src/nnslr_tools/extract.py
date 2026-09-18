# [nnslr-t2] - START
"""Classification-only frame extraction from ordinary image files (T2).

This is the *media-side* half of T2 that does not require a video decoder, a
comma device or any third-party package: given ordinary image files that
already exist under the data root, it produces :class:`FrameManifestRecord`
objects with an **explicit non-timing status** (plan §T2: "support
classification-only extraction with an explicit non-timing status").

What it does *not* do, by design:

- It does **not** decode video. Real frame extraction from comma video needs
  the route/encoding adapter and an authorized decode path; that is a later,
  separately authorized step (plan §5.3, §T2). Until then, no real
  frame/log alignment is established.
- It does **not** invent timing. Every record it emits has
  ``alignment_status = "unresolved"``, ``capture_mono_ns = null`` and
  ``estimated_error_ns = null``. Such a frame is a legitimate *crop* source
  for classification but is excluded from any precise passage/latency claim
  (plan §5.2 last rule).
- It does **not** guess image dimensions. Width/height/pixel_format are
  explicit inputs (the caller measured them); absent values are an error, not
  a silent default (plan §5.2 requires them for reproducible preprocessing).

Path safety is shared with :mod:`nnslr_tools.manifest` (no remote URLs, no
absolute paths, no ``..`` traversal). The data root is always an explicit
argument so the module stays testable in any environment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from nnslr_tools.manifest import (
    FrameManifestRecord,
    MANIFEST_SCHEMA_VERSION,
    NnslerManifestError,
    normalize_relpath,
    sha256_of,
)


def extract_classification_frames(
    data_root: Path,
    image_paths: Sequence[Path],
    *,
    route_id: str,
    session_id: str,
    segment_index: int,
    stream: str,
    width: int,
    height: int,
    pixel_format: str,
    route_group: str = "unknown",
    site_group: str = "unknown",
    encounter_id: str = "unknown",
) -> list[FrameManifestRecord]:
    """Build classification-only frame records from ordinary image files.

    ``image_paths`` are resolved against ``data_root`` and must stay inside
    it. Each file must exist. The result is ordered by input order (the
    caller controls sampling; bounded sampling is the caller's decision, not a
    hidden behavior here). Every record is ``unresolved`` timing.

    Raises :class:`NnslerManifestError` for unsafe paths, missing files or
    non-positive dimensions — never a silent fallback.
    """
    if width <= 0 or height <= 0:
        raise NnslerManifestError("invalid_dimensions", f"{width}x{height}")
    if not pixel_format:
        raise NnslerManifestError("invalid_pixel_format", "")

    records: list[FrameManifestRecord] = []
    for path in image_paths:
        relpath = normalize_relpath(data_root, path)
        resolved = data_root / relpath
        if not resolved.is_file():
            raise NnslerManifestError("not_a_file", relpath)
        records.append(
            FrameManifestRecord(
                schema_version=MANIFEST_SCHEMA_VERSION,
                route_id=route_id,
                session_id=session_id,
                segment_index=segment_index,
                stream=stream,
                video_relpath=relpath,
                video_sha256=sha256_of(resolved),
                decoded_frame_index=None,
                encoded_frame_id=None,
                capture_mono_ns=None,
                capture_reference="unknown",
                alignment_status="unresolved",
                estimated_error_ns=None,
                width=width,
                height=height,
                pixel_format=pixel_format,
                color_metadata=None,
                calibration_id=None,
                route_group=route_group,
                site_group=site_group,
                encounter_id=encounter_id,
                frame_sha256=None,
            )
        )
    return records


def extract_report_dict(records: Sequence[FrameManifestRecord]) -> dict[str, object]:
    """Deterministic summary of an extraction batch (feeds the CLI/JSONL).

    ``timing_status`` is always ``"classification_only"`` here: the report must
    make it explicit that no timing claim is being made (plan §T2)."""
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "frame_count": len(records),
        "timing_status": "classification_only",
        "alignment_status": "unresolved",
        "note": (
            "classification-only extraction: frames are valid crop sources but "
            "carry no capture timing; they are excluded from passage/latency "
            "claims (plan §5.2/§T2). Real video extraction is a separately "
            "authorized step and is not implemented here."
        ),
    }
# [nnslr-t2] - END

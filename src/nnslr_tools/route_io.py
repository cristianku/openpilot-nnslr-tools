# [nnslr-t2] - START
"""Filesystem discovery for LOCAL comma route segments."""
from __future__ import annotations

from pathlib import Path

from nnslr_tools.manifest import (
    MANIFEST_SCHEMA_VERSION,
    FileKind,
    NnslerManifestError,
    RouteFile,
    RouteIdentity,
    RouteManifest,
    SegmentStatus,
    classify_filekind,
    parse_route_id,
    normalize_relpath,
    sha256_of,
)
from nnslr_tools.media import infer_camera_stream


def discover_segment_dirs(path: Path) -> list[tuple[RouteIdentity, Path]]:
    """Find segment directories without network/device access."""
    if not path.is_dir():
        raise NnslerManifestError("route_path_not_directory", str(path))

    found: list[tuple[RouteIdentity, Path]] = []

    # Layout A (loggerd/native copy): <root>/<route_id>--<segment>/
    candidates = [path, *sorted(p for p in path.iterdir() if p.is_dir())]
    for candidate in candidates:
        try:
            ident = RouteIdentity.from_segment_dir(candidate.name)
        except NnslerManifestError:
            continue
        found.append((ident, candidate))

    # Layout B (organized data root): <root>/<route_id>/<segment>/
    # where segment directories are numeric.
    try:
        parse_route_id(path.name)
        route_id = path.name
    except NnslerManifestError:
        route_id = None
    if route_id is not None:
        for candidate in sorted(p for p in path.iterdir() if p.is_dir() and p.name.isdigit()):
            found.append((RouteIdentity.from_route(route_id, int(candidate.name)), candidate))

    if not found:
        raise NnslerManifestError("no_segment_directories", str(path))

    route_ids = {ident.route_id for ident, _ in found}
    if len(route_ids) != 1:
        raise NnslerManifestError("mixed_routes", ",".join(sorted(route_ids)))
    return sorted(found, key=lambda pair: pair[0].segment_index)


def build_route_manifest(data_root: Path, route_path: Path) -> RouteManifest:
    root = data_root.resolve()
    route_abs = route_path if route_path.is_absolute() else root / route_path
    route_abs = route_abs.resolve()
    try:
        route_abs.relative_to(root)
    except ValueError as exc:
        raise NnslerManifestError("route_outside_data_root", str(route_abs)) from exc

    found = discover_segment_dirs(route_abs)
    indices = [ident.segment_index for ident, _ in found]
    declared = tuple(range(min(indices), max(indices) + 1))
    present = {ident.segment_index: (ident, directory) for ident, directory in found}

    files: list[RouteFile] = []
    statuses: dict[int, SegmentStatus] = {}

    for segment in declared:
        item = present.get(segment)
        if item is None:
            statuses[segment] = SegmentStatus.MISSING
            continue

        ident, directory = item
        segment_files = sorted(p for p in directory.iterdir() if p.is_file())
        has_video = False
        has_log = False
        for file_path in segment_files:
            kind = classify_filekind(file_path.name)
            if kind == FileKind.VIDEO:
                has_video = True
            elif kind in (FileKind.RLOG, FileKind.QLOG):
                has_log = True

            relpath = normalize_relpath(root, file_path.relative_to(root))
            stream = infer_camera_stream(file_path) if kind == FileKind.VIDEO else None
            if kind == FileKind.VIDEO and stream is None:
                stream = "unknown"
            files.append(
                RouteFile(
                    route=ident,
                    relpath=relpath,
                    kind=kind,
                    stream=stream,
                    sha256=sha256_of(file_path),
                    size_bytes=file_path.stat().st_size,
                )
            )

        statuses[segment] = SegmentStatus.COMPLETE if has_video and has_log else SegmentStatus.PARTIAL

    return RouteManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        data_root=".",
        declared_segments=declared,
        files=tuple(files),
        segment_status=statuses,
    )
# [nnslr-t2] - END

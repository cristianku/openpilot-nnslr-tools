from __future__ import annotations

from nnslr_tools.manifest import FileKind, RouteIdentity, SegmentStatus
from nnslr_tools.route_io import build_route_manifest


def test_route_manifest_preserves_missing_segment_and_qcamera_ts(tmp_path) -> None:
    root = tmp_path
    seg0 = root / "00000001--abcdef1234--0"
    seg2 = root / "00000001--abcdef1234--2"
    seg0.mkdir()
    seg2.mkdir()
    (seg0 / "rlog.zst").write_bytes(b"log0")
    (seg0 / "fcamera.hevc").write_bytes(b"video0")
    (seg2 / "qlog.zst").write_bytes(b"log2")
    (seg2 / "qcamera.ts").write_bytes(b"video2")

    manifest = build_route_manifest(root, root)
    assert manifest.declared_segments == (0, 1, 2)
    assert manifest.segment_status[0] == SegmentStatus.COMPLETE
    assert manifest.segment_status[1] == SegmentStatus.MISSING
    assert manifest.segment_status[2] == SegmentStatus.COMPLETE

    qcamera = next(f for f in manifest.files if f.relpath.endswith("qcamera.ts"))
    assert qcamera.kind == FileKind.VIDEO
    assert qcamera.stream == "q_narrow_road"


def test_current_loggerd_route_id_uses_hex_counter() -> None:
    route = RouteIdentity.from_segment_dir("000001a3--c20ba54385--7")
    assert route.route_counter == 0x1A3
    assert route.route_id == "000001a3--c20ba54385"
    assert route.segment_dir == "000001a3--c20ba54385--7"


def test_organized_route_layout_with_numeric_segment_dirs(tmp_path) -> None:
    route = tmp_path / "000001a3--c20ba54385"
    seg0 = route / "0"
    seg1 = route / "1"
    seg0.mkdir(parents=True)
    seg1.mkdir()
    (seg0 / "rlog.zst").write_bytes(b"log0")
    (seg0 / "fcamera.hevc").write_bytes(b"video0")
    (seg1 / "qlog.zst").write_bytes(b"log1")
    (seg1 / "fcamera.hevc").write_bytes(b"video1")

    manifest = build_route_manifest(tmp_path, route)
    assert manifest.declared_segments == (0, 1)
    assert all(status == SegmentStatus.COMPLETE for status in manifest.segment_status.values())

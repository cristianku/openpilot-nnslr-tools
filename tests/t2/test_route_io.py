from __future__ import annotations

from nnslr_tools.manifest import FileKind, SegmentStatus
from nnslr_tools.route_io import build_route_manifest


def test_route_manifest_preserves_missing_segment_and_qcamera_ts(tmp_path) -> None:
    root = tmp_path
    seg0 = root / "00000001--abcdef12--0"
    seg2 = root / "00000001--abcdef12--2"
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

# [nnslr-t1] - START
"""``nnslr`` command-line interface (T1).

Implemented subcommands (T1):

- ``version``       — print package/core versions.
- ``env``           — machine-readable environment report. Distinguishes
  *available*, *untested* and *failed* for GPU support; never runs a GPU
  workload implicitly (plan §12.4 ``check_environment`` contract).
- ``validate-batch``— validate an ObservationBatch JSON document against the
  §7 contract; deterministic, stable reason codes; nonzero exit on rejection.
- ``selftest``      — run the bundled synthetic fixtures through
  :func:`speed_vision_core.types.validate_batch` and the serialization
  round-trip; the CPU/synthetic quickstart for a clean clone.

Not implemented yet (owning tasks): dataset manifest/extraction (T2),
annotation validation and splits (T3), training (T4), evaluation (T5–T6),
export/bundle (T7–T8). They are listed here so a clean clone documents what
does not exist rather than reporting untested stubs as working.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

# Importing this module must not drag in torch/CUDA/openpilot (test-guarded).
import speed_vision_core
from speed_vision_core import types
from speed_vision_core.types import (
    MAX_DETECTIONS_PER_BATCH,
    ReasonCode,
    SUPPORTED_SCHEMA_VERSION,
    NnslerContractError,
)

from nnslr_tools import __version__
from nnslr_tools.alignment import align_comma_segment, comma_aligned_frame_from_dict
from nnslr_tools.candidates import find_speed_candidates, project_candidates_to_video
from nnslr_tools.comma_log import (
    CommaLogError,
    encode_events,
    map_speed_events,
    read_log_metadata,
)
from nnslr_tools.media import MediaToolError, extract_frames, make_clip, probe_frame_timestamps, probe_video
from nnslr_tools.manifest import NnslerManifestError, RouteManifest
from nnslr_tools.route_io import build_route_manifest
# [nnslr-sync] - START
from nnslr_tools.sync import sync_route
# [nnslr-sync] - END


# ---------------------------------------------------------------------------
# Environment report
# ---------------------------------------------------------------------------

def collect_env_report() -> dict[str, Any]:
    """Machine-readable environment facts. No GPU probe without an explicit
    flag; the report marks GPU as *untested* by default."""
    data_root = os.environ.get("NNSLR_DATA_ROOT", "").strip()
    data_root_path = Path(data_root) if data_root else None
    report: dict[str, Any] = {
        "nnslr_tools": __version__,
        "speed_vision_core": speed_vision_core.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "nnsler_data_root": {
            "configured": bool(data_root),
            "path": data_root or None,
            "exists": bool(data_root_path and data_root_path.is_dir()),
            "note": (
                "NNSLR_DATA_ROOT is read from the environment, never "
                "hardcoded (decision D4)."
                if data_root
                # [nnslr-sync] - START
                else "NNSLR_DATA_ROOT is not set; sync-routes defaults to "
                "/srv/nnslr-data. Other data-root commands need --data-root."
                # [nnslr-sync] - END
            ),
        },
        "cpu": {
            "available": True,
            "status": "available",
        },
        "gpu": {
            "status": "untested",
            "available": None,
            "note": (
                "GPU support is only established by an explicitly authorized "
                "forward/backward/export probe (plan §12.4 check_environment "
                "--gpu-smoke). This report does not probe."
            ),
        },
        "openpilot_dependent": False,
    }
    return report


# ---------------------------------------------------------------------------
# validate-batch
# ---------------------------------------------------------------------------

def _parse_frame_key(key: str) -> types.FrameIdentity:
    """Parse ``session:stream:frame_id`` into a :class:`FrameIdentity`.

    The session id must not contain ``:``; the stream is one of the
    :class:`StreamId` members; the frame id is the trailing integer."""
    session_id, stream, frame_id = key.split(":", 2)
    return types.FrameIdentity(session_id, types.StreamId(stream), int(frame_id))


def validate_batch_from_payload(
    payload: dict[str, Any],
    now_mono_ns: int,
    expected_session_id: str,
    recent_frame_keys: frozenset[str] | None = None,
) -> types.ValidationResult:
    """Deserialize a batch payload and run the §7.4 validation contract.

    ``recent_frame_keys`` are frame keys (``session:stream:frame_id``) of frames
    already accepted in the session; a batch matching one is rejected with
    ``DUPLICATE_FRAME``."""
    batch = types.batch_from_dict(payload)
    recent = (
        frozenset(_parse_frame_key(k) for k in recent_frame_keys)
        if recent_frame_keys is not None
        else None
    )
    return types.validate_batch(
        batch,
        now_mono_ns=now_mono_ns,
        expected_session_id=expected_session_id,
        recent_frame_ids=recent,
    )


def _cmd_validate_batch(args: argparse.Namespace) -> int:
    path = Path(args.batch)
    if not path.is_file():
        print(f"error: batch file not found: {path}", file=sys.stderr)
        return 2
    payload = json.loads(path.read_text(encoding="utf-8"))
    now = int(args.now_mono_ns)
    session = args.expected_session_id
    try:
        result = validate_batch_from_payload(payload, now_mono_ns=now, expected_session_id=session)
    except NnslerContractError as exc:
        print(
            json.dumps(
                {
                    "accepted": False,
                    "batch_identity": None,
                    "reason_codes": [exc.reason_code],
                    "rejected_indices": [],
                    "error": exc.message,
                },
                indent=2,
            )
        )
        return 1
    out = types.validation_result_to_dict(result)
    print(json.dumps(out, indent=2))
    return 0 if result.accepted else 1


# ---------------------------------------------------------------------------
# selftest (synthetic quickstart)
# ---------------------------------------------------------------------------

def _fixture_dir() -> Path:
    """Locate the synthetic fixture directory.

    Preferred order: a ``nnslr_tools/fixtures`` directory shipped with the
    installed package, then the repository checkout's ``tests/fixtures``
    (where the canonical fixtures live).
    """
    import importlib.resources as resources

    try:
        pkg_fixtures = resources.files("nnslr_tools").joinpath("fixtures")
        if pkg_fixtures.is_dir() and any(pkg_fixtures.iterdir()):
            return Path(str(pkg_fixtures))
    except FileNotFoundError:
        pass
    return Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures"


def _selftest() -> int:
    """Run the bundled synthetic fixtures end-to-end on CPU.

    Returns 0 only if every fixture produced its documented reason code and
    the serialization round-trip preserved absence/unknown.

    Time is injected, never read. Most fixtures use ``now = capture + 10 ms``;
    the fixtures that test the freshness/duplicate rules override ``now`` or
    provide ``recent_frame_keys`` explicitly (see ``_NOW_OVERRIDES`` and
    ``_RECENT_KEYS``).
    """
    # Most fixtures use a small positive offset (fresh capture). A few test the
    # freshness rules and need a specific offset relative to the capture:
    #   stale  -> now = capture + 5 s  (age 5 s > 0.35 s window)
    #   future -> now = capture - 1 s  (capture is in the future of "now")
    _NOW_OFFSETS_NS = {
        "malformed_stale_evidence.json": 5_000_000_000,
        "malformed_invalid_capture_timestamp_future.json": -1_000_000_000,
    }
    _DEFAULT_NOW_OFFSET_NS = 10_000_000  # +10 ms, well inside the 0.35 s window

    # Fixture whose rejection depends on the recent-frame set, not on the
    # payload alone: the frame identity is already present.
    _RECENT_KEYS = {
        "malformed_duplicate_frame.json": "selftest:narrow_road:2",
    }

    failures: list[str] = []
    passed: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        (passed if ok else failures).append(name + (f" — {detail}" if detail and not ok else ""))

    fixtures_dir = _fixture_dir()
    if not fixtures_dir.is_dir():
        print(f"selftest FAILED: fixture directory not found: {fixtures_dir}")
        return 1

    # 1) validate-batch on the valid fixture.
    with (fixtures_dir / "valid_batch.json").open() as f:
        payload = json.load(f)
    now = int(payload["frame"]["capture_mono_ns"]) + _DEFAULT_NOW_OFFSET_NS  # +10 ms
    result = validate_batch_from_payload(payload, now_mono_ns=now, expected_session_id="selftest")
    check("valid_batch accepted", result.accepted, str(result.reason_codes))

    # 2) every malformed fixture must be rejected with its documented code.
    malformed = sorted(p.name for p in fixtures_dir.iterdir() if p.name.startswith("malformed_"))
    for name in malformed:
        with (fixtures_dir / name).open() as f:
            payload = json.load(f)
        expected_code = name[len("malformed_") : -len(".json")]
        capture = int(payload["frame"]["capture_mono_ns"])
        offset = _NOW_OFFSETS_NS.get(name, _DEFAULT_NOW_OFFSET_NS)
        recent_keys = (
            frozenset({_RECENT_KEYS[name]}) if name in _RECENT_KEYS else None
        )
        try:
            result = validate_batch_from_payload(
                payload,
                now_mono_ns=capture + offset,
                expected_session_id="selftest",
                recent_frame_keys=recent_keys,
            )
        except NnslerContractError as exc:
            check(f"{name} rejected", exc.reason_code == expected_code, f"got {exc.reason_code}")
            continue
        codes = set(result.reason_codes)
        check(
            f"{name} rejected with {expected_code}",
            (not result.accepted) and expected_code in codes,
            f"accepted={result.accepted} codes={sorted(codes)}",
        )

    # 3) serialization round-trip preserves absence and never reinterprets zero.
    with (fixtures_dir / "valid_batch.json").open() as f:
        payload = json.load(f)
    batch = types.batch_from_dict(payload)
    again = types.batch_to_dict(batch)
    check(
        "round-trip stable",
        json.dumps(again, sort_keys=True) == json.dumps(payload, sort_keys=True),
    )
    zero_payload = json.loads(json.dumps(payload))
    det = zero_payload["detections"][0]
    det["value_state"] = types.ValueState.UNKNOWN.value
    det["value_kph"] = 0  # zero with has_value=false must be rejected
    try:
        types.batch_from_dict(zero_payload)
        check("zero-not-limit rejected", False, "construction accepted zero as value")
    except NnslerContractError as exc:
        check(
            "zero-not-limit rejected",
            exc.reason_code == ReasonCode.INVALID_VALUE_CONSISTENCY,
            exc.reason_code,
        )

    print(f"nnslr selftest: nnslr_tools {__version__}")
    for line in passed:
        print(f"  PASS  {line}")
    for line in failures:
        print(f"  FAIL  {line}")
    if failures:
        print(f"selftest FAILED: {len(failures)} of {len(passed) + len(failures)}")
        return 1
    print(f"selftest OK: {len(passed)} checks")
    return 0


# ---------------------------------------------------------------------------
# T2 local comma file processing (never device access)
# ---------------------------------------------------------------------------

_STREAM_TO_ENCODE_SERVICE = {
    "narrow_road": "narrowRoadEncodeIdx",
    "wide_road": "wideRoadEncodeIdx",
    "q_narrow_road": "qNarrowRoadEncodeIdx",
}


def _write_jsonl(path: Path | None, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    if path is None:
        sys.stdout.write(text)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        rows.append(item)
    return rows


def _resolve_data_root(explicit: str | None) -> Path:
    raw = (explicit or os.environ.get("NNSLR_DATA_ROOT", "")).strip()
    if not raw:
        raise ValueError("NNSLR_DATA_ROOT is not set; use --data-root or export it")
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"data root is not a directory: {root}")
    return root


def _cmd_route_manifest(args: argparse.Namespace) -> int:
    root = _resolve_data_root(args.data_root)
    manifest = build_route_manifest(root, Path(args.route))
    text = manifest.to_jsonl()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "declared_segments": list(manifest.declared_segments),
        **manifest.gap_report(),
        "file_count": len(manifest.files),
    }, indent=2, sort_keys=True))
    return 0


def _cmd_inspect_manifest(args: argparse.Namespace) -> int:
    path = Path(args.manifest)
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = RouteManifest.from_jsonl(path.read_text(encoding="utf-8"))
    print(json.dumps({
        "schema_version": manifest.schema_version,
        "declared_segments": list(manifest.declared_segments),
        "file_count": len(manifest.files),
        **manifest.gap_report(),
    }, indent=2, sort_keys=True))
    return 0


def _cmd_video_probe(args: argparse.Namespace) -> int:
    result = probe_video(Path(args.video), ffprobe=args.ffprobe)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _cmd_extract_frames(args: argparse.Namespace) -> int:
    frames = extract_frames(
        Path(args.video),
        Path(args.output),
        fps=None if args.all_frames else args.fps,
        start_s=args.start,
        end_s=args.end,
        ffmpeg=args.ffmpeg,
        overwrite=args.overwrite,
    )
    rows = [frame.to_dict() for frame in frames]
    if args.manifest:
        _write_jsonl(Path(args.manifest), rows)
    print(json.dumps({
        "frame_count": len(rows),
        "output": str(Path(args.output)),
        "manifest": args.manifest,
        "timing": "media_only",
        "capture_time_established": False,
    }, indent=2, sort_keys=True))
    return 0


def _cmd_log_metadata(args: argparse.Namespace) -> int:
    events = read_log_metadata(
        Path(args.log),
        openpilot_root=Path(args.openpilot_root) if args.openpilot_root else None,
        python_executable=args.openpilot_python,
    )
    rows = [event.to_dict() for event in events]
    _write_jsonl(Path(args.output) if args.output else None, rows)
    return 0


def _cmd_align_route(args: argparse.Namespace) -> int:
    stream = args.stream
    try:
        service = _STREAM_TO_ENCODE_SERVICE[stream]
    except KeyError as exc:
        raise ValueError(f"unsupported stream {stream!r}") from exc

    frames = probe_frame_timestamps(Path(args.video), ffprobe=args.ffprobe)
    events = read_log_metadata(
        Path(args.log),
        openpilot_root=Path(args.openpilot_root) if args.openpilot_root else None,
        python_executable=args.openpilot_python,
    )
    indexes = encode_events(events, service=service, segment_num=args.segment_num)
    aligned, report = align_comma_segment(frames, indexes, segment_num=args.segment_num)

    rows: list[dict[str, Any]] = [{
        "kind": "alignment_report",
        "stream": stream,
        "video": str(Path(args.video)),
        "log": str(Path(args.log)),
        "segment_num": args.segment_num,
        **report.to_dict(),
    }]
    rows.extend({"kind": "frame", **frame.to_dict()} for frame in aligned)
    _write_jsonl(Path(args.output), rows)

    print(json.dumps({
        "output": str(Path(args.output)),
        "stream": stream,
        "segment_num": args.segment_num,
        **report.to_dict(),
    }, indent=2, sort_keys=True))
    return 0 if report.unresolved == 0 else 1


def _cmd_alignment_report(args: argparse.Namespace) -> int:
    rows = _read_jsonl(Path(args.alignment))
    header = next((row for row in rows if row.get("kind") == "alignment_report"), None)
    if header is None:
        raise ValueError("alignment JSONL has no alignment_report record")
    print(json.dumps(header, indent=2, sort_keys=True))
    return 0


def _cmd_find_candidates(args: argparse.Namespace) -> int:
    events = read_log_metadata(
        Path(args.log),
        openpilot_root=Path(args.openpilot_root) if args.openpilot_root else None,
        python_executable=args.openpilot_python,
    )
    candidates = find_speed_candidates(map_speed_events(events))

    if args.alignment:
        rows = _read_jsonl(Path(args.alignment))
        aligned = [
            comma_aligned_frame_from_dict(row)
            for row in rows
            if row.get("kind") == "frame"
        ]
        candidates = project_candidates_to_video(
            candidates,
            aligned,
            max_error_ns=args.max_projection_error_ns,
        )

    _write_jsonl(
        Path(args.output) if args.output else None,
        [{"kind": "candidate", **candidate.to_dict()} for candidate in candidates],
    )
    return 0


def _cmd_make_clips(args: argparse.Namespace) -> int:
    rows = _read_jsonl(Path(args.candidates))
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    made: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        if row.get("kind") != "candidate":
            continue
        center = row.get("video_time_s")
        if not isinstance(center, (int, float)) or isinstance(center, bool):
            skipped += 1
            continue
        start = max(0.0, float(center) - args.before)
        end = float(center) + args.after
        candidate_id = str(row.get("candidate_id", f"candidate-{len(made)}"))
        safe_id = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in candidate_id)
        out = output_dir / f"{safe_id}.mp4"
        make_clip(
            Path(args.video),
            out,
            start_s=start,
            end_s=end,
            ffmpeg=args.ffmpeg,
            reencode=args.reencode,
        )
        made.append({
            "kind": "clip",
            "candidate_id": candidate_id,
            "clip_path": str(out),
            "start_s": start,
            "end_s": end,
            "source_video": str(Path(args.video)),
            "candidate_reason": row.get("candidate_reason"),
            "source_log_mono_time": row.get("log_mono_time"),
        })

    if args.manifest:
        _write_jsonl(Path(args.manifest), made)
    print(json.dumps({
        "clips_created": len(made),
        "candidates_without_video_time": skipped,
        "output": str(output_dir),
        "manifest": args.manifest,
    }, indent=2, sort_keys=True))
    return 0


# ---------------------------------------------------------------------------
# [nnslr-sync] - START
def _cmd_sync_routes(args: argparse.Namespace) -> int:
    try:
        report = sync_route(
            route=args.route, host=args.host, segments=args.segments,
            camera=args.camera, data_root=args.data_root, dry_run=args.dry_run,
        )
    except OSError as exc:
        print(f"error: sync-routes: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1
# [nnslr-sync] - END


# Not-yet-implemented subcommands (documented, not stubbed)
# ---------------------------------------------------------------------------

_NOT_IMPLEMENTED: dict[str, tuple[str, str]] = {
    "check-environment": ("T1", "full report is implemented via `nnslr env`"),
    "import-annotations": ("T3", "annotation validator lands in T3"),
    "validate-dataset": ("T3", "annotation validator lands in T3"),
    "build-splits": ("T3", "leakage-resistant split builder lands in T3"),
    "train": ("T4", "V100 training requires separate compute authorization"),
    "evaluate": ("T5", "offline evaluation lands in T5"),
    "mine-hard-examples": ("T6", "hard-example mining lands in T6"),
    "export-onnx": ("T7", "ONNX export lands in T7"),
    "replay": ("T7", "annotated replay lands in T7"),
    "package-model": ("T8", "bundle packaging lands in T8"),
    "verify-bundle": ("T8", "bundle verification lands in T8"),
    "export-core": ("T8", "core snapshot export lands in T8"),
}


def _not_implemented(name: str, _args: argparse.Namespace) -> int:
    task, note = _NOT_IMPLEMENTED[name]
    print(
        f"error: `nnslr {name}` is not implemented yet (planned in {task}). {note}.",
        file=sys.stderr,
    )
    print("See docs/plan.md §12.4 for the full script contract.", file=sys.stderr)
    return 3


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nnslr",
        description=(
            "NNSLR — NN Vision Speed Limit tooling. Advisory-only project: "
            "never part of the vehicle control path. T1 ships the domain "
            "contracts, a validation command and the synthetic quickstart."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ver = sub.add_parser("version", help="print package versions")
    p_ver.set_defaults(func=lambda a: _cmd_version(a))

    p_env = sub.add_parser(
        "env", help="machine-readable environment report (no implicit GPU probe)"
    )
    p_env.set_defaults(func=lambda a: _cmd_env(a))

    p_val = sub.add_parser(
        "validate-batch",
        help="validate an ObservationBatch JSON file against the §7 contract",
    )
    p_val.add_argument("batch", help="path to the batch JSON file")
    p_val.add_argument(
        "--now-mono-ns",
        type=int,
        required=True,
        help="current session-monotonic time in ns (injected, never read live)",
    )
    p_val.add_argument(
        "--expected-session-id",
        required=True,
        help="session the batch must belong to",
    )
    p_val.set_defaults(func=_cmd_validate_batch)

    p_self = sub.add_parser("selftest", help="run the bundled synthetic fixtures (CPU quickstart)")
    p_self.set_defaults(func=lambda a: _selftest())

    # [nnslr-sync] - START
    p_sync = sub.add_parser("sync-routes", help="copy one comma route via SSH/rsync (front camera + full log)")
    p_sync.add_argument("--route", required=True, help="route ID to copy (always explicit)")
    p_sync.add_argument("--host", help="SSH alias, hostname, IPv4 or user@host (default: NNSLR_COMMA_HOST or comma-remote)")
    p_sync.add_argument("--segments", default="all", help="all (default), 0-3, or 0,2,4-6")
    p_sync.add_argument("--camera", choices=("narrow", "wide", "both"), default="narrow",
                        help="narrow=fcamera (default), wide=ecamera, both=both road cameras")
    p_sync.add_argument("--data-root", help="destination root (default: NNSLR_DATA_ROOT or /srv/nnslr-data)")
    p_sync.add_argument("--dry-run", action="store_true", help="inventory via SSH and print the plan without copying or writing files")
    p_sync.set_defaults(func=_cmd_sync_routes)
    # [nnslr-sync] - END

    p_route = sub.add_parser("route-manifest", help="inventory LOCAL route segments and preserve gaps")
    p_route.add_argument("route", help="route directory under the data root, or one local segment directory")
    p_route.add_argument("--data-root")
    p_route.add_argument("--output", required=True)
    p_route.set_defaults(func=_cmd_route_manifest)

    p_inspect = sub.add_parser("inspect-manifest", help="summarize a route manifest JSONL")
    p_inspect.add_argument("manifest")
    p_inspect.set_defaults(func=_cmd_inspect_manifest)

    p_probe = sub.add_parser("video-probe", help="probe a LOCAL camera video with ffprobe")
    p_probe.add_argument("video")
    p_probe.add_argument("--ffprobe")
    p_probe.set_defaults(func=_cmd_video_probe)

    p_extract = sub.add_parser("extract-frames", help="extract LOCAL video frames with ffmpeg")
    p_extract.add_argument("--video", required=True)
    p_extract.add_argument("--output", required=True)
    p_extract.add_argument("--fps", type=float, default=1.0, help="sampling FPS (default: 1 for broad discovery)")
    p_extract.add_argument("--all-frames", action="store_true", help="decode every frame instead of sampling")
    p_extract.add_argument("--start", type=float)
    p_extract.add_argument("--end", type=float)
    p_extract.add_argument("--ffmpeg")
    p_extract.add_argument("--manifest")
    p_extract.add_argument("--overwrite", action="store_true")
    p_extract.set_defaults(func=_cmd_extract_frames)

    p_log = sub.add_parser("log-metadata", help="read LOCAL qlog/rlog camera/map metadata")
    p_log.add_argument("log")
    p_log.add_argument("--openpilot-root")
    p_log.add_argument("--openpilot-python")
    p_log.add_argument("--output")
    p_log.set_defaults(func=_cmd_log_metadata)

    p_align = sub.add_parser("align-route", help="align LOCAL video presentation order to EncodeIndex")
    p_align.add_argument("--video", required=True)
    p_align.add_argument("--log", required=True)
    p_align.add_argument("--stream", required=True, choices=sorted(_STREAM_TO_ENCODE_SERVICE))
    p_align.add_argument("--segment-num", required=True, type=int)
    p_align.add_argument("--output", required=True)
    p_align.add_argument("--openpilot-root")
    p_align.add_argument("--openpilot-python")
    p_align.add_argument("--ffprobe")
    p_align.set_defaults(func=_cmd_align_route)

    p_report = sub.add_parser("alignment-report", help="print the report header from alignment JSONL")
    p_report.add_argument("alignment")
    p_report.set_defaults(func=_cmd_alignment_report)

    p_candidates = sub.add_parser("find-candidates", help="find map transitions as search hints, never ground truth")
    p_candidates.add_argument("log")
    p_candidates.add_argument("--openpilot-root")
    p_candidates.add_argument("--openpilot-python")
    p_candidates.add_argument("--alignment")
    p_candidates.add_argument("--output")
    p_candidates.add_argument("--max-projection-error-ns", type=int, default=250_000_000)
    p_candidates.set_defaults(func=_cmd_find_candidates)

    p_clips = sub.add_parser("make-clips", help="make LOCAL review clips around projected candidates")
    p_clips.add_argument("--video", required=True)
    p_clips.add_argument("--candidates", required=True)
    p_clips.add_argument("--output", required=True)
    p_clips.add_argument("--manifest")
    p_clips.add_argument("--before", type=float, default=30.0)
    p_clips.add_argument("--after", type=float, default=30.0)
    p_clips.add_argument("--ffmpeg")
    p_clips.add_argument("--reencode", action="store_true")
    p_clips.set_defaults(func=_cmd_make_clips)

    for name in _NOT_IMPLEMENTED:
        p = sub.add_parser(name, help=f"NOT IMPLEMENTED YET (planned in {_NOT_IMPLEMENTED[name][0]})")
        p.set_defaults(func=lambda a, _n=name: _not_implemented(_n, a))

    return parser


def _cmd_version(args: argparse.Namespace) -> int:
    print(f"nnslr_tools {__version__}")
    print("speed_vision_core 0.1.0")
    print(f"python {platform.python_version()}")
    print(f"schema_version {SUPPORTED_SCHEMA_VERSION}")
    print(f"max_detections_per_batch {MAX_DETECTIONS_PER_BATCH}")
    return 0


def _cmd_env(args: argparse.Namespace) -> int:
    print(json.dumps(collect_env_report(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except NnslerContractError as exc:
        print(f"error: {exc.reason_code}: {exc.message}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON: {exc}", file=sys.stderr)
        return 2
    except (MediaToolError, CommaLogError, NnslerManifestError) as exc:
        detail = getattr(exc, "detail", "")
        stderr = getattr(exc, "stderr", "")
        print(f"error: {exc}", file=sys.stderr)
        if detail and detail not in str(exc):
            print(detail, file=sys.stderr)
        if stderr:
            print(stderr, file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
# [nnslr-t1] - END

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
                else "NNSLR_DATA_ROOT is not set; commands that need a data "
                "root must refuse to run (plan §12.4)."
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
# Not-yet-implemented subcommands (documented, not stubbed)
# ---------------------------------------------------------------------------

_NOT_IMPLEMENTED: dict[str, tuple[str, str]] = {
    "check-environment": ("T1", "full report is implemented via `nnslr env`"),
    "sync-routes": ("T2", "remote access requires explicit authorization per drive"),
    "extract-frames": ("T2", "needs the route/alignment contract from T2"),
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


if __name__ == "__main__":
    raise SystemExit(main())
# [nnslr-t1] - END

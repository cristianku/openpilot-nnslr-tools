# [nnslr-t1] - START
"""CLI entry-point tests for the ``nnslr`` command (T1).

Exercises the real ``main()`` entry point (the same code the ``nnslr`` console
script runs) in-process: help, version, env report, validate-batch accepted /
rejected / missing-file exit codes, and the documented not-yet-implemented
subcommands. No model, no GPU, no openpilot, no device.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nnslr_tools import cli

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "src" / "nnslr_tools" / "fixtures"
VALID = FIXTURES / "valid_batch.json"
VALID_CAPTURE_NS = 1_000_000_000_000
SESSION = "selftest"


def run(argv: list[str], capsys) -> tuple[int, str, str]:
    try:
        code = cli.main(argv)
    except SystemExit as exc:  # argparse --help / --error path
        code = exc.code if isinstance(exc.code, int) else 0
    captured = capsys.readouterr()
    return int(code), captured.out, captured.err


# ---------------------------------------------------------------------------
# Help / version / env
# ---------------------------------------------------------------------------

def test_help_exits_zero_and_prints_usage(capsys) -> None:
    code, out, _ = run(["--help"], capsys)
    assert code == 0
    assert "usage:" in out.lower()
    assert "validate-batch" in out
    assert "selftest" in out
    assert "video-probe" in out
    assert "extract-frames" in out
    assert "align-route" in out


def test_version_exits_zero_and_reports_schema(capsys) -> None:
    code, out, _ = run(["version"], capsys)
    assert code == 0
    assert "nnslr_tools" in out
    assert "speed_vision_core" in out
    assert "schema_version 1" in out


def test_env_is_json_and_does_not_probe_gpu(capsys, monkeypatch) -> None:
    monkeypatch.delenv("NNSLR_DATA_ROOT", raising=False)
    code, out, _ = run(["env"], capsys)
    assert code == 0
    report = json.loads(out)
    assert report["openpilot_dependent"] is False
    assert report["gpu"]["status"] == "untested"
    assert report["gpu"]["available"] is None
    assert report["nnsler_data_root"]["configured"] is False
    assert report["cpu"]["available"] is True


def test_env_reports_configured_data_root(capsys, monkeypatch) -> None:
    monkeypatch.setenv("NNSLR_DATA_ROOT", "/path/to/speed-vision-data")
    code, out, _ = run(["env"], capsys)
    assert code == 0
    report = json.loads(out)
    assert report["nnsler_data_root"]["configured"] is True
    assert report["nnsler_data_root"]["path"] == "/path/to/speed-vision-data"


# ---------------------------------------------------------------------------
# validate-batch
# ---------------------------------------------------------------------------

def test_validate_batch_accepts_valid_fixture(capsys) -> None:
    now = VALID_CAPTURE_NS + 10_000_000
    code, out, _ = run(
        ["validate-batch", str(VALID),
         "--now-mono-ns", str(now), "--expected-session-id", SESSION],
        capsys,
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["accepted"] is True
    assert payload["batch_identity"] == f"{SESSION}:narrow_road:2"


def test_validate_batch_rejects_bad_value(capsys) -> None:
    fixture = FIXTURES / "malformed_invalid_value_zero.json"
    now = VALID_CAPTURE_NS + 10_000_000
    code, out, _ = run(
        ["validate-batch", str(fixture),
         "--now-mono-ns", str(now), "--expected-session-id", SESSION],
        capsys,
    )
    assert code == 1
    payload = json.loads(out)
    assert payload["accepted"] is False
    assert "invalid_value_zero" in payload["reason_codes"]


def test_validate_batch_rejects_value_without_state(capsys) -> None:
    # Rejected at construction (hard contract error) -> exit 1 with a code.
    fixture = FIXTURES / "malformed_invalid_value_consistency.json"
    now = VALID_CAPTURE_NS + 10_000_000
    code, out, _ = run(
        ["validate-batch", str(fixture),
         "--now-mono-ns", str(now), "--expected-session-id", SESSION],
        capsys,
    )
    assert code == 1
    payload = json.loads(out)
    assert payload["accepted"] is False
    assert payload["reason_codes"] == ["invalid_value_consistency"]


def test_validate_batch_missing_file_exits_two(capsys) -> None:
    now = VALID_CAPTURE_NS + 10_000_000
    code, _, err = run(
        ["validate-batch", str(FIXTURES / "does_not_exist.json"),
         "--now-mono-ns", str(now), "--expected-session-id", SESSION],
        capsys,
    )
    assert code == 2
    assert "not found" in err


def test_validate_batch_rejects_wrong_numeric_type(capsys, tmp_path) -> None:
    # A string frame_id must be rejected as a type error (no coercion),
    # reported with the stable invalid_numeric_type code and exit 1.
    payload = json.loads(VALID.read_text(encoding="utf-8"))
    payload["frame"]["frame_id"] = "2"
    bad = tmp_path / "bad_frame_id.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    code, out, _ = run(
        ["validate-batch", str(bad),
         "--now-mono-ns", str(VALID_CAPTURE_NS + 10_000_000),
         "--expected-session-id", SESSION],
        capsys,
    )
    assert code == 1
    result = json.loads(out)
    assert result["accepted"] is False
    assert result["reason_codes"] == ["invalid_numeric_type"]


# ---------------------------------------------------------------------------
# Selftest quickstart
# ---------------------------------------------------------------------------

def test_selftest_passes(capsys) -> None:
    code, out, _ = run(["selftest"], capsys)
    assert code == 0
    assert "selftest OK" in out


# ---------------------------------------------------------------------------
# Not-yet-implemented subcommands are documented, not silently stubbed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name",
    [
        "package-model", "verify-bundle", "export-core",
    ],
)
def test_not_implemented_subcommands_exit_three(capsys, name) -> None:
    code, _, err = run([name], capsys)
    assert code == 3
    assert "not implemented" in err.lower()


# [training-baseline] - START
def test_train_is_implemented_and_requires_dataset_inputs(capsys, tmp_path) -> None:
    code, _, err = run(["train", "--dry-run", "--data-root", str(tmp_path)], capsys)
    assert code == 2
    assert "objects.jsonl" in err
# [training-baseline] - END

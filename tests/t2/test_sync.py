# [nnslr-sync] - START
"""Exercise the CLI and real rsync protocol against local files, never a device."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

from nnslr_tools import cli

ROUTE = "000001a3--c20ba54385"


@pytest.fixture
def source(tmp_path, monkeypatch):
    if shutil.which("rsync") is None:
        pytest.skip("local transfer integration tests require rsync")
    remote = tmp_path / "remote"
    remote.mkdir()
    commands = tmp_path / "ssh-calls"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Only SSH transport is replaced. The remote shell, file inventory and
    # both ends of rsync run for real on the test fixture's filesystem.
    ssh = bin_dir / "ssh"
    ssh.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "args = sys.argv[1:]\n"
        "while args and args[0].startswith('-'):\n"
        "    option = args.pop(0)\n"
        "    if option in ('-o', '-l', '-p', '-i'): args.pop(0)\n"
        "args.pop(0)  # host\n"
        "command = ' '.join(args)\n"
        "with open(os.environ['TEST_SSH_CALLS'], 'a') as f: f.write(command + '\\n')\n"
        "if os.environ.get('TEST_SSH_FAIL'): sys.exit(255)\n"
        "if os.environ.get('TEST_TRANSFER_FAIL') and '--server' in command: sys.exit(12)\n"
        "command = command.replace('/data/media/0/realdata', os.environ['TEST_REMOTE'])\n"
        "os.execv('/bin/sh', ['sh', '-c', command])\n",
        encoding="utf-8",
    )
    ssh.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("TEST_REMOTE", str(remote))
    monkeypatch.setenv("TEST_SSH_CALLS", str(commands))
    monkeypatch.setenv("NNSLR_DATA_ROOT", str(tmp_path / "data with spaces"))
    monkeypatch.delenv("NNSLR_COMMA_HOST", raising=False)
    return remote, Path(os.environ["NNSLR_DATA_ROOT"]), commands


def segment(remote: Path, index: int, *, route: str = ROUTE) -> Path:
    directory = remote / f"{route}--{index}"
    directory.mkdir()
    for name in ("fcamera.hevc", "ecamera.hevc", "dcamera.hevc", "qcamera.ts", "rlog.zst", "qlog.zst"):
        (directory / name).write_bytes(f"{index}:{name}".encode())
    return directory


def run(capfd, *args):
    try:
        code = cli.main(["sync-routes", "--route", ROUTE, *args])
    except SystemExit as exc:
        code = exc.code
    captured = capfd.readouterr()
    return code, captured.out, captured.err


def test_defaults_copy_only_front_and_full_log_for_selected_route(source, capfd):
    remote, root, _ = source
    first = segment(remote, 0)
    segment(remote, 2)
    segment(remote, 1, route="000001a4--abcdef1234")
    code, out, err = run(capfd)
    assert code == 0, err
    report = json.loads(out)
    assert report["host"] == "comma"
    assert report["segments"] == [0, 2]
    assert report["ok"] is True
    dest = root / "raw" / "routes" / ROUTE
    assert sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()) == [
        "0/fcamera.hevc", "0/rlog.zst", "2/fcamera.hevc", "2/rlog.zst",
    ]
    assert (dest / "0/fcamera.hevc").read_bytes() == (first / "fcamera.hevc").read_bytes()
    assert report["files"][0]["sha256"] == hashlib.sha256(b"0:fcamera.hevc").hexdigest()
    assert (first / "dcamera.hevc").is_file()  # source is untouched


def test_dry_run_uses_fallback_defaults_without_creating_destination(source, capfd, monkeypatch):
    remote, _, _ = source
    segment(remote, 0)
    monkeypatch.delenv("NNSLR_DATA_ROOT")
    code, out, err = run(capfd, "--dry-run")
    assert code == 0, err
    report = json.loads(out)
    assert report["data_root"] == "/srv/nnslr-data"
    assert report["dry_run"] is True
    assert all(row["status"] == "planned" and row["sha256"] is None for row in report["files"])


def test_overrides_select_segments_camera_host_and_destination(source, tmp_path, capfd):
    remote, default_root, _ = source
    for index in range(4):
        segment(remote, index)
    explicit = tmp_path / "explicit"
    code, out, err = run(capfd, "--host", "comma@192.0.2.10", "--segments", "1-2,2",
                         "--camera", "both", "--data-root", str(explicit))
    assert code == 0, err
    report = json.loads(out)
    assert report["host"] == "comma@192.0.2.10"
    assert report["segments"] == [1, 2]
    assert not default_root.exists()
    assert sorted(p.name for p in (explicit / "raw/routes" / ROUTE / "1").iterdir()) == [
        "ecamera.hevc", "fcamera.hevc", "rlog.zst",
    ]


def test_dry_run_does_not_create_files(source, capfd):
    remote, root, _ = source
    segment(remote, 0)
    code, out, err = run(capfd, "--dry-run", "--camera", "wide")
    assert code == 0, err
    assert not root.exists()
    assert [row["name"] for row in json.loads(out)["files"]] == ["ecamera.hevc", "rlog.zst"]


def test_missing_video_full_log_and_requested_segment_are_reported(source, capfd):
    remote, _, _ = source
    directory = segment(remote, 0)
    (directory / "rlog.zst").unlink()
    (directory / "fcamera.hevc").unlink()
    code, out, _ = run(capfd, "--segments", "0-1", "--dry-run")
    assert code == 1
    report = json.loads(out)
    assert report["ok"] is False
    assert report["missing_segments"] == [1]
    assert {(row["segment"], row["name"]) for row in report["missing_files"]} == {(0, "fcamera.hevc"), (0, "rlog")}
    assert not report["files"]  # qlog must not silently substitute for rlog


@pytest.mark.parametrize("name", ["rlog", "rlog.bz2"])
def test_full_log_variants(source, capfd, name):
    remote, root, _ = source
    directory = segment(remote, 0)
    (directory / "rlog.zst").rename(directory / name)
    code, _, err = run(capfd)
    assert code == 0, err
    assert (root / "raw/routes" / ROUTE / "0" / name).read_bytes() == b"0:rlog.zst"


def test_rerun_reuses_identical_files_and_preserves_unrelated_files(source, capfd):
    remote, root, _ = source
    segment(remote, 0)
    assert run(capfd)[0] == 0
    dest = root / "raw/routes" / ROUTE / "0"
    (dest / "notes.txt").write_text("keep")
    before = (dest / "fcamera.hevc").stat().st_mtime_ns
    code, out, err = run(capfd)
    assert code == 0, err
    assert all(row["status"] == "existing" for row in json.loads(out)["files"])
    assert (dest / "fcamera.hevc").stat().st_mtime_ns == before
    assert (dest / "notes.txt").read_text() == "keep"


def test_existing_different_content_is_not_overwritten_even_if_same_size(source, capfd):
    remote, root, _ = source
    segment(remote, 0)
    dest = root / "raw/routes" / ROUTE / "0"
    dest.mkdir(parents=True)
    original = b"x" * len(b"0:fcamera.hevc")
    (dest / "fcamera.hevc").write_bytes(original)
    code, out, err = run(capfd)
    assert code != 0
    assert "conflict" in out + err
    assert (dest / "fcamera.hevc").read_bytes() == original


def test_resume_from_rsync_partial_directory(source, capfd):
    remote, root, _ = source
    directory = segment(remote, 0)
    payload = bytes(range(256)) * 4096
    (directory / "fcamera.hevc").write_bytes(payload)
    partial = root / "raw/routes" / ROUTE / "0/.nnslr-partial"
    partial.mkdir(parents=True)
    (partial / "fcamera.hevc").write_bytes(payload[:len(payload) // 2])
    code, out, err = run(capfd)
    assert code == 0, err
    assert (partial.parent / "fcamera.hevc").read_bytes() == payload
    frame = next(row for row in json.loads(out)["files"] if row["name"] == "fcamera.hevc")
    assert frame["status"] == "resumed"
    assert frame["sha256"] == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("args", [
    ["--host", "-oProxyCommand=bad"], ["--host", "user@host;touch bad"],
    ["--route", "../other"], ["--segments", "3-1"], ["--segments", "-1"],
    ["--segments", ""], ["--segments", "1,,2"],
])
def test_invalid_input_is_rejected_before_ssh(source, capfd, args):
    _, root, commands = source
    code, _, _ = run(capfd, *args)
    assert code != 0
    assert not commands.exists()
    assert not root.exists()


def test_destination_symlink_escape_is_rejected(source, tmp_path, capfd):
    remote, root, _ = source
    segment(remote, 0)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "raw/routes").mkdir(parents=True)
    (root / "raw/routes" / ROUTE).symlink_to(outside, target_is_directory=True)
    code, _, _ = run(capfd)
    assert code != 0
    assert list(outside.iterdir()) == []


def test_ssh_failure_is_actionable_and_not_success(source, capfd, monkeypatch):
    monkeypatch.setenv("TEST_SSH_FAIL", "1")
    code, out, err = run(capfd)
    assert code != 0
    assert "ssh" in (out + err).lower()
    assert "Traceback" not in err


def test_host_preference_can_be_saved_and_explicit_host_wins(source, capfd, monkeypatch):
    remote, _, _ = source
    segment(remote, 0)
    monkeypatch.setenv("NNSLR_COMMA_HOST", "my-comma")
    code, out, err = run(capfd, "--dry-run")
    assert code == 0, err
    assert json.loads(out)["host"] == "my-comma"
    code, out, err = run(capfd, "--dry-run", "--host", "comma@192.0.2.20")
    assert code == 0, err
    assert json.loads(out)["host"] == "comma@192.0.2.20"


def test_failed_transfer_reports_pending_files_and_rerun_succeeds(source, capfd, monkeypatch):
    remote, root, _ = source
    segment(remote, 0)
    monkeypatch.setenv("TEST_TRANSFER_FAIL", "1")
    code, out, _ = run(capfd)
    assert code == 1
    report = json.loads(out)
    assert report["ok"] is False
    assert [row["status"] for row in report["files"]] == ["failed", "planned"]
    assert "rsync_transfer_failed" in report["files"][0]["error"]
    assert not (root / "raw/routes" / ROUTE / "0/fcamera.hevc").exists()
    monkeypatch.delenv("TEST_TRANSFER_FAIL")
    code, out, err = run(capfd)
    assert code == 0, err
    assert json.loads(out)["ok"] is True


def test_partial_directory_symlink_cannot_redirect_resume(source, tmp_path, capfd):
    remote, root, _ = source
    segment(remote, 0)
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = root / "raw/routes" / ROUTE / "0"
    destination.mkdir(parents=True)
    (destination / ".nnslr-partial").symlink_to(outside, target_is_directory=True)
    code, out, err = run(capfd)
    assert code == 2
    assert "destination_symlink" in out + err
    assert list(outside.iterdir()) == []


def test_unknown_route_is_not_reported_as_success(source, capfd):
    _, root, _ = source
    code, out, err = run(capfd, "--dry-run")
    assert code == 2
    assert "route_not_found" in out + err
    assert not root.exists()
# [nnslr-sync] - END

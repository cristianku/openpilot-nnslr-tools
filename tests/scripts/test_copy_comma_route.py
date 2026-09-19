from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "copy-comma-route.sh"


def _fake_tools(tmp_path: Path, remote_listing: str) -> tuple[Path, Path]:
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    log = tmp_path / "calls.log"

    ssh = fakebin / "ssh"
    ssh.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"printf 'ssh %s\\n' \"$*\" >> {log!s}\n"
        f"cat <<'EOF'\n{remote_listing}EOF\n",
        encoding="utf-8",
    )
    ssh.chmod(0o755)

    rsync = fakebin / "rsync"
    rsync.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"printf 'rsync %s\\n' \"$*\" >> {log!s}\n"
        "dest=\"${@: -1}\"\n"
        "mkdir -p \"$dest\"\n"
        "touch \"$dest/fcamera.hevc\" \"$dest/rlog.zst\"\n",
        encoding="utf-8",
    )
    rsync.chmod(0o755)
    return fakebin, log


def _run(tmp_path: Path, *args: str, listing: str) -> subprocess.CompletedProcess[str]:
    fakebin, _ = _fake_tools(tmp_path, listing)
    env = os.environ.copy()
    env["PATH"] = f"{fakebin}:{env['PATH']}"
    env["NNSLR_DATA_ROOT"] = str(tmp_path / "data")
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_copies_all_discovered_segments_with_explicit_host(tmp_path: Path) -> None:
    listing = (
        "/data/media/0/realdata/00000089--0ac1c0fdec--0\n"
        "/data/media/0/realdata/00000089--0ac1c0fdec--1\n"
        "/data/media/0/realdata/00000089--0ac1c0fdec--2\n"
        "/data/media/0/realdata/00000089--0ac1c0fdec--3\n"
    )
    proc = _run(tmp_path, "00000089--0ac1c0fdec", "192.168.1.77", listing=listing)
    assert proc.returncode == 0, proc.stderr
    root = tmp_path / "data" / "raw" / "routes" / "00000089--0ac1c0fdec"
    assert [p.name for p in sorted(root.iterdir())] == ["0", "1", "2", "3"]
    for segment in range(4):
        assert (root / str(segment) / "fcamera.hevc").is_file()
        assert (root / str(segment) / "rlog.zst").is_file()


def test_defaults_to_comma_remote_and_accepts_connect_style_route(tmp_path: Path) -> None:
    listing = "/data/media/0/realdata/00000089--0ac1c0fdec--0\n"
    fakebin, log = _fake_tools(tmp_path, listing)
    env = os.environ.copy()
    env["PATH"] = f"{fakebin}:{env['PATH']}"
    env["NNSLR_DATA_ROOT"] = str(tmp_path / "data")
    proc = subprocess.run(
        ["bash", str(SCRIPT), "6616faac453a3064/00000089--0ac1c0fdec"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    calls = log.read_text(encoding="utf-8")
    assert "comma-remote" in calls
    assert "00000089--0ac1c0fdec--0" in calls


def test_no_remote_segments_is_an_error(tmp_path: Path) -> None:
    proc = _run(tmp_path, "00000089--0ac1c0fdec", "comma-remote", listing="")
    assert proc.returncode != 0
    assert "no segments found" in proc.stderr.lower()


def test_requires_nnsler_data_root(tmp_path: Path) -> None:
    fakebin, _ = _fake_tools(tmp_path, "/data/media/0/realdata/00000089--0ac1c0fdec--0\n")
    env = os.environ.copy()
    env["PATH"] = f"{fakebin}:{env['PATH']}"
    env.pop("NNSLR_DATA_ROOT", None)
    proc = subprocess.run(
        ["bash", str(SCRIPT), "00000089--0ac1c0fdec"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "NNSLR_DATA_ROOT" in proc.stderr

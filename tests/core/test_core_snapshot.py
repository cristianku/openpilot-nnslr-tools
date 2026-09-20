from __future__ import annotations

import json
from pathlib import Path

import pytest

from nnslr_tools.core_snapshot import (
    CoreSnapshotError,
    export_core_snapshot,
    verify_core_snapshot,
)
from nnslr_tools import cli


def invoke(capsys, *args):
    try:
        code = cli.main(list(args))
    except SystemExit as exc:
        code = exc.code
    output = capsys.readouterr()
    return code, output


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_export_and_verify_core_snapshot(tmp_path) -> None:
    destination = tmp_path / "speed_vision_core"

    manifest, manifest_path = export_core_snapshot(_repo_root(), destination)
    verified = verify_core_snapshot(destination, manifest_path)

    assert verified.tree_digest == manifest.tree_digest
    assert verified.source_repository == "cristianku/openpilot-nnslr-tools"
    assert set(verified.files) == {"__init__.py", "py.typed", "temporal.py", "types.py"}
    payload = json.loads(manifest_path.read_text())
    assert payload["source_commit"] == manifest.source_commit


def test_core_snapshot_detects_tamper(tmp_path) -> None:
    destination = tmp_path / "speed_vision_core"
    _, manifest_path = export_core_snapshot(_repo_root(), destination)
    with (destination / "types.py").open("a") as handle:
        handle.write("\n# tampered\n")

    with pytest.raises(CoreSnapshotError, match="digest mismatch"):
        verify_core_snapshot(destination, manifest_path)


def test_export_core_cli_roundtrip(tmp_path, capsys) -> None:
    destination = tmp_path / "speed_vision_core"
    code, out = invoke(
        capsys,
        "export-core",
        "--output",
        str(destination),
    )

    assert code == 0, out.err + out.out
    report = json.loads(out.out)
    assert report["valid"] is True
    assert report["tree_digest"]
    assert Path(report["manifest"]).is_file()

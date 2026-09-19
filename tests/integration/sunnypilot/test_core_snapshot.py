from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


def _api():
    name = "nnslr_tools.integration.sunnypilot.core_snapshot"
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        spec = None
    assert spec is not None, f"{name} is not implemented"
    return importlib.import_module(name)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _repo(tmp_path: Path, *, types_body: str = "VALUE = 1\n") -> Path:
    repo = tmp_path / "repo"
    pkg = repo / "src" / "speed_vision_core"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("__version__ = '0.1.0'\n", encoding="utf-8")
    (pkg / "types.py").write_text(types_body, encoding="utf-8")
    (pkg / "py.typed").write_text("", encoding="utf-8")
    subprocess.check_call(["git", "-C", str(repo), "init", "-q"])
    subprocess.check_call(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"])
    subprocess.check_call(["git", "-C", str(repo), "config", "user.name", "Test"])
    subprocess.check_call(["git", "-C", str(repo), "add", "."])
    subprocess.check_call(["git", "-C", str(repo), "commit", "-q", "-m", "core"])
    return repo


def test_export_core_snapshot_is_exact_and_verifiable(tmp_path: Path) -> None:
    api = _api()
    repo = _repo(tmp_path)
    dest = tmp_path / "runtime" / "_vendor" / "speed_vision_core"
    manifest = api.export_core_snapshot(repo, dest)
    assert manifest.source_commit == _git(repo, "rev-parse", "HEAD")
    assert (dest / "types.py").read_bytes() == (repo / "src/speed_vision_core/types.py").read_bytes()
    assert set(manifest.files) == {"__init__.py", "types.py", "py.typed"}
    verified = api.verify_core_snapshot(dest, dest.parent / "speed_vision_core.snapshot.json")
    assert verified.tree_digest == manifest.tree_digest


def test_export_rejects_training_or_unregistered_python_file(tmp_path: Path) -> None:
    api = _api()
    repo = _repo(tmp_path)
    extra = repo / "src" / "speed_vision_core" / "train.py"
    extra.write_text("raise SystemExit('training')\n", encoding="utf-8")
    subprocess.check_call(["git", "-C", str(repo), "add", "."])
    subprocess.check_call(["git", "-C", str(repo), "commit", "-q", "-m", "extra"])
    with pytest.raises(api.CoreSnapshotError, match="unexpected"):
        api.export_core_snapshot(repo, tmp_path / "out")


def test_export_rejects_forbidden_framework_import(tmp_path: Path) -> None:
    api = _api()
    repo = _repo(tmp_path, types_body="import torch\nVALUE = 1\n")
    with pytest.raises(api.CoreSnapshotError, match="forbidden import"):
        api.export_core_snapshot(repo, tmp_path / "out")


def test_export_rejects_dirty_core_source(tmp_path: Path) -> None:
    api = _api()
    repo = _repo(tmp_path)
    (repo / "src/speed_vision_core/types.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(api.CoreSnapshotError, match="dirty"):
        api.export_core_snapshot(repo, tmp_path / "out")


def test_verify_rejects_tampered_snapshot(tmp_path: Path) -> None:
    api = _api()
    repo = _repo(tmp_path)
    dest = tmp_path / "out" / "speed_vision_core"
    api.export_core_snapshot(repo, dest)
    (dest / "types.py").write_text("VALUE = 999\n", encoding="utf-8")
    with pytest.raises(api.CoreSnapshotError, match="digest"):
        api.verify_core_snapshot(dest, dest.parent / "speed_vision_core.snapshot.json")

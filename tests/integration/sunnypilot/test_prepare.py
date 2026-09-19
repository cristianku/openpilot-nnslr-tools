from __future__ import annotations

import importlib
import importlib.util
import subprocess
from pathlib import Path

import pytest


def _api():
    name = "nnslr_tools.integration.sunnypilot.git_ops"
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        spec = None
    assert spec is not None, f"{name} is not implemented"
    return importlib.import_module(name)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _make_repo(path: Path) -> tuple[str, str]:
    path.mkdir()
    subprocess.check_call(["git", "-C", str(path), "init", "-q"])
    subprocess.check_call(["git", "-C", str(path), "config", "user.email", "test@example.invalid"])
    subprocess.check_call(["git", "-C", str(path), "config", "user.name", "Test"])
    (path / "value.txt").write_text("one\n", encoding="utf-8")
    subprocess.check_call(["git", "-C", str(path), "add", "."])
    subprocess.check_call(["git", "-C", str(path), "commit", "-q", "-m", "one"])
    first = _git(path, "rev-parse", "HEAD")
    (path / "value.txt").write_text("two\n", encoding="utf-8")
    subprocess.check_call(["git", "-C", str(path), "commit", "-qam", "two"])
    second = _git(path, "rev-parse", "HEAD")
    return first, second


def test_prepare_requires_exactly_one_upstream_selector(tmp_path: Path) -> None:
    api = _api()
    repo = tmp_path / "upstream"
    first, _ = _make_repo(repo)
    with pytest.raises(api.PrepareError, match="exactly one"):
        api.prepare_run(repo, tmp_path / "ws-a", upstream_ref=None, upstream_sha=None)
    with pytest.raises(api.PrepareError, match="exactly one"):
        api.prepare_run(repo, tmp_path / "ws-b", upstream_ref="HEAD", upstream_sha=first)


def test_prepare_freezes_requested_sha_in_separate_clones(tmp_path: Path) -> None:
    api = _api()
    repo = tmp_path / "upstream"
    first, second = _make_repo(repo)
    run = api.prepare_run(repo, tmp_path / "workspace", upstream_sha=first)
    assert run.resolved_upstream_commit == first
    assert _git(run.baseline_dir, "rev-parse", "HEAD") == first
    assert _git(run.candidate_dir, "rev-parse", "HEAD") == first
    assert _git(repo, "rev-parse", "HEAD") == second
    assert (repo / "value.txt").read_text(encoding="utf-8") == "two\n"


def test_prepare_ref_is_resolved_once_and_recorded(tmp_path: Path) -> None:
    api = _api()
    repo = tmp_path / "upstream"
    _, second = _make_repo(repo)
    run = api.prepare_run(repo, tmp_path / "workspace", upstream_ref="HEAD")
    assert run.resolved_upstream_commit == second
    assert run.requested_upstream_ref == "HEAD"


def test_prepare_rejects_occupied_workspace(tmp_path: Path) -> None:
    api = _api()
    repo = tmp_path / "upstream"
    _make_repo(repo)
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "keep.txt").write_text("do not touch", encoding="utf-8")
    with pytest.raises(api.PrepareError, match="workspace"):
        api.prepare_run(repo, ws, upstream_ref="HEAD")
    assert (ws / "keep.txt").read_text(encoding="utf-8") == "do not touch"


def test_workspace_lock_rejects_second_holder(tmp_path: Path) -> None:
    api = _api()
    root = tmp_path / "workspace"
    root.mkdir()
    with api.workspace_lock(root):
        with pytest.raises(api.PrepareError, match="locked"):
            with api.workspace_lock(root):
                pass

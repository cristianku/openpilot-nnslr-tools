from __future__ import annotations

import json
import subprocess
from pathlib import Path

from nnslr_tools.integration.sunnypilot import cli


def _git(repo: Path, *args: str) -> str:
  return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _repo(path: Path) -> str:
  path.mkdir()
  subprocess.check_call(["git", "-C", str(path), "init", "-q"])
  subprocess.check_call(["git", "-C", str(path), "config", "user.email", "test@example.invalid"])
  subprocess.check_call(["git", "-C", str(path), "config", "user.name", "Test"])
  (path / "x").write_text("x\n", encoding="utf-8")
  subprocess.check_call(["git", "-C", str(path), "add", "."])
  subprocess.check_call(["git", "-C", str(path), "commit", "-q", "-m", "base"])
  return _git(path, "rev-parse", "HEAD")


def _bootstrap_lock(path: Path, sha: str) -> Path:
  path.write_text(json.dumps({
    "schema_version": 1,
    "feature_id": "nnslr-speed-vision",
    "state": "bootstrap",
    "runtime_repository": "local/runtime",
    "upstream_repository": "local/upstream",
    "upstream_ref": "refs/heads/master",
    "source_base_commit": sha,
    "source_feature_commit": None,
    "patches": [],
    "core": {"state": "unavailable"},
    "schema_bindings": {"state": "unassigned"},
    "capability_profile": "bootstrap",
    "hook_map_digest": "0" * 64
  }), encoding="utf-8")
  return path


def test_inspect_reports_bootstrap_state(tmp_path: Path, capsys) -> None:
  repo = tmp_path / "repo"
  sha = _repo(repo)
  lock = _bootstrap_lock(tmp_path / "lock.json", sha)
  code = cli.main(["inspect", "--feature-lock", str(lock)])
  assert code == 0
  payload = json.loads(capsys.readouterr().out)
  assert payload["state"] == "bootstrap"
  assert payload["sealed"] is False


def test_prepare_local_repo_freezes_sha(tmp_path: Path, capsys) -> None:
  repo = tmp_path / "repo"
  sha = _repo(repo)
  lock = _bootstrap_lock(tmp_path / "lock.json", sha)
  run = tmp_path / "run"
  code = cli.main([
    "prepare", "--feature-lock", str(lock), "--upstream-repo", str(repo),
    "--upstream-sha", sha, "--workspace", str(run),
  ])
  assert code == 0
  payload = json.loads(capsys.readouterr().out)
  assert payload["resolved_upstream_commit"] == sha
  assert (run / "run.json").is_file()


def test_apply_bootstrap_returns_input_error(tmp_path: Path, capsys) -> None:
  repo = tmp_path / "repo"
  sha = _repo(repo)
  lock = _bootstrap_lock(tmp_path / "lock.json", sha)
  run = tmp_path / "run"
  assert cli.main([
    "prepare", "--feature-lock", str(lock), "--upstream-repo", str(repo),
    "--upstream-sha", sha, "--workspace", str(run),
  ]) == 0
  capsys.readouterr()
  code = cli.main([
    "apply", "--feature-lock", str(lock), "--run-dir", str(run),
    "--feature-repo", str(repo),
  ])
  assert code == 2
  assert "sealed" in capsys.readouterr().err

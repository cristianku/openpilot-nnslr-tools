from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from nnslr_tools.integration.sunnypilot.apply import ApplyError, apply_patchset
from nnslr_tools.integration.sunnypilot.git_ops import prepare_run
from nnslr_tools.integration.sunnypilot.manifest import load_feature_lock


def git(repo: Path, *args: str) -> str:
  return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def init_repo(path: Path) -> str:
  path.mkdir()
  subprocess.check_call(["git", "-C", str(path), "init", "-q"])
  subprocess.check_call(["git", "-C", str(path), "config", "user.email", "test@example.invalid"])
  subprocess.check_call(["git", "-C", str(path), "config", "user.name", "Test"])
  (path / "base.txt").write_text("base\n", encoding="utf-8")
  subprocess.check_call(["git", "-C", str(path), "add", "."])
  subprocess.check_call(["git", "-C", str(path), "commit", "-q", "-m", "base"])
  return git(path, "rev-parse", "HEAD")


def feature_repo(tmp_path: Path) -> tuple[Path, str, list[dict[str, object]]]:
  repo = tmp_path / "feature"
  base = init_repo(repo)
  patches = []
  parent = base
  for logical_id, filename, body in [
    ("nnslr.one", "one.txt", "one\n"),
    ("nnslr.two", "two.txt", "two\n"),
  ]:
    (repo / filename).write_text(body, encoding="utf-8")
    subprocess.check_call(["git", "-C", str(repo), "add", filename])
    subprocess.check_call(["git", "-C", str(repo), "commit", "-q", "-m", logical_id])
    commit = git(repo, "rev-parse", "HEAD")
    patches.append({
      "logical_id": logical_id,
      "commit": commit,
      "parent": parent,
      "required": True,
      "owned_paths": [filename],
      "integration_hooks": [],
      "postconditions": [],
    })
    parent = commit
  return repo, base, patches


def lock_file(tmp_path: Path, base: str, patches: list[dict[str, object]]) -> Path:
  payload = {
    "schema_version": 1,
    "feature_id": "nnslr-speed-vision",
    "state": "sealed",
    "runtime_repository": "local/feature",
    "upstream_repository": "local/upstream",
    "upstream_ref": "refs/heads/master",
    "source_base_commit": base,
    "source_feature_commit": patches[-1]["commit"],
    "patches": patches,
    "core": {"state": "snapshot"},
    "schema_bindings": {"state": "assigned"},
    "capability_profile": "source",
    "hook_map_digest": "0" * 64,
  }
  path = tmp_path / "feature.lock.json"
  path.write_text(json.dumps(payload), encoding="utf-8")
  return path


def test_apply_patchset_maps_source_commits_without_modifying_feature_repo(tmp_path: Path) -> None:
  feature, base, patches = feature_repo(tmp_path)
  lock = load_feature_lock(lock_file(tmp_path, base, patches))
  run = prepare_run(feature, tmp_path / "run", upstream_sha=base)
  before = git(feature, "rev-parse", "HEAD")
  result = apply_patchset(run, feature, lock)
  assert result.success is True
  assert [m.logical_id for m in result.commit_mapping] == ["nnslr.one", "nnslr.two"]
  assert (run.candidate_dir / "one.txt").read_text() == "one\n"
  assert (run.candidate_dir / "two.txt").read_text() == "two\n"
  assert git(feature, "rev-parse", "HEAD") == before
  assert result.result_tree == git(run.candidate_dir, "rev-parse", "HEAD^{tree}")


def test_apply_refuses_bootstrap_lock(tmp_path: Path) -> None:
  feature, base, _ = feature_repo(tmp_path)
  payload = {
    "schema_version": 1, "feature_id": "nnslr-speed-vision", "state": "bootstrap",
    "runtime_repository": "x/y", "upstream_repository": "x/z", "upstream_ref": "refs/heads/master",
    "source_base_commit": base, "source_feature_commit": None, "patches": [],
    "core": {"state": "unavailable"}, "schema_bindings": {"state": "unassigned"},
    "capability_profile": "bootstrap", "hook_map_digest": "0" * 64,
  }
  path = tmp_path / "bootstrap.json"
  path.write_text(json.dumps(payload), encoding="utf-8")
  run = prepare_run(feature, tmp_path / "run", upstream_sha=base)
  with pytest.raises(ApplyError, match="sealed"):
    apply_patchset(run, feature, load_feature_lock(path))

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from nnslr_tools.integration.sunnypilot.git_ops import prepare_run
from nnslr_tools.integration.sunnypilot.verify import verify_source


def git(repo: Path, *args: str) -> str:
  return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def make_repo(path: Path) -> str:
  path.mkdir()
  subprocess.check_call(["git", "-C", str(path), "init", "-q"])
  subprocess.check_call(["git", "-C", str(path), "config", "user.email", "test@example.invalid"])
  subprocess.check_call(["git", "-C", str(path), "config", "user.name", "Test"])
  (path / "hook.py").write_text("BASELINE_ANCHOR\n", encoding="utf-8")
  (path / "protected.py").write_text("PROTECTED_ANCHOR\n", encoding="utf-8")
  subprocess.check_call(["git", "-C", str(path), "add", "."])
  subprocess.check_call(["git", "-C", str(path), "commit", "-q", "-m", "base"])
  return git(path, "rev-parse", "HEAD")


def config(tmp_path: Path, base: str) -> tuple[Path, Path]:
  hooks = tmp_path / "hooks.json"
  hooks.write_text(json.dumps({
    "schema_version": 1,
    "baseline": {"repository": "local/test", "commit": base},
    "hooks": [{"id": "hook", "path": "hook.py", "anchors": ["BASELINE_ANCHOR"], "required": True}],
  }), encoding="utf-8")
  protected = tmp_path / "protected.json"
  protected.write_text(json.dumps({
    "schema_version": 1,
    "baseline_commit": base,
    "forbidden_write_paths": ["protected.py"],
    "protected_contracts": [{"id": "p", "path": "protected.py", "anchors": ["PROTECTED_ANCHOR"], "rule": "unchanged"}],
    "forbidden_runtime_effects": [],
  }), encoding="utf-8")
  return hooks, protected


def test_verify_source_accepts_owned_nonprotected_change(tmp_path: Path) -> None:
  repo = tmp_path / "repo"
  base = make_repo(repo)
  hooks, protected = config(tmp_path, base)
  run = prepare_run(repo, tmp_path / "run", upstream_sha=base)
  (run.candidate_dir / "feature.py").write_text("feature\n", encoding="utf-8")
  subprocess.check_call(["git", "-C", str(run.candidate_dir), "add", "."])
  subprocess.check_call(["git", "-C", str(run.candidate_dir), "config", "user.email", "x@y.invalid"])
  subprocess.check_call(["git", "-C", str(run.candidate_dir), "config", "user.name", "X"])
  subprocess.check_call(["git", "-C", str(run.candidate_dir), "commit", "-q", "-m", "feature"])
  report = verify_source(run, hooks, protected)
  assert report.compatible is True
  assert "feature.py" in report.changed_paths


def test_verify_source_rejects_forbidden_path_change(tmp_path: Path) -> None:
  repo = tmp_path / "repo"
  base = make_repo(repo)
  hooks, protected = config(tmp_path, base)
  run = prepare_run(repo, tmp_path / "run", upstream_sha=base)
  (run.candidate_dir / "protected.py").write_text("PROTECTED_ANCHOR\nchanged\n", encoding="utf-8")
  subprocess.check_call(["git", "-C", str(run.candidate_dir), "add", "."])
  subprocess.check_call(["git", "-C", str(run.candidate_dir), "config", "user.email", "x@y.invalid"])
  subprocess.check_call(["git", "-C", str(run.candidate_dir), "config", "user.name", "X"])
  subprocess.check_call(["git", "-C", str(run.candidate_dir), "commit", "-q", "-m", "bad"])
  report = verify_source(run, hooks, protected)
  assert report.compatible is False
  assert any(c.name == "protected_paths" and c.status == "failed" for c in report.checks)


def test_verify_source_detects_missing_hook_anchor(tmp_path: Path) -> None:
  repo = tmp_path / "repo"
  base = make_repo(repo)
  hooks, protected = config(tmp_path, base)
  run = prepare_run(repo, tmp_path / "run", upstream_sha=base)
  (run.candidate_dir / "hook.py").write_text("moved without adapter\n", encoding="utf-8")
  subprocess.check_call(["git", "-C", str(run.candidate_dir), "add", "."])
  subprocess.check_call(["git", "-C", str(run.candidate_dir), "config", "user.email", "x@y.invalid"])
  subprocess.check_call(["git", "-C", str(run.candidate_dir), "config", "user.name", "X"])
  subprocess.check_call(["git", "-C", str(run.candidate_dir), "commit", "-q", "-m", "move"])
  report = verify_source(run, hooks, protected)
  assert report.compatible is False
  assert any(c.name == "hook_discovery" and c.status == "failed" for c in report.checks)

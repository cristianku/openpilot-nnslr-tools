from __future__ import annotations

import subprocess
from pathlib import Path

from test_apply import feature_repo, git, lock_file
from nnslr_tools.integration.sunnypilot.apply import apply_patchset
from nnslr_tools.integration.sunnypilot.git_ops import prepare_run
from nnslr_tools.integration.sunnypilot.manifest import load_feature_lock


def test_conflict_is_reported_and_cherry_pick_is_aborted(tmp_path: Path) -> None:
  feature, base, patches = feature_repo(tmp_path)

  subprocess.check_call(["git", "-C", str(feature), "checkout", "-q", base])
  (feature / "one.txt").write_text("upstream incompatible\n", encoding="utf-8")
  subprocess.check_call(["git", "-C", str(feature), "add", "one.txt"])
  subprocess.check_call(["git", "-C", str(feature), "commit", "-q", "-m", "upstream"])
  advanced = git(feature, "rev-parse", "HEAD")

  lock = load_feature_lock(lock_file(tmp_path, base, patches))
  run = prepare_run(feature, tmp_path / "run", upstream_sha=advanced)
  result = apply_patchset(run, feature, lock)
  assert result.success is False
  assert result.conflicts
  assert result.conflicts[0].logical_id == "nnslr.one"
  assert result.conflicts[0].kind in {"content_conflict", "empty_patch_requires_review"}
  assert not (run.candidate_dir / ".git" / "CHERRY_PICK_HEAD").exists()

from __future__ import annotations

from test_apply import feature_repo, git, lock_file
from nnslr_tools.integration.sunnypilot.apply import apply_patchset
from nnslr_tools.integration.sunnypilot.git_ops import prepare_run
from nnslr_tools.integration.sunnypilot.manifest import load_feature_lock


def test_same_inputs_produce_same_tree(tmp_path) -> None:
  feature, base, patches = feature_repo(tmp_path)
  lock = load_feature_lock(lock_file(tmp_path, base, patches))
  run1 = prepare_run(feature, tmp_path / "run1", upstream_sha=base)
  run2 = prepare_run(feature, tmp_path / "run2", upstream_sha=base)
  result1 = apply_patchset(run1, feature, lock)
  result2 = apply_patchset(run2, feature, lock)
  assert result1.success and result2.success
  assert result1.result_tree == result2.result_tree
  assert git(run1.candidate_dir, "rev-parse", "HEAD^{tree}") == git(run2.candidate_dir, "rev-parse", "HEAD^{tree}")


def test_second_apply_on_completed_run_is_idempotent(tmp_path) -> None:
  feature, base, patches = feature_repo(tmp_path)
  lock = load_feature_lock(lock_file(tmp_path, base, patches))
  run = prepare_run(feature, tmp_path / "run", upstream_sha=base)
  first = apply_patchset(run, feature, lock)
  second = apply_patchset(run, feature, lock)
  assert first.result_tree == second.result_tree
  assert second.resumed is True
  assert second.added_commit_count == 0

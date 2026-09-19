from __future__ import annotations

import json
from pathlib import Path

from test_apply import feature_repo, lock_file
from nnslr_tools.integration.sunnypilot.apply import apply_patchset
from nnslr_tools.integration.sunnypilot.git_ops import prepare_run
from nnslr_tools.integration.sunnypilot.manifest import load_feature_lock
from nnslr_tools.integration.sunnypilot.reports import write_report


def test_report_is_machine_readable_and_public_markdown_omits_workspace_path(tmp_path: Path) -> None:
  feature, base, patches = feature_repo(tmp_path)
  lock = load_feature_lock(lock_file(tmp_path, base, patches))
  run = prepare_run(feature, tmp_path / "private-workspace", upstream_sha=base)
  result = apply_patchset(run, feature, lock)
  paths = write_report(run, result, readiness_level="source_applied", verification={})
  payload = json.loads(paths.json.read_text(encoding="utf-8"))
  assert payload["resolved_upstream_commit"] == base
  assert payload["result_tree"] == result.result_tree
  markdown = paths.markdown.read_text(encoding="utf-8")
  assert str(tmp_path) not in markdown
  assert "source_applied" in markdown

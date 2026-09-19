from __future__ import annotations

import json
from pathlib import Path

import pytest

from nnslr_tools.integration.sunnypilot.discover import (
    DiscoveryError,
    discover_hooks,
    load_hook_map,
)

ROOT = Path(__file__).resolve().parents[3]
HOOK_MAP = ROOT / "integration" / "sunnypilot" / "hook-map.json"
FIXTURE = ROOT / "tests" / "integration" / "sunnypilot" / "fixtures" / "pinned_baseline_index.json"


def _fixture_files() -> dict[str, str]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return {item["path"]: "\n".join(item["anchors"]) for item in payload["files"]}


def test_discovery_matches_pinned_baseline() -> None:
    hook_map = load_hook_map(HOOK_MAP)
    report = discover_hooks(_fixture_files(), hook_map)
    assert report.compatible is True
    assert report.missing_paths == ()
    assert report.missing_anchors == ()
    assert {h.hook_id for h in report.resolved} == {h.hook_id for h in hook_map.hooks}


def test_discovery_reports_missing_required_anchor() -> None:
    hook_map = load_hook_map(HOOK_MAP)
    files = _fixture_files()
    files["openpilot/cereal/services.py"] = files["openpilot/cereal/services.py"].replace('"liveMapDataSP"', "")
    report = discover_hooks(files, hook_map)
    assert report.compatible is False
    assert ("services", '"liveMapDataSP"') in report.missing_anchors


def test_load_hook_map_rejects_duplicate_hook_ids(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "baseline": {"repository": "x/y", "commit": "a" * 40},
        "hooks": [
            {"id": "same", "path": "a", "anchors": ["x"], "required": True},
            {"id": "same", "path": "b", "anchors": ["y"], "required": True},
        ],
    }
    path = tmp_path / "hook-map.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DiscoveryError, match="duplicate hook id"):
        load_hook_map(path)

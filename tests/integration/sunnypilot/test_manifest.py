from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
LOCK = ROOT / "integration" / "sunnypilot" / "feature.lock.json"


def _api():
    name = "nnslr_tools.integration.sunnypilot.manifest"
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        spec = None
    assert spec is not None, f"{name} is not implemented"
    module = importlib.import_module(name)
    return module


def test_bootstrap_lock_is_valid_but_not_sealed() -> None:
    api = _api()
    lock = api.load_feature_lock(LOCK)
    assert lock.feature_id == "nnslr-speed-vision"
    assert lock.state == "bootstrap"
    assert lock.source_base_commit == "a5f44653d7f43ad57fef2f546f3916ec4cbf3c56"
    assert lock.source_feature_commit is None
    assert lock.patches == ()


def test_sealed_lock_requires_feature_commit_and_nonempty_patchset(tmp_path: Path) -> None:
    api = _api()
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    payload["state"] = "sealed"
    p = tmp_path / "sealed.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(api.FeatureLockError, match="sealed"):
        api.load_feature_lock(p)


def test_rejects_non_linear_patch_parent_chain(tmp_path: Path) -> None:
    api = _api()
    base = "a" * 40
    p1 = "b" * 40
    p2 = "c" * 40
    payload = {
        "schema_version": 1,
        "feature_id": "nnslr-speed-vision",
        "state": "sealed",
        "runtime_repository": "owner/runtime",
        "upstream_repository": "owner/upstream",
        "upstream_ref": "refs/heads/master",
        "source_base_commit": base,
        "source_feature_commit": p2,
        "patches": [
            {"logical_id": "one", "commit": p1, "parent": base, "required": True,
             "owned_paths": ["a"], "integration_hooks": [], "postconditions": []},
            {"logical_id": "two", "commit": p2, "parent": base, "required": True,
             "owned_paths": ["b"], "integration_hooks": [], "postconditions": []}
        ],
        "core": {"state": "unavailable"},
        "schema_bindings": {"state": "unassigned"},
        "capability_profile": "bootstrap",
        "hook_map_digest": "0" * 64
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(api.FeatureLockError, match="parent"):
        api.load_feature_lock(p)

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class FeatureLockError(ValueError):
    """The feature lock is malformed or internally inconsistent."""


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise FeatureLockError(f"{field} must be a non-empty string")
    return value


def _sha(value: object, field: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    text = _text(value, field)
    if len(text) != 40 or any(ch not in "0123456789abcdef" for ch in text):
        raise FeatureLockError(f"{field} must be a 40-character lowercase hex SHA")
    return text


def _sha256(value: object, field: str) -> str:
    text = _text(value, field)
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise FeatureLockError(f"{field} must be a 64-character lowercase hex digest")
    return text


def _str_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise FeatureLockError(f"{field} must be a list")
    return tuple(_text(v, field) for v in value)


@dataclass(frozen=True)
class PatchSpec:
    logical_id: str
    commit: str
    parent: str
    required: bool
    owned_paths: tuple[str, ...]
    integration_hooks: tuple[str, ...]
    postconditions: tuple[str, ...]


@dataclass(frozen=True)
class FeatureLock:
    schema_version: int
    feature_id: str
    state: str
    runtime_repository: str
    upstream_repository: str
    upstream_ref: str
    source_base_commit: str
    source_feature_commit: str | None
    patches: tuple[PatchSpec, ...]
    core: Mapping[str, Any]
    schema_bindings: Mapping[str, Any]
    capability_profile: str
    hook_map_digest: str

    @property
    def sealed(self) -> bool:
        return self.state == "sealed"


def load_feature_lock(path: Path) -> FeatureLock:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FeatureLockError(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise FeatureLockError("feature lock must be a JSON object")
    if payload.get("schema_version") != 1:
        raise FeatureLockError("unsupported schema_version")

    feature_id = _text(payload.get("feature_id"), "feature_id")
    if feature_id != "nnslr-speed-vision":
        raise FeatureLockError("feature_id must be nnslr-speed-vision")
    state = _text(payload.get("state"), "state")
    if state not in {"bootstrap", "sealed"}:
        raise FeatureLockError("state must be bootstrap or sealed")

    base = _sha(payload.get("source_base_commit"), "source_base_commit")
    assert base is not None
    feature_commit = _sha(payload.get("source_feature_commit"), "source_feature_commit", optional=True)

    raw_patches = payload.get("patches")
    if not isinstance(raw_patches, list):
        raise FeatureLockError("patches must be a list")
    patches: list[PatchSpec] = []
    seen_ids: set[str] = set()
    expected_parent = base
    for i, raw in enumerate(raw_patches):
        if not isinstance(raw, dict):
            raise FeatureLockError(f"patches[{i}] must be an object")
        logical_id = _text(raw.get("logical_id"), f"patches[{i}].logical_id")
        if logical_id in seen_ids:
            raise FeatureLockError(f"duplicate patch logical_id: {logical_id}")
        seen_ids.add(logical_id)
        commit = _sha(raw.get("commit"), f"patches[{i}].commit")
        parent = _sha(raw.get("parent"), f"patches[{i}].parent")
        assert commit is not None and parent is not None
        if parent != expected_parent:
            raise FeatureLockError(
                f"patches[{i}].parent does not continue the linear parent chain: expected {expected_parent}, got {parent}"
            )
        required = raw.get("required")
        if not isinstance(required, bool):
            raise FeatureLockError(f"patches[{i}].required must be bool")
        patches.append(PatchSpec(
            logical_id=logical_id,
            commit=commit,
            parent=parent,
            required=required,
            owned_paths=_str_tuple(raw.get("owned_paths", []), f"patches[{i}].owned_paths"),
            integration_hooks=_str_tuple(raw.get("integration_hooks", []), f"patches[{i}].integration_hooks"),
            postconditions=_str_tuple(raw.get("postconditions", []), f"patches[{i}].postconditions"),
        ))
        expected_parent = commit

    if state == "bootstrap":
        if feature_commit is not None or patches:
            raise FeatureLockError("bootstrap lock must not claim a sealed feature commit or patchset")
    else:
        if feature_commit is None or not patches:
            raise FeatureLockError("sealed lock requires source_feature_commit and a non-empty patchset")
        if patches[-1].commit != feature_commit:
            raise FeatureLockError("sealed source_feature_commit must equal the last patch commit")

    core = payload.get("core")
    schema_bindings = payload.get("schema_bindings")
    if not isinstance(core, dict) or not isinstance(schema_bindings, dict):
        raise FeatureLockError("core and schema_bindings must be objects")

    return FeatureLock(
        schema_version=1,
        feature_id=feature_id,
        state=state,
        runtime_repository=_text(payload.get("runtime_repository"), "runtime_repository"),
        upstream_repository=_text(payload.get("upstream_repository"), "upstream_repository"),
        upstream_ref=_text(payload.get("upstream_ref"), "upstream_ref"),
        source_base_commit=base,
        source_feature_commit=feature_commit,
        patches=tuple(patches),
        core=core,
        schema_bindings=schema_bindings,
        capability_profile=_text(payload.get("capability_profile"), "capability_profile"),
        hook_map_digest=_sha256(payload.get("hook_map_digest"), "hook_map_digest"),
    )

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class DiscoveryError(ValueError):
    """Raised when a hook map is malformed."""


@dataclass(frozen=True)
class HookSpec:
    hook_id: str
    path: str
    anchors: tuple[str, ...]
    required: bool = True


@dataclass(frozen=True)
class HookMap:
    schema_version: int
    baseline_repository: str
    baseline_commit: str
    hooks: tuple[HookSpec, ...]


@dataclass(frozen=True)
class ResolvedHook:
    hook_id: str
    path: str


@dataclass(frozen=True)
class DiscoveryReport:
    compatible: bool
    resolved: tuple[ResolvedHook, ...]
    missing_paths: tuple[str, ...]
    missing_anchors: tuple[tuple[str, str], ...]


def _require_str(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise DiscoveryError(f"{field} must be a non-empty string")
    return value


def load_hook_map(path: Path) -> HookMap:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise DiscoveryError(f"invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise DiscoveryError("hook map must be a JSON object")
    if payload.get("schema_version") != 1:
        raise DiscoveryError("unsupported hook-map schema_version")

    baseline = payload.get("baseline")
    if not isinstance(baseline, dict):
        raise DiscoveryError("baseline must be an object")
    repository = _require_str(baseline.get("repository"), "baseline.repository")
    commit = _require_str(baseline.get("commit"), "baseline.commit")
    if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise DiscoveryError("baseline.commit must be a 40-character lowercase hex SHA")

    raw_hooks = payload.get("hooks")
    if not isinstance(raw_hooks, list) or not raw_hooks:
        raise DiscoveryError("hooks must be a non-empty list")

    hooks: list[HookSpec] = []
    seen: set[str] = set()
    for i, raw in enumerate(raw_hooks):
        if not isinstance(raw, dict):
            raise DiscoveryError(f"hooks[{i}] must be an object")
        hook_id = _require_str(raw.get("id"), f"hooks[{i}].id")
        if hook_id in seen:
            raise DiscoveryError(f"duplicate hook id: {hook_id}")
        seen.add(hook_id)
        hook_path = _require_str(raw.get("path"), f"hooks[{i}].path")
        raw_anchors = raw.get("anchors")
        if not isinstance(raw_anchors, list) or not raw_anchors:
            raise DiscoveryError(f"hooks[{i}].anchors must be a non-empty list")
        anchors = tuple(_require_str(v, f"hooks[{i}].anchors") for v in raw_anchors)
        required = raw.get("required", True)
        if not isinstance(required, bool):
            raise DiscoveryError(f"hooks[{i}].required must be bool")
        hooks.append(HookSpec(hook_id, hook_path, anchors, required))

    return HookMap(1, repository, commit, tuple(hooks))


def discover_hooks(files: Mapping[str, str], hook_map: HookMap) -> DiscoveryReport:
    resolved: list[ResolvedHook] = []
    missing_paths: list[str] = []
    missing_anchors: list[tuple[str, str]] = []

    for hook in hook_map.hooks:
        text = files.get(hook.path)
        if text is None:
            if hook.required:
                missing_paths.append(hook.path)
            continue
        missing_for_hook = [anchor for anchor in hook.anchors if anchor not in text]
        if missing_for_hook:
            if hook.required:
                missing_anchors.extend((hook.hook_id, anchor) for anchor in missing_for_hook)
            continue
        resolved.append(ResolvedHook(hook.hook_id, hook.path))

    return DiscoveryReport(
        compatible=not missing_paths and not missing_anchors,
        resolved=tuple(resolved),
        missing_paths=tuple(sorted(set(missing_paths))),
        missing_anchors=tuple(missing_anchors),
    )

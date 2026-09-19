from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .git_ops import PreparedRun
from .manifest import FeatureLock, PatchSpec


class ApplyError(RuntimeError):
    """Patchset application cannot proceed safely."""


@dataclass(frozen=True)
class CommitMapping:
    logical_id: str
    source_commit: str
    result_commit: str
    status: str = "applied"


@dataclass(frozen=True)
class ApplyConflict:
    logical_id: str
    source_commit: str
    kind: str
    detail: str


@dataclass(frozen=True)
class ApplyResult:
    success: bool
    commit_mapping: tuple[CommitMapping, ...]
    conflicts: tuple[ApplyConflict, ...]
    result_tree: str | None
    resumed: bool = False
    added_commit_count: int = 0


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and proc.returncode != 0:
        raise ApplyError(proc.stderr.strip() or proc.stdout.strip() or f"git {' '.join(args)} failed")
    return proc


def _git_text(repo: Path, *args: str) -> str:
    return _git(repo, *args).stdout.strip()


def _state_path(run: PreparedRun) -> Path:
    return run.run_dir / "apply.json"


def _save(run: PreparedRun, result: ApplyResult) -> None:
    payload = {
        "schema_version": 1,
        "success": result.success,
        "commit_mapping": [asdict(v) for v in result.commit_mapping],
        "conflicts": [asdict(v) for v in result.conflicts],
        "result_tree": result.result_tree,
    }
    _state_path(run).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_completed(run: PreparedRun) -> ApplyResult | None:
    path = _state_path(run)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("success"):
        return None
    current_tree = _git_text(run.candidate_dir, "rev-parse", "HEAD^{tree}")
    if current_tree != payload.get("result_tree"):
        raise ApplyError("candidate changed after a completed apply; refusing implicit reconciliation")
    return ApplyResult(
        success=True,
        commit_mapping=tuple(CommitMapping(**row) for row in payload.get("commit_mapping", [])),
        conflicts=(),
        result_tree=current_tree,
        resumed=True,
        added_commit_count=0,
    )


def _verify_source_patch(feature_repo: Path, patch: PatchSpec) -> None:
    source = _git_text(feature_repo, "rev-parse", "--verify", f"{patch.commit}^{{commit}}")
    if source != patch.commit:
        raise ApplyError(f"source commit did not resolve exactly for {patch.logical_id}")
    parent_proc = _git(feature_repo, "rev-parse", f"{patch.commit}^", check=False)
    parent = parent_proc.stdout.strip() if parent_proc.returncode == 0 else ""
    if parent != patch.parent:
        raise ApplyError(
            f"source parent mismatch for {patch.logical_id}: lock={patch.parent}, repository={parent}"
        )
    changed = {
        line.strip()
        for line in _git_text(feature_repo, "diff-tree", "--no-commit-id", "--name-only", "-r", patch.commit).splitlines()
        if line.strip()
    }
    declared = set(patch.owned_paths)
    undeclared = changed - declared
    if undeclared:
        raise ApplyError(
            f"patch {patch.logical_id} changes undeclared paths: {', '.join(sorted(undeclared))}"
        )


def _has_unmerged(repo: Path) -> bool:
    rows = _git_text(repo, "status", "--porcelain").splitlines()
    unmerged = {"DD", "AU", "UD", "UA", "DU", "AA", "UU"}
    return any(row[:2] in unmerged for row in rows if len(row) >= 2)


def apply_patchset(run: PreparedRun, feature_repo: Path, lock: FeatureLock) -> ApplyResult:
    """Apply a sealed, linear patchset to an isolated candidate checkout."""
    if not lock.sealed:
        raise ApplyError("feature lock must be sealed before patchset application")
    feature_repo = Path(feature_repo).resolve()
    if not (feature_repo / ".git").exists():
        raise ApplyError(f"feature repository is not a Git checkout: {feature_repo}")

    completed = _load_completed(run)
    if completed is not None:
        return completed

    if _git_text(run.candidate_dir, "status", "--porcelain"):
        raise ApplyError("candidate worktree is dirty before apply")
    if _git_text(run.candidate_dir, "rev-parse", "HEAD") != run.resolved_upstream_commit:
        raise ApplyError("candidate HEAD no longer matches the frozen upstream base")

    for patch in lock.patches:
        _verify_source_patch(feature_repo, patch)

    _git(run.candidate_dir, "config", "user.email", "nnslr-reapply@example.invalid")
    _git(run.candidate_dir, "config", "user.name", "NNSLR Reapply")

    mappings: list[CommitMapping] = []
    for patch in lock.patches:
        fetch = _git(run.candidate_dir, "fetch", "--quiet", str(feature_repo), patch.commit, check=False)
        if fetch.returncode != 0:
            raise ApplyError(fetch.stderr.strip() or f"cannot fetch {patch.logical_id}")

        pick = _git(run.candidate_dir, "cherry-pick", "--no-edit", patch.commit, check=False)
        if pick.returncode != 0:
            kind = "content_conflict" if _has_unmerged(run.candidate_dir) else "empty_patch_requires_review"
            detail = (pick.stderr.strip() or pick.stdout.strip() or "cherry-pick failed")[:2000]
            _git(run.candidate_dir, "cherry-pick", "--abort", check=False)
            result = ApplyResult(
                success=False,
                commit_mapping=tuple(mappings),
                conflicts=(ApplyConflict(patch.logical_id, patch.commit, kind, detail),),
                result_tree=_git_text(run.candidate_dir, "rev-parse", "HEAD^{tree}"),
                added_commit_count=len(mappings),
            )
            _save(run, result)
            return result

        result_commit = _git_text(run.candidate_dir, "rev-parse", "HEAD")
        mappings.append(CommitMapping(patch.logical_id, patch.commit, result_commit))

    result = ApplyResult(
        success=True,
        commit_mapping=tuple(mappings),
        conflicts=(),
        result_tree=_git_text(run.candidate_dir, "rev-parse", "HEAD^{tree}"),
        added_commit_count=len(mappings),
    )
    _save(run, result)
    return result

from __future__ import annotations

import json
import os
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


class PrepareError(RuntimeError):
    """Preparation failed without mutating the source checkout."""


@dataclass(frozen=True)
class PreparedRun:
    run_dir: Path
    baseline_dir: Path
    candidate_dir: Path
    resolved_upstream_commit: str
    requested_upstream_ref: str | None
    requested_upstream_sha: str | None


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise PrepareError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc.stdout.strip()


@contextmanager
def workspace_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".nnslr-reapply.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise PrepareError(f"workspace is locked: {root}") from exc
    try:
        os.write(fd, f"pid={os.getpid()}\n".encode())
        os.close(fd)
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _resolve_commit(repo: Path, ref: str) -> str:
    commit = _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")
    if len(commit) != 40:
        raise PrepareError(f"resolved commit is not a full SHA: {commit}")
    return commit


def _clone_at(source: Path, destination: Path, commit: str) -> None:
    proc = subprocess.run(
        ["git", "clone", "--quiet", "--no-checkout", str(source), str(destination)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise PrepareError(proc.stderr.strip() or f"git clone failed: {source}")
    _git(destination, "checkout", "--quiet", "--detach", commit)


def prepare_run(
    upstream_repository: Path,
    workspace: Path,
    *,
    upstream_ref: str | None = None,
    upstream_sha: str | None = None,
) -> PreparedRun:
    if (upstream_ref is None) == (upstream_sha is None):
        raise PrepareError("provide exactly one of upstream_ref or upstream_sha")
    source = Path(upstream_repository).resolve()
    if not (source / ".git").exists():
        raise PrepareError(f"upstream repository is not a Git checkout: {source}")
    workspace = Path(workspace).resolve()
    if workspace.exists():
        raise PrepareError(f"workspace already exists; refusing to overwrite: {workspace}")

    workspace.mkdir(parents=True)
    with workspace_lock(workspace):
        selector = upstream_ref if upstream_ref is not None else upstream_sha
        assert selector is not None
        resolved = _resolve_commit(source, selector)
        if upstream_sha is not None and resolved != upstream_sha:
            raise PrepareError(f"requested upstream SHA did not resolve exactly: {upstream_sha} -> {resolved}")

        baseline = workspace / "baseline"
        candidate = workspace / "candidate"
        _clone_at(source, baseline, resolved)
        _clone_at(source, candidate, resolved)

        receipt = {
            "schema_version": 1,
            "requested_upstream_ref": upstream_ref,
            "requested_upstream_sha": upstream_sha,
            "resolved_upstream_commit": resolved,
            "baseline_dir": str(baseline),
            "candidate_dir": str(candidate),
        }
        (workspace / "run.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return PreparedRun(
        run_dir=workspace,
        baseline_dir=baseline,
        candidate_dir=candidate,
        resolved_upstream_commit=resolved,
        requested_upstream_ref=upstream_ref,
        requested_upstream_sha=upstream_sha,
    )

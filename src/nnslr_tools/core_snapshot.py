"""Portable stdlib-only speed_vision_core snapshot export and verification."""
from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


SOURCE_REPOSITORY = "cristianku/openpilot-nnslr-tools"
DEFAULT_CORE_FILES = ("__init__.py", "py.typed", "types.py")
FORBIDDEN_IMPORT_ROOTS = frozenset({
    "torch", "numpy", "cv2", "onnx", "onnxruntime", "tinygrad",
    "openpilot", "cereal", "opendbc", "pandas", "cupy", "pycuda", "triton",
})


class CoreSnapshotError(ValueError):
    pass


@dataclass(frozen=True)
class CoreSnapshotManifest:
    schema_version: int
    source_repository: str
    source_commit: str
    tree_digest: str
    files: tuple[str, ...]
    file_sha256: Mapping[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_repository": self.source_repository,
            "source_commit": self.source_commit,
            "tree_digest": self.tree_digest,
            "files": list(self.files),
            "file_sha256": dict(self.file_sha256),
        }


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise CoreSnapshotError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_digest(file_hashes: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(file_hashes):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hashes[name].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _check_imports(path: Path) -> None:
    if path.suffix != ".py":
        return
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        raise CoreSnapshotError(f"invalid Python in core snapshot source: {path}: {exc}") from exc
    for node in ast.walk(tree):
        modules = []
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        for module in modules:
            if module.split(".", 1)[0] in FORBIDDEN_IMPORT_ROOTS:
                raise CoreSnapshotError(f"forbidden import {module!r} in {path.name}")


def _load_manifest(path: Path) -> CoreSnapshotManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise CoreSnapshotError("unsupported snapshot manifest")
    files = payload.get("files")
    hashes = payload.get("file_sha256")
    if not isinstance(files, list) or not all(isinstance(v, str) for v in files):
        raise CoreSnapshotError("invalid snapshot file list")
    if not isinstance(hashes, dict) or set(hashes) != set(files):
        raise CoreSnapshotError("invalid snapshot digest table")
    return CoreSnapshotManifest(
        schema_version=1,
        source_repository=str(payload.get("source_repository", "")),
        source_commit=str(payload.get("source_commit", "")),
        tree_digest=str(payload.get("tree_digest", "")),
        files=tuple(files),
        file_sha256={str(k): str(v) for k, v in hashes.items()},
    )


def export_core_snapshot(
    repo_root: Path,
    destination: Path,
    *,
    allowed_files: tuple[str, ...] = DEFAULT_CORE_FILES,
) -> tuple[CoreSnapshotManifest, Path]:
    repo_root = Path(repo_root).resolve()
    package = repo_root / "src" / "speed_vision_core"
    if not package.is_dir():
        raise CoreSnapshotError(f"core package not found: {package}")
    if not (repo_root / ".git").exists():
        raise CoreSnapshotError(f"source is not a Git checkout: {repo_root}")

    dirty = _git(repo_root, "status", "--porcelain", "--", "src/speed_vision_core")
    if dirty:
        raise CoreSnapshotError(f"dirty core source: {dirty.splitlines()[0]}")

    actual = tuple(sorted(p.name for p in package.iterdir() if p.is_file()))
    expected = tuple(sorted(allowed_files))
    if actual != expected:
        raise CoreSnapshotError(
            f"unexpected core file set; unexpected={sorted(set(actual)-set(expected))}, "
            f"missing={sorted(set(expected)-set(actual))}"
        )
    for name in expected:
        _check_imports(package / name)

    destination = Path(destination).resolve()
    if destination.exists():
        raise CoreSnapshotError(f"snapshot destination already exists: {destination}")
    destination.mkdir(parents=True)
    for name in expected:
        shutil.copy2(package / name, destination / name)

    hashes = {name: _sha256(destination / name) for name in expected}
    manifest = CoreSnapshotManifest(
        schema_version=1,
        source_repository=SOURCE_REPOSITORY,
        source_commit=_git(repo_root, "rev-parse", "HEAD"),
        tree_digest=_tree_digest(hashes),
        files=expected,
        file_sha256=hashes,
    )
    manifest_path = destination.parent / "speed_vision_core.snapshot.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest, manifest_path


def verify_core_snapshot(destination: Path, manifest_path: Path) -> CoreSnapshotManifest:
    destination = Path(destination).resolve()
    manifest = _load_manifest(Path(manifest_path))
    if manifest.source_repository != SOURCE_REPOSITORY:
        raise CoreSnapshotError("snapshot source repository mismatch")
    actual_files = tuple(sorted(p.name for p in destination.iterdir() if p.is_file())) if destination.is_dir() else ()
    if actual_files != tuple(sorted(manifest.files)):
        raise CoreSnapshotError("snapshot file set does not match manifest")
    actual_hashes = {name: _sha256(destination / name) for name in manifest.files}
    for name, expected in manifest.file_sha256.items():
        if actual_hashes.get(name) != expected:
            raise CoreSnapshotError(f"snapshot digest mismatch: {name}")
    if _tree_digest(actual_hashes) != manifest.tree_digest:
        raise CoreSnapshotError("snapshot tree digest mismatch")
    for name in manifest.files:
        _check_imports(destination / name)
    return manifest

# [nnslr-sync] - START
"""Explicit, read-only remote acquisition of one comma route via SSH/rsync.

Only calling sync_route accesses the network. Complete local files are never
overwritten; rsync's separate partial directory makes interrupted copies resumable.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any

from nnslr_tools.manifest import parse_route_id, sha256_of

DEFAULT_HOST = "comma-remote"
DEFAULT_DATA_ROOT = "/srv/nnslr-data"
REMOTE_ROOT = "/data/media/0/realdata"
_CAMERAS = {"narrow": ("fcamera.hevc",), "wide": ("ecamera.hevc",),
            "both": ("fcamera.hevc", "ecamera.hevc")}
_LOG_NAMES = ("rlog.zst", "rlog", "rlog.bz2")
_SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
_HOST_RE = re.compile(r"(?:[A-Za-z0-9_][A-Za-z0-9_.-]*@)?[A-Za-z0-9_][A-Za-z0-9_.-]*")


class SyncError(ValueError):
    """A local preflight or remote transport failure, safe to show in the CLI."""


def _parse_segments(value: str) -> list[int] | None:
    if value == "all":
        return None
    selected: set[int] = set()
    for part in value.split(","):
        match = re.fullmatch(r"(0|[1-9][0-9]*)(?:-(0|[1-9][0-9]*))?", part.strip())
        if match is None:
            raise SyncError("invalid segments: use all, 0-3, or 0,2,4-6")
        start = int(match[1])
        end = int(match[2]) if match[2] is not None else start
        if end < start:
            raise SyncError("invalid segments: range end precedes start")
        selected.update(range(start, end + 1))
    return sorted(selected)


def _check_destination(root: Path, path: Path) -> None:
    # Check every component, including partial files. A symlink must not redirect
    # rsync into another dataset (even one under the same configured root).
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise SyncError(f"destination_symlink: {current}")
    if root.exists() and not root.is_dir():
        raise SyncError(f"data root is not a directory: {root}")


def _inventory(host: str, route: str, camera_files: tuple[str, ...]) -> dict[int, set[str]]:
    # Route and host have already been validated. The remote command uses only
    # POSIX shell builtins; no remote Python, GNU find or software installation.
    names = " ".join(shlex.quote(name) for name in (*camera_files, *_LOG_NAMES))
    command = (
        f"test -d {shlex.quote(REMOTE_ROOT)} || exit 1; "
        f"for d in {shlex.quote(REMOTE_ROOT + '/' + route + '--')}*; do "
        '[ -d "$d" ] && [ ! -L "$d" ] || continue; '
        'printf "segment\\t%s\\n" "${d##*/}"; '
        f"for name in {names}; do "
        '[ -f "$d/$name" ] && [ ! -L "$d/$name" ] || continue; '
        'printf "file\\t%s\\t%s\\n" "${d##*/}" "$name"; '
        "done; done"
    )
    try:
        proc = subprocess.run([*_SSH, host, command], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired as exc:
        raise SyncError(f"ssh_inventory_timeout: {host}") from exc
    if proc.returncode:
        raise SyncError(f"ssh_inventory_failed: {host} (exit {proc.returncode}): {proc.stderr.strip()}")
    inventory: dict[int, set[str]] = {}
    allowed = set((*camera_files, *_LOG_NAMES))
    for line in proc.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) not in (2, 3):
            raise SyncError(f"invalid_remote_inventory: {line!r}")
        match = re.fullmatch(re.escape(route) + r"--(0|[1-9][0-9]*)", fields[1])
        if match is None:
            raise SyncError(f"invalid_remote_segment: {fields[1]!r}")
        index = int(match[1])
        if fields[0] == "segment" and len(fields) == 2 and index not in inventory:
            inventory[index] = set()
        elif fields[0] == "file" and len(fields) == 3 and index in inventory and fields[2] in allowed:
            inventory[index].add(fields[2])
        else:
            raise SyncError(f"invalid_remote_inventory: {line!r}")
    if not inventory:
        raise SyncError(f"route_not_found: {route} on {host}:{REMOTE_ROOT}")
    return inventory


def _rsync_source(host: str, route: str, segment: int, name: str) -> str:
    return f"{host}:{REMOTE_ROOT}/{route}--{segment}/{name}"


def _copy_file(source: str, destination: Path) -> tuple[str, str | None]:
    transport = ["-e", shlex.join(_SSH)]
    if destination.exists():
        if not destination.is_file():
            return "failed", f"destination_conflict: not a regular file: {destination}"
        # Size/mtime alone cannot establish that an existing recording is the
        # same file. Check content without ever overwriting it or its metadata.
        proc = subprocess.run(
            ["rsync", "--checksum", "--dry-run", "--itemize-changes", *transport,
             "--", source, str(destination)], capture_output=True, text=True,
        )
        if proc.returncode:
            return "failed", f"rsync_check_failed (exit {proc.returncode}): {proc.stderr.strip()}"
        if proc.stdout.strip():
            return "failed", f"destination_conflict: content differs: {destination}"
        return "existing", None

    partial = destination.parent / ".nnslr-partial" / destination.name
    resuming = partial.is_file()
    destination.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["rsync", "--times", "--ignore-existing", "--partial-dir=.nnslr-partial",
         "--progress", *transport, "--", source, str(destination)],
        stdout=sys.stderr,  # progress remains visible; stdout is the JSON report
    )
    if proc.returncode:
        return "failed", f"rsync_transfer_failed (exit {proc.returncode}); rerun the same command to resume"
    if not destination.is_file():
        return "failed", f"rsync_destination_missing: {destination}"
    return ("resumed" if resuming else "copied"), None


def sync_route(
    *, route: str, host: str | None = None, segments: str = "all",
    camera: str = "narrow", data_root: str | Path | None = None, dry_run: bool = False,
) -> dict[str, Any]:
    """Inventory and copy selected camera files plus one full log per segment.

    The route is always explicit. Omitted segments mean only the available
    segments of that route. Dry-run contacts SSH for inventory but writes nothing.
    """
    parse_route_id(route)
    host = host if host is not None else os.environ.get("NNSLR_COMMA_HOST", DEFAULT_HOST)
    if _HOST_RE.fullmatch(host) is None:
        raise SyncError("invalid SSH host: use an alias, hostname, IPv4 address or user@host")
    selected = _parse_segments(segments)
    if camera not in _CAMERAS:
        raise SyncError(f"invalid camera: {camera}")
    root = Path(data_root or os.environ.get("NNSLR_DATA_ROOT") or DEFAULT_DATA_ROOT).expanduser().resolve()
    destination_root = root / "raw" / "routes" / route
    _check_destination(root, destination_root)
    for executable in ("ssh", "rsync"):
        if shutil.which(executable) is None:
            raise SyncError(f"sync_tool_not_found: install {executable} before using sync-routes")

    inventory = _inventory(host, route, _CAMERAS[camera])
    selected = sorted(inventory) if selected is None else selected
    report: dict[str, Any] = {
        "host": host, "route": route, "segments": selected, "camera": camera,
        "data_root": str(root), "destination": str(destination_root), "dry_run": dry_run,
        "missing_segments": [], "missing_files": [], "files": [], "ok": True,
    }
    for index in selected:
        if index not in inventory:
            report["missing_segments"].append(index)
            continue
        available = inventory[index]
        log = next((name for name in _LOG_NAMES if name in available), None)
        for name in (*_CAMERAS[camera], log or "rlog"):
            if name not in available:
                report["missing_files"].append({"segment": index, "name": name})
                continue
            destination = destination_root / str(index) / name
            _check_destination(root, destination)
            _check_destination(root, destination.parent / ".nnslr-partial" / name)
            report["files"].append({
                "segment": index, "name": name, "destination": str(destination),
                "source": _rsync_source(host, route, index, name),
                "status": "planned", "size_bytes": None, "sha256": None,
            })

    report["ok"] = not (report["missing_segments"] or report["missing_files"])
    if dry_run:
        return report
    for row in report["files"]:
        destination = Path(row["destination"])
        try:
            row["status"], error = _copy_file(row["source"], destination)
            if error:
                row["error"] = error
                report["ok"] = False
                break
            row["size_bytes"] = destination.stat().st_size
            row["sha256"] = sha256_of(destination)
        except OSError as exc:
            row.update(status="failed", error=str(exc))
            report["ok"] = False
            break
    return report
# [nnslr-sync] - END

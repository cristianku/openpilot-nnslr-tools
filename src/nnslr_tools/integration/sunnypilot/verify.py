from __future__ import annotations

import fnmatch
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .discover import discover_hooks, load_hook_map
from .git_ops import PreparedRun


@dataclass(frozen=True)
class VerificationCheck:
  name: str
  status: str
  detail: str = ""


@dataclass(frozen=True)
class VerificationReport:
  compatible: bool
  checks: tuple[VerificationCheck, ...]
  changed_paths: tuple[str, ...]


def _git(repo: Path, *args: str) -> str:
  proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
  if proc.returncode != 0:
    raise RuntimeError(proc.stderr.strip() or f"git {' '.join(args)} failed")
  return proc.stdout.strip()


def _read_json(path: Path) -> dict:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise ValueError(f"{path} must contain a JSON object")
  return payload


def _gitlinks(repo: Path) -> dict[str, str]:
  out: dict[str, str] = {}
  for line in _git(repo, "ls-files", "-s").splitlines():
    if not line.startswith("160000 "):
      continue
    meta, path = line.split("\t", 1)
    _, sha, _ = meta.split()
    out[path] = sha
  return out


def verify_source(run: PreparedRun, hook_map_path: Path, protected_path: Path) -> VerificationReport:
  changed = tuple(sorted(
    p for p in _git(run.candidate_dir, "diff", "--name-only", f"{run.resolved_upstream_commit}..HEAD").splitlines() if p
  ))
  protected = _read_json(Path(protected_path))
  checks: list[VerificationCheck] = []

  patterns = protected.get("forbidden_write_paths", [])
  forbidden_hits = sorted({
    path for path in changed
    for pattern in patterns
    if isinstance(pattern, str) and fnmatch.fnmatch(path, pattern)
  })
  checks.append(VerificationCheck(
    "protected_paths",
    "failed" if forbidden_hits else "passed",
    ", ".join(forbidden_hits),
  ))

  hook_map = load_hook_map(Path(hook_map_path))
  candidate_files: dict[str, str] = {}
  for hook in hook_map.hooks:
    path = run.candidate_dir / hook.path
    if path.is_file():
      candidate_files[hook.path] = path.read_text(encoding="utf-8")
  discovery = discover_hooks(candidate_files, hook_map)
  detail = ""
  if not discovery.compatible:
    detail = f"missing_paths={list(discovery.missing_paths)} missing_anchors={list(discovery.missing_anchors)}"
  checks.append(VerificationCheck("hook_discovery", "passed" if discovery.compatible else "failed", detail))

  contracts_ok = True
  contract_failures: list[str] = []
  for contract in protected.get("protected_contracts", []):
    if not isinstance(contract, dict):
      contracts_ok = False
      contract_failures.append("malformed_contract")
      continue
    path = contract.get("path")
    anchors = contract.get("anchors", [])
    if not isinstance(path, str) or not isinstance(anchors, list):
      contracts_ok = False
      contract_failures.append(str(contract.get("id", "unknown")))
      continue
    baseline_file = run.baseline_dir / path
    candidate_file = run.candidate_dir / path
    if not baseline_file.is_file() or not candidate_file.is_file():
      contracts_ok = False
      contract_failures.append(str(contract.get("id", path)))
      continue
    base_text = baseline_file.read_text(encoding="utf-8")
    cand_text = candidate_file.read_text(encoding="utf-8")
    if any(not isinstance(a, str) or a not in base_text or a not in cand_text for a in anchors):
      contracts_ok = False
      contract_failures.append(str(contract.get("id", path)))
  checks.append(VerificationCheck(
    "protected_contracts",
    "passed" if contracts_ok else "failed",
    ", ".join(contract_failures),
  ))

  base_links = _gitlinks(run.baseline_dir)
  candidate_links = _gitlinks(run.candidate_dir)
  gitlinks_ok = base_links == candidate_links
  checks.append(VerificationCheck(
    "gitlinks",
    "passed" if gitlinks_ok else "failed",
    "" if gitlinks_ok else f"baseline={base_links} candidate={candidate_links}",
  ))

  return VerificationReport(
    compatible=all(check.status == "passed" for check in checks),
    checks=tuple(checks),
    changed_paths=changed,
  )

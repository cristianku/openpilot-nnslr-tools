from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from .apply import ApplyError, apply_patchset, load_apply_result
from .git_ops import PrepareError, load_prepared_run, prepare_run
from .manifest import FeatureLockError, load_feature_lock
from .reports import write_report
from .verify import verify_source


EXIT_OK = 0
EXIT_INPUT = 2
EXIT_CONFLICT = 3
EXIT_INCOMPATIBLE = 4
EXIT_TEST_FAILED = 5
EXIT_NOT_ESTABLISHED = 6


def _repo_root() -> Path:
  return Path(__file__).resolve().parents[4]


def _default_lock() -> Path:
  return _repo_root() / "integration" / "sunnypilot" / "feature.lock.json"


def _default_hook_map() -> Path:
  return _repo_root() / "integration" / "sunnypilot" / "hook-map.json"


def _default_protected() -> Path:
  return _repo_root() / "integration" / "sunnypilot" / "protected-surfaces.json"


def _github_clone_url(repository: str) -> str:
  if "/" not in repository:
    raise ValueError(f"repository must be owner/name: {repository!r}")
  return f"https://github.com/{repository}.git"


@contextmanager
def _source_repo(explicit: str | None, repository: str) -> Iterator[Path]:
  if explicit:
    path = Path(explicit).expanduser().resolve()
    if not (path / ".git").exists():
      raise PrepareError(f"repository is not a Git checkout: {path}")
    yield path
    return

  with tempfile.TemporaryDirectory(prefix="nnslr-reapply-source-") as tmp:
    path = Path(tmp) / "repo.git"
    proc = subprocess.run(
      ["git", "clone", "--mirror", "--quiet", _github_clone_url(repository), str(path)],
      capture_output=True,
      text=True,
      check=False,
    )
    if proc.returncode != 0:
      raise PrepareError(proc.stderr.strip() or f"failed to clone {repository}")
    yield path


def _cmd_inspect(args: argparse.Namespace) -> int:
  lock = load_feature_lock(Path(args.feature_lock))
  print(json.dumps({
    "feature_id": lock.feature_id,
    "state": lock.state,
    "sealed": lock.sealed,
    "runtime_repository": lock.runtime_repository,
    "upstream_repository": lock.upstream_repository,
    "upstream_ref": lock.upstream_ref,
    "source_base_commit": lock.source_base_commit,
    "source_feature_commit": lock.source_feature_commit,
    "patch_count": len(lock.patches),
    "capability_profile": lock.capability_profile,
  }, indent=2, sort_keys=True))
  return EXIT_OK


def _cmd_prepare(args: argparse.Namespace) -> int:
  lock = load_feature_lock(Path(args.feature_lock))
  if bool(args.upstream_ref) == bool(args.upstream_sha):
    raise PrepareError("provide exactly one of --upstream-ref or --upstream-sha")
  with _source_repo(args.upstream_repo, lock.upstream_repository) as source:
    run = prepare_run(
      source,
      Path(args.workspace),
      upstream_ref=args.upstream_ref,
      upstream_sha=args.upstream_sha,
    )
  print(json.dumps({
    "run_dir": str(run.run_dir),
    "resolved_upstream_commit": run.resolved_upstream_commit,
    "requested_upstream_ref": run.requested_upstream_ref,
    "requested_upstream_sha": run.requested_upstream_sha,
  }, indent=2, sort_keys=True))
  return EXIT_OK


def _cmd_apply(args: argparse.Namespace) -> int:
  lock = load_feature_lock(Path(args.feature_lock))
  if not lock.sealed:
    raise ApplyError("feature lock must be sealed before apply")
  run = load_prepared_run(Path(args.run_dir))
  with _source_repo(args.feature_repo, lock.runtime_repository) as source:
    result = apply_patchset(run, source, lock)
  print(json.dumps({
    "success": result.success,
    "result_tree": result.result_tree,
    "added_commit_count": result.added_commit_count,
    "resumed": result.resumed,
    "commit_mapping": [asdict(v) for v in result.commit_mapping],
    "conflicts": [asdict(v) for v in result.conflicts],
  }, indent=2, sort_keys=True))
  return EXIT_OK if result.success else EXIT_CONFLICT


def _cmd_verify(args: argparse.Namespace) -> int:
  run = load_prepared_run(Path(args.run_dir))
  report = verify_source(run, Path(args.hook_map), Path(args.protected_surfaces))
  payload = {
    "compatible": report.compatible,
    "changed_paths": list(report.changed_paths),
    "checks": [asdict(v) for v in report.checks],
  }
  path = run.run_dir / "verification.json"
  path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  print(json.dumps(payload, indent=2, sort_keys=True))
  return EXIT_OK if report.compatible else EXIT_INCOMPATIBLE


def _cmd_report(args: argparse.Namespace) -> int:
  run = load_prepared_run(Path(args.run_dir))
  result = load_apply_result(run)
  verification_path = run.run_dir / "verification.json"
  verification = {}
  readiness = "source_applied" if result.success else "conflict"
  if verification_path.is_file():
    payload = json.loads(verification_path.read_text(encoding="utf-8"))
    verification = {row["name"]: row["status"] for row in payload.get("checks", [])}
    if result.success and payload.get("compatible") is True:
      readiness = "source_ready"
  paths = write_report(run, result, readiness_level=readiness, verification=verification)
  print(json.dumps({"json": str(paths.json), "markdown": str(paths.markdown), "readiness_level": readiness}, indent=2))
  return EXIT_OK


def _cmd_seal(args: argparse.Namespace) -> int:
  run = load_prepared_run(Path(args.run_dir))
  result = load_apply_result(run)
  verification_path = run.run_dir / "verification.json"
  if not result.success or not verification_path.is_file():
    raise RuntimeError("source verification is not established")
  verification = json.loads(verification_path.read_text(encoding="utf-8"))
  if verification.get("compatible") is not True:
    raise RuntimeError("source verification is incompatible")
  head = subprocess.check_output(["git", "-C", str(run.candidate_dir), "rev-parse", "HEAD"], text=True).strip()
  tree = subprocess.check_output(["git", "-C", str(run.candidate_dir), "rev-parse", "HEAD^{tree}"], text=True).strip()
  receipt = {
    "schema_version": 1,
    "readiness_level": "source_ready",
    "resolved_upstream_commit": run.resolved_upstream_commit,
    "candidate_commit": head,
    "candidate_tree": tree,
    "deployment_authorized": False,
  }
  path = run.run_dir / "candidate.json"
  path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  print(json.dumps(receipt, indent=2, sort_keys=True))
  return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(prog="nnslr-reapply", description="Rebuild the sealed NNSLR Sunnypilot patchset on a frozen upstream base.")
  sub = p.add_subparsers(dest="command", required=True)

  inspect = sub.add_parser("inspect")
  inspect.add_argument("--feature-lock", default=str(_default_lock()))
  inspect.set_defaults(func=_cmd_inspect)

  prepare = sub.add_parser("prepare")
  prepare.add_argument("--feature-lock", default=str(_default_lock()))
  prepare.add_argument("--workspace", required=True)
  prepare.add_argument("--upstream-repo", help="local Git checkout/mirror; omit to clone upstream_repository")
  group = prepare.add_mutually_exclusive_group(required=True)
  group.add_argument("--upstream-ref")
  group.add_argument("--upstream-sha")
  prepare.set_defaults(func=_cmd_prepare)

  apply = sub.add_parser("apply")
  apply.add_argument("--feature-lock", default=str(_default_lock()))
  apply.add_argument("--run-dir", required=True)
  apply.add_argument("--feature-repo", help="local feature Git checkout/mirror; omit to clone runtime_repository")
  apply.set_defaults(func=_cmd_apply)

  verify = sub.add_parser("verify")
  verify.add_argument("--run-dir", required=True)
  verify.add_argument("--profile", choices=["source"], default="source")
  verify.add_argument("--hook-map", default=str(_default_hook_map()))
  verify.add_argument("--protected-surfaces", default=str(_default_protected()))
  verify.set_defaults(func=_cmd_verify)

  report = sub.add_parser("report")
  report.add_argument("--run-dir", required=True)
  report.set_defaults(func=_cmd_report)

  seal = sub.add_parser("seal")
  seal.add_argument("--run-dir", required=True)
  seal.set_defaults(func=_cmd_seal)
  return p


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  try:
    return int(args.func(args))
  except (FeatureLockError, PrepareError, ApplyError, ValueError) as exc:
    print(f"error: {exc}", file=sys.stderr)
    return EXIT_INPUT
  except RuntimeError as exc:
    print(f"error: {exc}", file=sys.stderr)
    return EXIT_NOT_ESTABLISHED


if __name__ == "__main__":
  raise SystemExit(main())

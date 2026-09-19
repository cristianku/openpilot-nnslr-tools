from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .apply import ApplyResult
from .git_ops import PreparedRun


@dataclass(frozen=True)
class ReportPaths:
    json: Path
    markdown: Path


def write_report(
    run: PreparedRun,
    apply_result: ApplyResult,
    *,
    readiness_level: str,
    verification: Mapping[str, Any],
) -> ReportPaths:
    """Write private machine receipt plus path-sanitized Markdown summary."""
    report_dir = run.run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "reapply-report.json"
    markdown_path = report_dir / "reapply-report.md"

    payload = {
        "schema_version": 1,
        "run_id": run.run_dir.name,
        "requested_upstream_ref": run.requested_upstream_ref,
        "requested_upstream_sha": run.requested_upstream_sha,
        "resolved_upstream_commit": run.resolved_upstream_commit,
        "apply_success": apply_result.success,
        "commit_mapping": [asdict(v) for v in apply_result.commit_mapping],
        "conflicts": [asdict(v) for v in apply_result.conflicts],
        "result_tree": apply_result.result_tree,
        "readiness_level": readiness_level,
        "verification": dict(verification),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# NNSLR Sunnypilot reapply report",
        "",
        f"- Run: `{payload['run_id']}`",
        f"- Frozen upstream: `{run.resolved_upstream_commit}`",
        f"- Apply result: `{'success' if apply_result.success else 'conflict'}`",
        f"- Readiness: `{readiness_level}`",
        f"- Result tree: `{apply_result.result_tree or 'unavailable'}`",
        "",
        "## Patch mapping",
        "",
    ]
    if apply_result.commit_mapping:
        for row in apply_result.commit_mapping:
            lines.append(f"- `{row.logical_id}`: `{row.source_commit[:12]}` -> `{row.result_commit[:12]}` ({row.status})")
    else:
        lines.append("- No commits applied.")

    if apply_result.conflicts:
        lines += ["", "## Conflicts", ""]
        for conflict in apply_result.conflicts:
            lines.append(f"- `{conflict.logical_id}`: **{conflict.kind}**")

    lines += ["", "## Verification", ""]
    if verification:
        for key in sorted(verification):
            lines.append(f"- `{key}`: `{verification[key]}`")
    else:
        lines.append("- No verification profile executed.")

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ReportPaths(json=json_path, markdown=markdown_path)

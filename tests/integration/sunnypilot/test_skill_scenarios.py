from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SKILL = ROOT / ".github" / "skills" / "nnslr-reapply" / "SKILL.md"


def test_skill_points_to_versioned_plan_and_reapply_script() -> None:
  text = SKILL.read_text(encoding="utf-8")
  assert "runtime-reapply-plan.md" in text
  assert "scripts/nnslr_reapply.py" in text
  assert "feature.lock.json" in text


def test_skill_preserves_non_destructive_and_vehicle_boundaries() -> None:
  text = SKILL.read_text(encoding="utf-8").lower()
  for phrase in [
    "do not force-push",
    "do not access the comma device",
    "do not deploy",
    "do not renumber schema",
    "speedlimitresolver",
    "carstatesp.speedlimit",
  ]:
    assert phrase in text


def test_skill_does_not_misinterpret_local_comma_files_as_device_access() -> None:
  text = SKILL.read_text(encoding="utf-8")
  assert "Local comma-origin files are allowed" in text
  assert "live device" in text


def test_skill_requires_verification_receipt_not_clean_apply_only() -> None:
  text = SKILL.read_text(encoding="utf-8").lower()
  assert "clean cherry-pick is not compatibility proof" in text
  assert "report" in text
  assert "verification" in text

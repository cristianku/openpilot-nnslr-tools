# [nnslr-t1] - START
"""Import-isolation tests: the core and the CLI must import cleanly.

The whole point of the T1 reference core (plan §12.1) is that
``speed_vision_core`` and ``nnslr_tools`` import and run with *stdlib only* —
no CUDA, no PyTorch, no tinygrad, no openpilot, no cereal, no vehicle hardware.
A single stray ``import torch`` would defeat a clean-clone CPU quickstart, so
this test enforces the boundary.

The check is run in a fresh subprocess (the authoritative guarantee): the
in-process ``sys.modules`` of the pytest run cannot prove the boundary because
other test files may have imported things first. A brand-new interpreter that
imports only the two packages and then inspects ``sys.modules`` is the real test.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"

#: Top-level modules that must NOT appear in ``sys.modules`` after importing the
#: two NNSLR packages. Matched by exact name or ``<name>.<submodule>``.
BANNED_TOP_LEVEL = (
    "torch", "cuda", "nvidia", "cupy", "pycuda", "triton",
    "openpilot", "cereal", "selfdrive", "sunnypilot", "tinygrad",
    "pandas", "numpy",  # T1 core is deliberately stdlib-only
)

_PROBE = r"""
import sys

import speed_vision_core
import nnslr_tools
import nnslr_tools.cli

banned = {b!r}
hits = sorted(
    m for m in sys.modules
    if any(m == p or m.startswith(p + ".") for p in banned)
)
if hits:
    print("HITS:" + ",".join(hits))
    sys.exit(1)
print("ISOLATION_OK")
""".replace("{b!r}", repr(list(BANNED_TOP_LEVEL)))


def _run_isolated_probe() -> subprocess.CompletedProcess:
    env = dict(os.environ)
    # Make the src/ layout importable in the child without shadowing an
    # installed copy: prepend, and drop any inherited path so the probe is clean.
    env["PYTHONPATH"] = str(SRC)
    return subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


def test_core_and_cli_import_without_heavy_or_vehicle_deps() -> None:
    proc = _run_isolated_probe()
    assert proc.returncode == 0, (
        f"import isolation failed\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "ISOLATION_OK" in proc.stdout
    assert "HITS:" not in proc.stdout


def test_banned_modules_not_imported_in_current_process() -> None:
    """Fast in-process sanity: after importing the packages here, none of the
    banned top-level modules should be present. (Weaker than the subprocess
    probe, but catches a regression cheaply and gives a readable failure.)"""
    import speed_vision_core  # noqa: F401
    import nnslr_tools  # noqa: F401
    import nnslr_tools.cli  # noqa: F401

    hits = [
        m for m in sys.modules
        if any(m == p or m.startswith(p + ".") for p in BANNED_TOP_LEVEL)
    ]
    assert not hits, f"banned modules imported: {sorted(hits)}"


def test_core_module_has_no_torch_attribute() -> None:
    """Even a lazy ``import torch as _x`` would leave a module attribute. The
    core must not reference torch/cuda at all."""
    import speed_vision_core
    import speed_vision_core.types as types

    for mod in (speed_vision_core, types):
        for attr in ("torch", "cuda", "cuda_runtime", "c10"):
            assert not hasattr(mod, attr), f"{mod.__name__} exposes {attr!r}"
# [nnslr-t1] - END

# [nnslr-t1] - START
"""Pytest bootstrap: make the ``src/`` layout importable without installation.

Tests must run both after ``pip install -e .`` and from a bare checkout
(``python -m pytest``). We add ``src/`` to ``sys.path`` only if the packages
are not already importable, so an installed copy is never shadowed.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
try:  # pragma: no cover - trivial importability probe
    import speed_vision_core  # noqa: F401
except ImportError:  # not installed yet -> use the in-tree src layout
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
# [nnslr-t1] - END

# [nnslr-t1] - START
"""Hardware-independent reference core for the NN Vision Speed Limit project.

Importing this package must never pull in CUDA, PyTorch, tinygrad, openpilot,
cereal or any vehicle/network dependency. It is the single source of truth for
the domain types and validity invariants (plan §7); later modules
(`preprocess`, `tracking`, `applicability`, `passage`, `state`, `advisory`)
are added by their owning tasks and import only from :mod:`.types` and the
stdlib.
"""

from . import types

__all__ = ["types"]
__version__ = "0.1.0"
# [nnslr-t1] - END

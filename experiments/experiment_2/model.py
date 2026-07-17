"""Model for experiment 2 — thin re-export of the shared direct-encoding model.

The real implementation lives in ``experiments/shared_direct_model.py`` (the
single source of truth for the direct net, shared with experiment 3 so the EC
control and the GD control train a byte-identical network). This shim exists
only so the existing ``from model import ...`` call sites keep working when run
from this directory. Do NOT put model logic here — edit ``../shared_direct_model.py``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_direct_model import BrainConfig, DirectGenome, role_mask  # noqa: F401

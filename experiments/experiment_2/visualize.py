"""Visualiser for experiment 2 — thin re-export of the shared direct visualiser.

The real implementation lives in ``experiments/shared_direct_viz.py`` (shared
with experiment 3 so every direct-encoding experiment draws identical brain
images). This shim keeps the existing ``from visualize import ...`` call sites
working when run from this directory. Do NOT put drawing logic here — edit
``../shared_direct_viz.py``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_direct_viz import (  # noqa: F401
    brain_stats, visualize_brain, _auto_open,
    INPUT_COLOR, OUTPUT_COLOR, EXC_EDGE, INH_EDGE, HIDDEN_COLOR,
)

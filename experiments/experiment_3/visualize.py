"""Visualiser for experiment 3 — thin re-export of the shared direct visualiser.

Same brain images as experiment 2 (both use the direct-encoding model), so the
GD-trained and CMA-ES-trained networks can be compared side by side. The real
implementation lives in ``experiments/shared_direct_viz.py``; edit it there.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_direct_viz import (  # noqa: F401
    brain_stats, visualize_brain, _auto_open,
    INPUT_COLOR, OUTPUT_COLOR, EXC_EDGE, INH_EDGE, HIDDEN_COLOR,
)

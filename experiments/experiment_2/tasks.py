"""Tasks for experiment 2 — thin re-export of the shared definitions.

The real implementation lives in ``experiments/shared_tasks.py`` (the single
source of truth shared by every experiment). This shim exists only so the
existing ``import tasks`` call sites keep working when run from this directory.
Do NOT put task logic here — edit ``../shared_tasks.py`` instead. This guarantees
experiment 1 and experiment 2 train on byte-identical tasks.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_tasks import *  # noqa: F401,F403  (public API: all_binary_inputs, targets, min_inputs, TASKS, OPS)
from shared_tasks import _and, _or, _xor, _left_feature, _right_feature  # noqa: F401

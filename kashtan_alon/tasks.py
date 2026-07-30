"""Tasks for the Kashtan-Alon recreation — re-export of the shared definitions.

The retina lives in ``experiments/shared_tasks.py`` (the single source of truth
used by experiments 1-3). Reusing it here keeps the Kashtan-Alon reproduction on
a byte-identical retina, so its results are comparable with the rest of the
project. Do NOT put task logic here — edit ``../experiments/shared_tasks.py``.
"""

import os
import sys

# kashtan_alon/ sits next to experiments/; reach into it for the shared tasks.
_EXPERIMENTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments")
sys.path.insert(0, _EXPERIMENTS)

from shared_tasks import *  # noqa: F401,F403  (all_binary_inputs, targets, min_inputs, TASKS, OPS)
from shared_tasks import _and, _or, _xor, _left_feature, _right_feature  # noqa: F401

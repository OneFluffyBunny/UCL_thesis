"""Model for experiment 3 — thin re-export of the shared direct-encoding model.

Experiment 3 optimises the SAME direct-encoding network as experiment 2 (see
``experiments/shared_direct_model.py``); the only difference between the two
experiments is the optimiser (exp 2: CMA-ES, exp 3: gradient descent). This shim
re-exports the shared model so ``from model import ...`` works from this
directory. Do NOT put model logic here — edit ``../shared_direct_model.py``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_direct_model import BrainConfig, DirectGenome, role_mask  # noqa: F401

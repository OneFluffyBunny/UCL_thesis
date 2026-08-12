"""Infomap plumbing -- NOT a metric (see `infomap_io` for why it lives apart).

    from qmetrics.flow import run
    out = run(G)            # G a DiGraph -> flow_model="directed"

Imported on demand, never by `qmetrics/__init__.py`, so the optional `infomap`
dependency cannot break `import qmetrics`.
"""

from .infomap_io import FLOW_MODELS, available, build, run, version

__all__ = ["available", "version", "build", "run", "FLOW_MODELS"]

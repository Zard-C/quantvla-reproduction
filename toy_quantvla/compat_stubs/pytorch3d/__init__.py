"""Minimal import stub used by Phase 3 GR00T smoke tests.

The LIBERO checkpoint metadata does not request rotation representation
conversion, but NVIDIA's transform module imports pytorch3d.transforms at
module import time. This stub lets import-only smoke tests run on Python 3.12
where pipablepytorch3d is unavailable.
"""


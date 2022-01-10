"""
oplib: Onepredict Library

Explain oplib.
"""

import pkg_resources

from . import math, preprocessing, rotary, signal

__version__ = pkg_resources.get_distribution("oplib").version

__all__ = ["math", "preprocessing", "rotary", "signal", "feature"]

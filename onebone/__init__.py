"""
onebone: onepredict base algorithm library
"""

import pkg_resources

from . import feature, math, preprocessing, signal, utils

__version__ = pkg_resources.get_distribution("onebone").version

__all__ = ["math", "preprocessing", "signal", "feature", "utils"]

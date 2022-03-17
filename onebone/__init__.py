"""
onebone: An Open Source Signal Processing Library for Sensor Signals about vibration, current, etc.
"""

import pkg_resources

from . import feature, math, preprocessing, signal, utils

__version__ = pkg_resources.get_distribution("onebone").version

__all__ = ["math", "preprocessing", "signal", "feature", "utils"]

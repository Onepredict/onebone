"""Test time analysis.

- Author: Kyunghwan Kim
- Contact: kyunghwan.kim@onepredict.com
"""

import numpy as np

from oplib.signal import crestfactor, kurtosis, peak2peak, rms


def _generate_sin_signal(fs: float):
    # Generate signal
    t = np.linspace(0, 1, int(fs))
    signal = 2.0 * np.sin(2 * np.pi * 10.0 * t)
    return signal


if __name__ == "__main__":
    x = _generate_sin_signal(5000.0)
    p2p = peak2peak(x)

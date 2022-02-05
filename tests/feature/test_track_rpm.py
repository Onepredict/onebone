"""Test code for track_rpm.py

- Author: Kangwhi Kim
- Contact: kangwhi.kim@onepredict.com
"""

import numpy as np
from numpy.testing import assert_almost_equal

from onebone.feature import track_rpm


def _generate_signal():
    fs = 1e4
    n = 1e5
    time = np.arange(n) / fs
    mod = 500 * np.cos(2 * np.pi * 0.1 * time)
    carrier = 3 * np.sin(2 * np.pi * 3e3 * time + mod)
    signal = carrier + np.random.rand(carrier.size) / 5
    return signal


def _check_track_rpm_output():
    x = _generate_signal()
    fs = 1e4
    f_start = 3e3
    f_tol = 50
    filter_bw = 5
    window = "hann"
    nperseg = 4096
    noverlap = nperseg * 0.8
    # Get the estimated rpm.
    rpm = track_rpm(x, fs, f_start, f_tol, filter_bw, window, nperseg, noverlap)

    # Check the output
    assert_almost_equal(np.mean(rpm), 3e3, decimal=0)


def test_track_rpm():
    _check_track_rpm_output()


if __name__ == "__main__":
    test_track_rpm()

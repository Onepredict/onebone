"""Test code for estimate_if.py

- Author: Kangwhi Kim
- Contact: kangwhi.kim@onepredict.com
"""

import numpy as np
from numpy.testing import assert_almost_equal

from onebone.feature import estimate_if


def _generate_signal():
    fs = 1e4
    n = 1e5
    time = np.arange(n) / fs
    mod = 500 * np.cos(2 * np.pi * 0.1 * time)
    carrier = 3 * np.sin(2 * np.pi * 3e3 * time + mod)
    signal = carrier + np.random.rand(carrier.size) / 5
    return signal


def _check_estimate_if_output():
    x = _generate_signal()
    fs = 1e4
    f_start = 3e3
    f_tol = 50
    filter_bw = 5
    window = "hann"
    nperseg = 4096
    noverlap = 3985
    # Get the estimated inst_freq.
    inst_freq = estimate_if(x, fs, f_start, f_tol, filter_bw, window, nperseg, noverlap)

    # Check the output
    assert_almost_equal(np.mean(inst_freq), 3e3, decimal=0)


def test_estimate_if():
    _check_estimate_if_output()


if __name__ == "__main__":
    test_estimate_if()

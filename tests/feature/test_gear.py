"""Test condition metrics for gear.

- Author: Kyunghwan Kim, Kangwhi Kim
- Contact: kyunghwan.kim@onepredict.com, kangwhi.kim@onepredict.com
"""

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from onebone.feature import na4


def _generate_gear_signal():
    rpm = 180  # Revolution per minute
    fs = 50e3  # Sampling rate
    t = np.arange(0, (1 / 3) - 1 / fs, 1 / fs)  # Sample times
    f = (rpm / 60, 51, 153)  # Frequencies of signals
    # Motor shaft rotation and harmonic
    shaft_signal = np.sin(2 * np.pi * f[0] * t) + np.sin(2 * np.pi * 2 * f[0] * t)
    # Gear mesh vibration and harmonic for a pair of gears
    gm1_signal = 3 * np.sin(2 * np.pi * f[1] * t) + 3 * np.sin(2 * np.pi * 2 * f[1] * t)
    # Gear mesh vibration and harmonic for a pair of gears
    gm2_signal = 4 * np.sin(2 * np.pi * f[2] * t) + 4 * np.sin(2 * np.pi * 2 * f[2] * t)
    # Fault component signal
    fault_signal = 2 * (1 / 6) * np.sin(2 * np.pi * 10 * f[0] * t)
    # New signal is the sum of gm1_signal, gm2_signal, and fault_signal.
    signal = shaft_signal + gm1_signal + gm2_signal + fault_signal

    return signal


def _check_na4_1d_array():
    signal = _generate_gear_signal()
    prev_info = (0, 0)  # The information for the 'previous time record number' in run ensemble.
    fs = 50e3
    rpm = 180
    freq_list = (51, 153)  # Gear mesh frequencies
    n_harmonics = 2
    # Calculate NA4
    na4_, cur_info = na4(signal, prev_info, fs, rpm, freq_list, n_harmonics)
    # Check the output
    assert_almost_equal(na4_, 1.5369827, decimal=7)
    assert_array_almost_equal(cur_info, (1, 0.0560181), decimal=7)


def test_na4():
    _check_na4_1d_array()


if __name__ == "__main__":
    test_na4()

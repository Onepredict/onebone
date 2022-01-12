"""Test condition metrics for gear.

- Author: Kyunghwan Kim, Kangwhi Kim
- Contact: kyunghwan.kim@onepredict.com, kangwhi.kim@onepredict.com
"""

import numpy as np

from oplib.feature import na4

if __name__ == "__main__":
    rpm = 180  # Revolution per minute
    fs = 50e3  # Sampling rate
    t = np.arange(0, (1 / 3) - 1 / fs, 1 / fs)  # Sample times
    freq_list = (51, 153)  # Gear mesh frequencies
    f = (rpm / 60,) + freq_list  # Frequencies of signals
    prev_info = (0, 0)
    n_harmonics = 2

    na4_list = []
    # Assume that the gear condition is getting worse.
    for k in range(1, 11):
        shaft_signal = np.sin(2 * np.pi * f[0] * t) + np.sin(
            2 * np.pi * 2 * f[0] * t
        )  # motor shaft rotation and harmonic
        gm1_signal = 3 * np.sin(2 * np.pi * f[1] * t) + 3 * np.sin(
            2 * np.pi * 2 * f[1] * t
        )  # gear mesh vibration and harmonic for a pair of gears
        gm2_signal = 4 * np.sin(2 * np.pi * f[2] * t) + 4 * np.sin(
            2 * np.pi * 2 * f[2] * t
        )  # gear mesh vibration and harmonic for a pair of gears
        fault_signal = 2 * (k / 6) * np.sin(2 * np.pi * 10 * f[0] * t)  # fault component signal
        new_signal = shaft_signal + gm1_signal + gm2_signal + fault_signal

        na4_, cur_info = na4(new_signal, prev_info, fs, rpm, freq_list, n_harmonics)
        prev_info = cur_info
        na4_list.append(na4_)

    print(na4_list)
""

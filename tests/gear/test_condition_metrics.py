"""Test condition metrics for gear.

- Author: Kyunghwan Kim, Kangwhi Kim
- Contact: kyunghwan.kim@onepredict.com, kangwhi.kim@onepredict.com
"""

import numpy as np

from oplib.gear import na4

if __name__ == "__main__":
    rpm = 180
    fs = 50e3
    t = np.arange(0, (1 / 3) - 1 / fs, 1 / fs)  # sample times
    orderlist = [17.0, 51.0]
    f = rpm / 60 * np.array([1, 17, 51, 10])

    signal_list = []
    for k in range(1, 11):
        signal_list.append(
            # motor shaft rotation and harmonic
            (np.sin(2 * np.pi * f[0] * t) + np.sin(2 * np.pi * 2 * f[0] * t))
            # gear mesh vibration and harmonic for gears 1 and 2
            + (3 * np.sin(2 * np.pi * f[1] * t) + 3 * np.sin(2 * np.pi * 2 * f[1] * t))
            # gear mesh vibration and harmonic for gears 3 and 4
            + (4 * np.sin(2 * np.pi * f[2] * t) + 4 * np.sin(2 * np.pi * 2 * f[2] * t))
            # gear mesh vibration for gears 5 and 6 and noise
            + (2 * (k / 6) * np.sin(2 * np.pi * 10 * f[0] * t) + np.random.randn(t.size) / 5)
        )
        # signal_list.append((2 * (k / 6) * np.sin(2 * np.pi * 10 * f[0] * t)))

    import matplotlib.pyplot as plt
    from scipy.stats import kurtosis, moment

    na4_list, k_list = [], []
    for signal in signal_list:
        # plt.plot(t, signal, alpha=.3)
        # plt.hist(signal, 100, alpha=.3)
        # plt.show()

        # na4_, kurtosis = na4(signal, fs, rpm, orderlist)
        # na4_list.append(na4_)
        # k_list.append(kurtosis)
        v = moment(signal, 2)  # Variance at current time, k
        na4 = moment(signal, 4) / (np.mean(v) ** 2)  # use average variance up to k
        na4_list.append(na4)
        k_list.append(kurtosis(signal, fisher=False))

    print(na4_list)
    print(k_list)
    ""

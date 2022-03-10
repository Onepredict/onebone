"""Test_hampel_filter.

- Author: Sunjin.kim
- Contact: sunjin.kim@onepredict.com
"""

import matplotlib.pyplot as plt
import numpy as np

from onebone.signal import hampel_filter


def timeseries_data():
    t = np.linspace(0, 1, 100)
    y = np.sin(2 * np.pi * 10 * t)
    np.put(y, [9, 13, 24, 30, 45, 51, 78], 4)
    return y


def test_hampel_filter():
    first_feature = hampel_filter.hampel_filter(timeseries_data(), 2)[0]
    second_feature = hampel_filter.hampel_filter(timeseries_data(), 3)[0]
    plt.plot(first_feature)
    plt.plot(second_feature)
    plt.plot(timeseries_data())


if __name__ == "__main__":
    test_hampel_filter()

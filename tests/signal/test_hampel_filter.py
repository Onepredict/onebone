"""Test_hampel_filter.

- Author: Sunjin.kim
- Contact: sunjin.kim@onepredict.com
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal

from onebone.signal import hampel_filter


def timeseries_data(outlier: bool = True) -> np.ndarray:

    """
    make example timeseries
    You can set the number of random outliers.

    Parameters
    ----------
    num_outlier : bool
        You can generate/degenerate timeseries data with example outliers.
        default = True
    """

    t = np.linspace(0, 1, 1000)
    y = np.sin(2 * np.pi * 10 * t)
    np.put(y, [13, 124, 330, 445, 651, 775, 978], 10)
    if outlier is False:
        t = np.linspace(0, 1, 1000)
        y = np.sin(2 * np.pi * 10 * t)
    return y


def test_hampel_filter(is_plot: bool = False):
    window_size = 3

    origin_data = timeseries_data(outlier=False)
    noisy_data = timeseries_data(outlier=True)
    filtered_data = hampel_filter.hampel_filter(noisy_data, window_size)[0]

    check_window_region = [window_size, len(noisy_data) - window_size]

    assert_array_almost_equal(
        filtered_data[check_window_region[0] : check_window_region[1]],
        origin_data[check_window_region[0] : check_window_region[1]],
        decimal=0,
    )

    if is_plot is True:

        plt.plot(noisy_data)
        plt.show()

        plt.plot(filtered_data)
        plt.show()

        plt.plot(origin_data)
        plt.show()


if __name__ == "__main__":
    test_hampel_filter(is_plot=False)

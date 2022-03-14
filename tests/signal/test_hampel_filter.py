"""Test_hampel_filter.

- Author: Sunjin.kim
- Contact: sunjin.kim@onepredict.com
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal

from onebone.signal import hampel_filter


def timeseries_data(num_outlier: int = 10) -> np.ndarray:

    """
    make example timeseries
    You can set the number of random outliers.

    Parameters
    ----------
    num_outlier : numpy.ndarray
        You can set the number of random outliers.
        default = 10
    """
    t = np.linspace(0, 1, 1000)
    y = np.sin(2 * np.pi * 10 * t)
    make_outlier = np.random.randint(0, 1000, size=num_outlier)
    np.put(y, [make_outlier], 10)

    return y


def test_hampel_filter(is_plot: bool = False):
    window_size = 4

    origin_data = timeseries_data(num_outlier=0)
    noisy_data = timeseries_data(num_outlier=10)
    filtered_data = hampel_filter.hampel_filter(noisy_data, window_size)[0]

    check_window_region = [window_size, len(noisy_data) - window_size]

    assert_array_almost_equal(
        filtered_data[check_window_region[0] : check_window_region[1]],
        origin_data[check_window_region[0] : check_window_region[1]],
        decimal=0,
    )

    if is_plot:

        plt.plot(noisy_data)
        plt.show()
        plt.close()

        plt.plot(filtered_data)
        plt.show()
        plt.close()

        plt.plot(origin_data)
        plt.show()
        plt.close()


if __name__ == "__main__":
    test_hampel_filter(is_plot=True)

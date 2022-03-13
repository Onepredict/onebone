"""Test_hampel_filter.

- Author: Sunjin.kim
- Contact: sunjin.kim@onepredict.com
"""

import numpy as np
from numpy.testing import assert_array_almost_equal

from onebone.signal import hampel_filter


def timeseries_data(num_outlier: int = 4) -> np.ndarray:

    """
    make example timeseries
    You can set the number of random outliers.

    Parameters
    ----------
    num_outlier : numpy.ndarray
        You can set the number of random outliers.
        default = 4
    """
    t = np.linspace(0, 1, 100)
    y = np.sin(2 * np.pi * 10 * t)
    make_outlier = np.random.randint(0, 100, size=num_outlier)
    np.put(y, [make_outlier], 4)

    return y


def test_hampel_filter(is_plot: bool = False):
    filtered_data = hampel_filter.hampel_filter(timeseries_data(), 2)[0]
    origin_data = timeseries_data()
    perfect_data = timeseries_data(num_outlier=0)

    assert_array_almost_equal(filtered_data, perfect_data, decimal=0)

    if is_plot:
        import matplotlib.pyplot as plt

        plt.plot(origin_data)
        plt.show()
        plt.close()

        plt.plot(filtered_data)
        plt.show()
        plt.close()

        plt.plot(perfect_data)
        plt.show()
        plt.close()


if __name__ == "__main__":
    test_hampel_filter(is_plot=True)

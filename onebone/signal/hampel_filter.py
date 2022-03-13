"""hampel_filter.
- Author: Sunjin Kim
- Contact: sunjin.kim@onepredict.com
"""
from typing import Tuple

import numpy as np


def hampel_filter(
    series: np.ndarray, window_size: int, n_sigmas: float = 3, autowindow: bool = False
) -> Tuple[np.ndarray, list]:

    """
    A hampel filter removes outliers.
    Estimate the median and standard deviation of each sample using
    MAD(Median Absolute Deviation) in the window range set by the user.
    If the MAD > 3 * sigma condition is satisfied,
    the value is replaced with the median value.

    Parameters
    ----------
    x : numpy.ndarray
        1d-timeseries data
    window_size : int,
        Lenght of the sliding window.
        Only integer types are available,
        and the window size must be adjusted according to your data.
    n_sigma : float, defalut=3
        Coefficient of standard deviation.
    auto_window : boolean, defalut=False
        If set to True, the user does not need to modify the window size multiple times.

    Returns
    ----------
    filtered_series : numpy.ndarray
        A value from which Outlier or NaN has been removed by the filter.
    index : list
        Returns the index corresponding to Outlier.

    References
    ----------
    [1] Pearson, Ronald K., et al. "Generalized hampel filters."
    EURASIP Journal on Advances in Signal Processing 2016.1 (2016): 1-18.
    DOI: 10.1186/s13634-016-0383-6
    [2] https://github.com/MichaelisTrofficus/hampel_filter

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> fs = 100.0
    >>> t = np.linspace(0, 1, int(fs))
    >>> y = np.sin(2 * np.pi *10.0 * t)
    >>> np.put(y,[9, 13, 24, 30, 45, 51,78],4)
    >>> first_hampel_filter_array = hampel_filter.hampel_filter(y,2)[0]
    >>> second_hampel_filter_array = hampel_filter.hampel_filter(y,3)[0]

    .. image:: https://bit.ly/3t55Dlc #nopa
        :width: 300

    """

    filtered_series = series.copy()
    k = 1.4826  # scale factor for Gaussian distribution
    # The factor 1.4826 makes the MAD scale estimate an unbiased estimate
    # of the standard deviation for Gaussian data.

    # Check inputs
    if not isinstance(series, np.ndarray):
        raise TypeError("'series' must be np.ndarray")
    if not isinstance(window_size, int):
        raise TypeError("'window_size' must be integer")
    if not isinstance(n_sigmas, (int, float)):
        raise TypeError("'n_sigmas' must be int or float")
    if not isinstance(autowindow, bool):
        raise TypeError("'autowindow' must be boolean")

    index = []

    if autowindow is False:
        for i in range((window_size), (len(filtered_series) - window_size)):
            real_window_size = filtered_series[(i - window_size) : (i + window_size)]

            window_median = np.median(real_window_size)
            S_k = k * np.median(np.abs(real_window_size - window_median))

            """
            MAD scale estimate, defined as : S_k = 1.4826 * median_j∈-K,K]{|x_(k-j) - m_k|}.
            """

            if np.abs(filtered_series[i] - window_median) > n_sigmas * S_k:
                filtered_series[i] = window_median
                index.append(i)
        return filtered_series, index

    elif autowindow is True:
        # window size를 지정 및 가장 유사한 graph로 후보군을 정리해주는 코드 구현 하고싶음
        raise AttributeError("We are preparing to provide a function")

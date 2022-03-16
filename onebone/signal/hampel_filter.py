"""hampel_filter.
- Author: Sunjin Kim
- Contact: sunjin.kim@onepredict.com
"""
from typing import Tuple

import numpy as np


def hampel_filter(x: np.ndarray, window_size: int, n_sigmas: float = 3) -> Tuple[np.ndarray, list]:

    """
    A hampel filter removes outliers.
    Estimate the median and standard deviation of each sample using
    MAD(Median Absolute Deviation) in the window range set by the user.
    If the MAD > 3 * sigma condition is satisfied,
    the value is replaced with the median value.

    .. math::
        m_{i} = median(x[window_size])
        \\\\
        mad_i = median(|x_{i}-k-m_{i}|,...,|x_{i}+k-m_{i}|})
        \\\\
        k_mad = k * mad_i
    Where :math: `m_i` is Local median,
    :math: `mad_i` is Median absolute deviation.
           the residuals (deviations) from the data's median.
    :math: k_mad is the MAD may be used similarly to how one would use the deviation for the average.
           In order to use the MAD as a consistent estimator
           for the estimation of the standard deviation `{\sigma}`,one takes `k * mad_i`
    :math: `k` is a constant scale factor, which depends on the distribution.
           For normally distributed data `k` is taken to be `k` = 1.4826


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
        :width: 600

    """

    k = 1.4826  # scale factor for Gaussian distribution
    # The factor 1.4826 makes the MAD scale estimate an unbiased estimate
    # of the standard deviation for Gaussian data.

    # Check inputs
    if not isinstance(x, np.ndarray):
        raise TypeError("'series' must be np.ndarray")
    if not isinstance(window_size, int):
        raise TypeError("'window_size' must be integer")
    if not isinstance(n_sigmas, (int, float)):
        raise TypeError("'n_sigmas' must be int or float")

    copy_series = x.copy()
    oulier_index = []

    ## define sliding window
    indexer = np.arange(window_size)[None, :] + np.arange(len(copy_series) - window_size)[:, None]

    ## define window median
    window_median = np.median(copy_series[indexer], axis=1)
    window_median_array = np.repeat(window_median, window_size, axis=0).reshape(
        np.shape(copy_series[indexer])[0], window_size
    )

    ## get mad * k
    k_mad = k * np.median(np.abs(copy_series[indexer] - window_median_array), axis=1)

    ## get comparative value
    value = np.abs(copy_series[indexer][:, 0] - window_median)

    filtered_series = np.where(value > n_sigmas * k_mad, window_median, copy_series[indexer][:, 0])

    ## get index
    oulier_index = np.where(value <= n_sigmas * k_mad, None, indexer[:, 0])
    oulier_index = list(filter(None, oulier_index))

    return filtered_series, oulier_index

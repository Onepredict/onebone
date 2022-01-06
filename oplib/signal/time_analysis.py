"""Signal analysis for the time domain.

- Author: Kyunghwan Kim
- Contact: kyunghwan.kim@onepredict.com
"""

from typing import Union

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis


def peak2peak(x, axis: int = None) -> np.ndarray:
    """
    .. note:: This method uses `numpy.ptp`_ method as it is.
    .. _numpy.p2p: \
        https://numpy.org/doc/stable/reference/generated/numpy.ptp.html?highlight=ptp#

    Maximum to minimum difference along an axis.

    Parameters
    ----------
    x : array_like
        The data.
    axis : None or int, default=None
        Axis along which to find the peaks. By default, flatten the array.

    Returns
    -------
    p2p : numpy.ndarray
        The difference between the maximum and minimum values in x.

    Examples
    --------
    >>> x = np.array([[4, 9, 2, 10],
    ...               [6, 9, 7, 12]])
    >>> np.ptp(x, axis=1)
    array([8, 6])
    >>> np.ptp(x, axis=0)
    array([2, 0, 5, 2])
    >>> np.ptp(x)
    10
    """
    return np.ptp(x, axis=axis)


def rms(x: np.ndarray, axis: int = None) -> np.float64:
    """
    Root mean square along an axis.

    Parameters
    ----------
    x : array_like
        The data.
    axis : None or int, default=None
        Axis along which to find the peaks. By default, flatten the array.

    Returns
    -------

    Examples
    --------

    """
    return np.sqrt(np.mean(x ** 2, axis=axis))


def crestfactor(x: np.ndarray) -> np.float64:
    """
    Parameters
    ----------

    Returns
    -------

    Examples
    --------

    """
    p2p = peak2peak(x)
    rms_ = rms(x)
    if rms_ != 0:
        crest_factor = p2p / rms_
    else:
        crest_factor = 0.0
    return crest_factor


def kurtosis(x: np.ndarray) -> np.float64:
    """
    Parameters
    ----------

    Returns
    -------

    Examples
    --------

    """
    return np.float64(scipy_kurtosis(x))

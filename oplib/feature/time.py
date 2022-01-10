"""Signal analysis for the time domain.

- Author: Kyunghwan Kim
- Contact: kyunghwan.kim@onepredict.com
"""

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
    >>> ptp(x, axis=1)
    array([8, 6])
    >>> ptp(x, axis=0)
    array([2, 0, 5, 2])
    >>> ptp(x)
    10
    """
    return np.ptp(x, axis=axis)


def rms(x: np.ndarray, axis: int = None) -> np.ndarray:
    """
    Root mean square along an axis.

    Parameters
    ----------
    x : numpy.ndarray
        The data.
    axis : None or int, default=None
        Axis along which to calculate rms. By default, flatten the array.

    Returns
    -------
    rms : numpy.ndarray
        Root mean square value of x.

    Examples
    --------
    >>> x = np.array([[4, 9, 2, 10],
    ...               [6, 9, 7, 12]])
    >>> rms(x, axis=0)
    array([ 5.09901951,  9.        ,  5.14781507, 11.04536102])
    >>> rms(x, axis=1)
    array([7.08872344, 8.80340843])
    >>> rms(x)
    7.99218368157289
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Argument 'x' must be of type numpy.ndarray, not {type(x)}")

    return np.sqrt(np.mean(x ** 2, axis=axis))


def crestfactor(x: np.ndarray, axis: int = None) -> np.ndarray:
    """
    Peak to average ratio along an axis.

    Parameters
    ----------
    x : numpy.ndarray
        The data.
    axis : None or int, default=None
        Axis along which to calculate crestfactor. By default, flatten the array.

    Returns
    -------
    crestfactor : numpy.ndarray
        Peak to average ratio of x.

    Examples
    --------
    >>> x = np.array([[4, 9, 2, 10],
    ...               [6, 9, 7, 12]])
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Argument 'x' must be of type numpy.ndarray, not {type(x)}")

    p2p = peak2peak(x, axis=axis)
    rms_ = rms(x, axis=axis) + 1e-6
    crest_factor = p2p / rms_
    return crest_factor


def kurtosis(x: np.ndarray) -> np.float64:
    """
    .. note:: This method uses `scipy.stats.kurtosis`_ method as it is.
    .. _scipy.stats.kurtosis: \
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html

    Compute the kurtosis (Fisher or Pearson) of a signal.

    Parameters
    ----------

    Returns
    -------

    Examples
    --------

    """
    return np.float64(scipy_kurtosis(x))

"""Signal analysis for the time domain.

- Author: Kyunghwan Kim
- Contact: kyunghwan.kim@onepredict.com
"""

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis


def peak2peak(x, axis: int = None) -> np.ndarray:
    """
    .. note:: This method uses `numpy.ptp`_ method as it is.
    .. _numpy.ptp: \
        https://numpy.org/doc/stable/reference/generated/numpy.ptp.html

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
    if not (isinstance(axis, int) | (axis is None)):
        raise TypeError(f"Argument 'axis' must be of type int or None, not {type(axis).__name__}")

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
        raise TypeError(f"Argument 'x' must be of type numpy.ndarray, not {type(x).__name__}")
    if not (isinstance(axis, int) | (axis is None)):
        raise TypeError(f"Argument 'axis' must be of type int or None, not {type(axis).__name__}")

    return np.sqrt(np.mean(x**2, axis=axis))


def crest_factor(x: np.ndarray, axis: int = None) -> np.ndarray:
    """
    Peak to average ratio along an axis.

    .. math:: crest factor = {|x_{peak}| \over {x_{rms}}}

    Parameters
    ----------
    x : numpy.ndarray
        The data.
    axis : None or int, default=None
        Axis along which to calculate crest factor. By default, flatten the array.

    Returns
    -------
    crest_factor : numpy.ndarray
        Peak to average ratio of x.

    Examples
    --------
    >>> x = np.array([[4, 9, 2, 10],
    ...               [6, 9, 7, 12]])
    >>> crest_factor(x, axis=0)
    array([0.39223219, 0.        , 0.97128567, 0.18107148])
    >>> crest_factor(x, axis=1)
    array([1.12855283, 0.68155412])
    >>> crest_factor(x)
    1.2512223376239555
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Argument 'x' must be of type numpy.ndarray, not {type(x).__name__}")
    if not (isinstance(axis, int) | (axis is None)):
        raise TypeError(f"Argument 'axis' must be of type int or None, not {type(axis).__name__}")

    p2p = peak2peak(x, axis=axis)
    rms_ = rms(x, axis=axis) + 1e-6
    crest_factor = p2p / rms_
    return crest_factor


def kurtosis(x: np.ndarray, axis: int = 0, fisher: bool = True, bias: bool = True) -> np.ndarray:
    """
    .. note:: This method uses `scipy.stats.kurtosis`_ method as it is.
    .. _scipy.stats.kurtosis: \
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html

    Compute the kurtosis (Fisher or Pearson) of a signal.

    Parameters
    ----------
    x : numpy.ndarray
        The data.
    axis : None or int, default=0
        Axis along which the kurtosis is calculated. If None, compute over the whole array a.

    fisher : bool, default=True
        If True, Fisher’s definition is used (normal ==> 0.0).
        If False, Pearson’s definition is used (normal ==> 3.0).

    bias : bool, default=True
    If False, then the calculations are corrected for statistical bias.

    Returns
    -------
    kurtosis : numpy.ndarray
        The kurtosis of values along an axis.
        If all values are equal, return -3 for Fisher’s definition and 0 for Pearson’s definition.

    Examples
    --------
    >>> x = np.array([[4, 9, 2, 10],
    ...               [6, 9, 7, 12]])
    >>> kurtosis(x)
    array([-2., -3., -2., -2.])
    """
    return scipy_kurtosis(x, axis, fisher, bias)

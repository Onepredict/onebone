"""Data scaling methods.

- Author: Kyunghwan Kim
- Contact: kyunghwan.kim@onepredict.com
"""

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike
from sklearn.preprocessing import minmax_scale, scale


def minmax_scaling(
    x: ArrayLike, feature_range: Tuple[int, int] = (0, 1), axis: int = 0
) -> np.ndarray:
    """
    .. note:: This method uses `sklearn.preprocessing.minmax_scale`_ method as it is.
    .. _sklearn.preprocessing.minmax_scale: \
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.minmax_scale.html

    Transform features by scaling each feature to a given range.

    .. math:: x' = {(x - x_{min}) \over (x_{max} - x_{min})}

    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        The data.
    feature_range : tuple (min, max).
        Desired range of transformed data. Default is (0, 1)
    axis : int
        Axis used to scale along. Default is 0.

    Returns
    -------
    x_tr : numpy.ndarray of shape (n_samples, n_features)
        The transformed data.

    Examples
    --------
    >>> a = list(range(9))
    >>> a
    [0, 1, 2, 3, 4, 5, 6, 7, 8]
    >>> minmax_scaling(a)
    array([0.   , 0.125, 0.25 , 0.375, 0.5  , 0.625, 0.75 , 0.875, 1.   ])
    """
    return minmax_scale(x, feature_range=feature_range, axis=axis)


def zscore_scaling(x: ArrayLike, axis: int = 0):
    """
    .. note:: This method uses `sklearn.preprocessing.scale`_ method as it is.
    .. _sklearn.preprocessing.scale: \
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html

    Transform input data so that they can be described as a normal distribution.

    Notes
    -----
    .. math:: x' = {(x - x_{mean}) \over x_{std}}


    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        The data.
    axis : int
        Axis used to compute the means and standard deviations along.

    Returns
    -------
    x_tr : numpy.ndarray of shape (n_samples, n_features)
        The transformed data.

    Examples
    --------
    >>> a = list(range(9))
    >>> a
    [0, 1, 2, 3, 4, 5, 6, 7, 8]
    >>> zscore_scaling(a)
    array([-1.54919334, -1.161895  , -0.77459667, -0.38729833,  0.        ,
            0.38729833,  0.77459667,  1.161895  ,  1.54919334])
    """
    return scale(x, axis=axis)

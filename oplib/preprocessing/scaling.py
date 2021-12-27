"""Data scaling methods.

- Author: Kyunghwan Kim
- Contact: kyunghwan.kim@onepredict.com
"""

from typing import Tuple

import numpy as np
from sklearn.preprocessing import minmax_scale, scale


def minmax_scaling(x: np.ndarray, feature_range: Tuple[int, int] = (0, 1)):
    """
    .. note:: This method uses `sklearn.preprocessing.minmax_scale`_ method as it is.
    .. _sklearn.preprocessing.minmax_scale: \
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.minmax_scale.html

    Transform features by scaling each feature to a given range.

    .. math:: x' = {(x - x_{min}) \over (x_{max} - x_{min})}

    Parameters
    ----------

    Returns
    -------

    Examples
    --------

    """
    return minmax_scale(x, feature_range)


def zscore_scaling(x: np.ndarray):
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

    Returns
    -------

    Examples
    --------
    """
    return scale(x)

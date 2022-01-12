"""Frequency domain feature.

- Author: Kangwhi Kim
- Contact: kangwhi.kim@onepredict.com
"""

from typing import Union

import numpy as np


def integrate_trapezoid(y, x=None, dx=1.0, axis=-1) -> Union[float, np.ndarray]:
    """
    .. note:: This method uses `numpy.trapz`_ method as it is.
    .. _numpy.trapz: https://numpy.org/doc/stable/reference/generated/numpy.trapz.html

    Integrate along the given axis using the composite trapezoidal rule.

    If `x` is provided, the integration happens in sequence along its
    elements - they are not sorted.

    Integrate `y` (`x`) along each 1d slice on the given axis, compute
    :math:`\int y(x) dx`.
    When `x` is specified, this integrates along the parametric curve,
    computing :math:`\int_t y(t) dt =
    \int_t y(t) \left.\\frac{dx}{dt}\\right|_{x=x(t)} dt`.

    Parameters
    ----------
    y : array_like
        Input array to integrate.
    x : array_like, optional
        The sample points corresponding to the `y` values. If `x` is None,
        the sample points are assumed to be evenly spaced `dx` apart. The
        default is None.
    dx : scalar, optional
        The spacing between sample points when `x` is None. The default is 1.
    axis : int, optional
        The axis along which to integrate.

    Returns
    -------
    trapz : float or ndarray
        Definite integral of 'y' = n-dimensional array as approximated along
        a single axis by the trapezoidal rule. If 'y' is a 1-dimensional array,
        then the result is a float. If 'n' is greater than 1, then the result
        is an 'n-1' dimensional array.

    References
    ----------
    .. [1] Wikipedia page: https://en.wikipedia.org/wiki/Trapezoidal_rule

    .. [2] Illustration image:
           https://en.wikipedia.org/wiki/File:Composite_trapezoidal_rule_illustration.png

    Examples
    --------
    >>> np.trapz([1,2,3])
    4.0
    >>> np.trapz([1,2,3], x=[4,6,8])
    8.0
    >>> np.trapz([1,2,3], dx=2)
    8.0

    Using a decreasing `x` corresponds to integrating in reverse:

    >>> np.trapz([1,2,3], x=[8,6,4])
    -8.0

    More generally `x` is used to integrate along a parametric curve.
    This finds the area of a circle, noting we repeat the sample which closes
    the curve:

    >>> theta = np.linspace(0, 2 * np.pi, num=1000, endpoint=True)
    >>> np.trapz(np.cos(theta), x=np.sin(theta))
    3.141571941375841

    >>> a = np.arange(6).reshape(2, 3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.trapz(a, axis=0)
    array([1.5, 2.5, 3.5])
    >>> np.trapz(a, axis=1)
    array([2.,  8.])
    """
    return np.trapz(y, x, dx, axis)

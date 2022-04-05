"""A Moving Average (MA) which returns the weighted average of array.

- Author: Kibum Park
- Contact: kibum.park@onepredict.com
"""

from typing import Union

import numpy as np
from scipy.signal import convolve


def _moving_average_1d(
    signal_1d: np.ndarray, window_size: Union[int, float], pad: bool, weights: np.ndarray
) -> np.ndarray:
    """Weighted moving average of 1d signal.
    This method is used for numpy.apply_along axis() in moving_average().
    """
    pad_size = window_size - 1

    if pad:
        convolve_1d = convolve(signal_1d, weights)[:-pad_size]
        div_arr = np.concatenate(
            [np.cumsum(weights), np.repeat(np.sum(weights), signal_1d.shape[0] - window_size)],
            axis=0,
        )
        moving_average_1d = convolve_1d / div_arr

    else:
        convolve_1d = convolve(signal_1d, weights)[pad_size:-pad_size]
        div_arr = np.repeat(np.sum(weights), signal_1d.shape[0] - window_size + 1)
        moving_average_1d = convolve_1d / div_arr

    return moving_average_1d


def _check_moving_average_args(
    signal: np.ndarray,
    pad: bool,
    window_size: Union[int, float],
    weights: np.ndarray,
    axis: Union[int, float],
) -> None:
    """Check the validity of the input parameters for moving average."""
    if not isinstance(signal, np.ndarray):
        raise TypeError(f"signal must be numpy.ndarray, not {type(signal).__name__}")
    if not (isinstance(window_size, int) or isinstance(window_size, float)):
        raise TypeError(
            f"Argument 'window_size' must be int or float, not {type(window_size).__name__}."
        )
    if not isinstance(pad, bool):
        raise TypeError(f"Argument 'pad' must be bool, not {type(pad).__name__}.")
    if not isinstance(weights, (np.ndarray, type(None))):
        raise TypeError(
            f"Argument 'weights' must be numpy.ndarray or None, not {type(weights).__name__}."
        )
    if not isinstance(axis, int) or isinstance(axis, float):
        raise TypeError(f"Argument 'axis' must be int or float, not {type(axis).__name__}.")

    if len(signal.shape) == 1:
        if signal.shape[0] < window_size:
            raise ValueError("Length of signal must be greater than window_size.")
        elif np.abs(axis) > 1:
            raise ValueError("Axis must be smaller than signal dimmesion.")
    elif len(signal.shape) == 2:
        if signal.shape[1] < window_size:
            raise ValueError("Length of signal must be greater than window_size.")
        elif np.abs(axis) > 2:
            raise ValueError("Axis must be smaller than signal dimmesion.")
    elif len(signal.shape) > 2:
        raise ValueError("Dimension of signal must be less than 3.")

    if weights is not None:
        if weights.shape[0] != window_size:
            raise ValueError(f"Length of weights must be {window_size}.")
        elif len(weights.shape) >= 2:
            raise ValueError("Dimension of weights must be less than 2.")
        elif weights[0] == 0:
            raise ValueError("First element of weights must be non-zero. Remove first 0 element.")


def moving_average(
    signal: np.ndarray,
    window_size: Union[int, float],
    pad: bool = False,
    weights: Union[np.ndarray, None] = None,
    axis: Union[int, float] = -1,
) -> np.ndarray:
    """
    Weighted moving average.
    .. math:: WMA(x, w, t, n) = \\sum_{i=n-t+1}^{n} w_i x_i,
    where :math:`x` is the input array,
    :math:`w_i` is the weight of the :math:`i`-th element,
    :math:`t` is the window size,
    :math:`n` is the :math:`n`th value of the input array,
    if pad is True and :math: `n` is smaller than :math:`t`, :math:`i` is set to :math:`0`.

    Parameters
    ----------
    signal : numpy.ndarray of shape (signal_length,), (n, signal_length,)
        Original time-domain signal.
    window_size : Union[int, float], optional, default=10
        Window size.
        One of `window_size`,`weights` must be specified.
    pad : bool, default=False
        Padding method.
        If True, Pads with the edge values of array is added. So the shape of output is same as `signal`.
    weights : Union[numpy.ndarray of shape (window_size,), None], optional, default=None
        Weighting coefficients.
        If None, the `weights` are uniform.
        One of `window_size`,`weights` must be specified.
    axis : Union[int, float], optional, default=-1
        The axis of the input data array along which to apply the moving average.

    Returns
    -------
    ma : numpy.ndarray
        Moving average signal.
        If input shape is [signal_length,], output shape is [signal_length,].
        If input shape is [n, signal_length,] and axis is 1, output shape is [n, signal_length,].
        If input shape is [signal_length, n] and axis is 0, output shape is [signal_length, n].
        If pad is False, output shape is [signal_length - window_size + 1,].
        If pad is True, output shape is [signal_length,].

    Examples
    --------
    >>> signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> window_size = 3
    >>> moving_average(signal, window_size)
    [2, 3, 4, 5, 6, 7, 8, 9]

    >>> signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> window_size = 3
    >>> moving_average(signal, window_size, pad=True)
    [1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]

    >>> signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> window_size = 3
    >>> weights = np.array([1, 2, 3])
    >>> moving_average(signal, window_size, weights=weights)
    [2.33333333, 3.33333333, 4.33333333, 5.33333333, 6.33333333, 7.33333333, 8.33333333, 9.33333333]

    """
    _check_moving_average_args(signal, pad, window_size, weights, axis)

    if window_size is None:
        window_size = weights.shape[0]

    if weights is None:
        weights = np.ones(window_size)
    weights = weights[::-1]

    if len(signal.shape) == 1:
        signal = np.atleast_2d(signal)

    if (axis == 0) or (axis == -2):
        signal = signal.T

    ma = np.apply_along_axis(_moving_average_1d, 1, signal, *(window_size, pad, weights))

    if (len(signal.shape) == 2) and ((axis == 0) or (axis == -2)):
        ma = ma.T

    if signal.shape[0] == 1:
        ma = ma.squeeze()

    return ma

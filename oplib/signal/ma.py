"""A Moving Average (MA) which returns the weighted average of array.

- Author: Kibum Park
- Contact: kibum.park@onepredict.com
"""

from typing import Union

import numpy as np


def moving_average(
    signal: np.ndarray,
    window_size: Union[int, float],
    pad: bool = False,
    weights: Union[np.ndarray, None] = None,
) -> np.ndarray:
    """
    Weighted moving average.
    .. math:: WMA(x, n, w) = \\sum_{i=0}^{n-1} w_i x_i

    Parameters
    ----------
    signal : numpy.ndarray of shape (signal_length,), (n, signal_length,)
        Original time-domain signal.
    window_size : Union[int, float]
        Window size.
    pad : bool, default=False
        Padding method.
        If True, Pads with the edge values of array is added. So the shape of output is same as `signal`.
    weights : Union[numpy.ndarray of shape (window_size,), None], optional, default=None
        Weighting coefficients. If None, the `weights` are uniform.

    Returns
    -------
    ma : numpy.ndarray
        Moving average signal.
        If input shape is [signal_length,], output shape is [signal_length,].
        If input shape is [n, signal_length,], output shape is [n, signal_length,].
        If mode is 'valid', output shape is [signal_length - window_size + 1,].
        If mode is 'same', output shape is [signal_length,].

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
    if (len(signal.shape) == 1) and (signal.shape[0] < window_size):
        raise ValueError("Length of signal must be greater than window_size.")
    elif (len(signal.shape) == 2) and (signal.shape[1] < window_size):
        raise ValueError("Length of signal must be greater than window_size.")
    if len(signal.shape) > 2:
        raise ValueError("Dimension of signal must be less than 3.")
    if weights is not None:
        if weights.shape[0] != window_size:
            raise ValueError(f"Length of weights must be {window_size}.")
        if len(weights.shape) >= 2:
            raise ValueError("Dimension of weights must be less than 2.")
        if weights[0] == 0:
            raise ValueError("First element of weights must be non-zero. Remove first 0 element.")

    if weights is None:
        weights = np.ones(window_size)
    weights = weights[::-1]

    if len(signal.shape) == 1:
        signal = np.atleast_2d(signal)

    def _ma_1d(signal_1d: np.ndarray) -> np.ndarray:
        nonlocal weights, pad, window_size
        pad_size = window_size - 1

        if pad:
            convolve_1d = np.convolve(signal_1d, weights, mode="full")[:-pad_size]
            div_arr = np.concatenate(
                [np.cumsum(weights), np.repeat(np.sum(weights), signal_1d.shape[0] - window_size)],
                axis=0,
            )
            ma_1d = convolve_1d / div_arr

        else:
            convolve_1d = np.convolve(signal_1d, weights, mode="full")[pad_size:-pad_size]
            div_arr = np.repeat(np.sum(weights), signal_1d.shape[0] - window_size + 1)
            ma_1d = convolve_1d / div_arr

        return ma_1d

    ma = np.apply_along_axis(_ma_1d, axis=1, arr=signal)
    if signal.shape[0] == 1:
        ma = ma.squeeze()

    return ma

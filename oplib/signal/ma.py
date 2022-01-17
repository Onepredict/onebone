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
    [1, 1.33333333, 2, 3, 4, 5, 6, 7, 8, 9]

    >>> signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> window_size = 3
    >>> weights = np.array([0, 1, 2])
    >>> moving_average(signal, window_size, weights=weights)
    [2.66666667, 3.66666667, 4.66666667, 5.66666667, 6.66666667, 7.66666667, 8.66666667, 9.66666667]

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
    if len(signal.shape) == 1:
        if signal.shape[0] < window_size:
            raise ValueError("Length of signal must be greater than window_size.")
    elif len(signal.shape) == 2:
        if signal.shape[1] < window_size:
            raise ValueError("Length of signal must be greater than window_size.")
    if len(signal.shape) > 2:
        raise ValueError("Dimension of signal must be less than 3.")
    if weights is not None:
        if weights.shape[0] != window_size:
            raise ValueError(f"Length of weights must be {window_size}.")
        if len(weights.shape) >= 2:
            raise ValueError("Dimension of weights must be less than 2.")

    if weights is None:
        weights = np.ones(window_size) / window_size
    else:
        weights = weights / np.sum(weights)
    weights = weights[::-1]

    if len(signal.shape) == 1:
        signal = np.atleast_2d(signal)

    def _ma_1d(signal_1d: np.ndarray) -> np.ndarray:
        if pad:
            signal_1d = np.pad(signal_1d, (window_size - 1, 0), mode="edge")
        ma_1d = np.convolve(signal_1d, weights, mode="valid")

        return ma_1d

    ma = np.apply_along_axis(_ma_1d, axis=1, arr=signal)
    if signal.shape[0] == 1:
        ma = ma.squeeze()

    return ma


if __name__ == "__main__":
    signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    window_size = 3
    weights = np.array([0, 1, 2])
    print(moving_average(signal, window_size, pad=True))

from typing import Tuple, Union

import numpy as np
from numpy.fft import fft, fftfreq


def _index_along_axis(x: np.ndarray, s: slice, axis: int):
    """Index under certain conditions along the axis you specify."""
    x = x.copy()  # shallow copy
    if axis == -1:
        lower_ndim, upper_ndim = len(x.shape[:axis]), 0
    else:
        lower_ndim, upper_ndim = len(x.shape[:axis]), len(x.shape[axis + 1 :])
    indices = (
        lower_ndim
        * np.s_[
            :,
        ]
        + (s,)
        + upper_ndim
        * np.s_[
            :,
        ]
    )
    x = x[indices]
    return x


def positive_fft(
    signal: np.ndarray,
    fs: Union[int, float],
    hann: bool = False,
    normalization: bool = False,
    axis: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Positive fourier transformation.

    Parameters
    ------------
    signal : numpy.ndarray of shape (signal_length,), (n, signal_length)
        Original time-domain signal
    fs : Union[int, float]
        Sampling rate
    hann : bool, default = False
        hann function used to perform Hann smoothing. It is implemented when hann is True
    normalization : bool, default = False
        Normalization after Fourier transform
    axis : int, default=-1
        The axis of the input data array along which to apply the fourier Transformation.

    Returns
    -------
    output :
    f : numpy.ndarray
        frequency(N-D) expressed in Hz.
    x_mag : numpy.ndarray
        magnitude(1-D) expressed in scala.

    If input shape is [signal_length,], output shape is
    f = [signal_length,], x_mag = [signal_length,].
    If input shape is [n, signal_length,], output shape is
    f = [signal_length,], x_mag = [n, signal_length,].

    Examples
    --------
    >>> N = 400  # array length
    >>> fs = 800  # Sampling frequency
    >>> T = 1 / Fs  # Sample interval time
    >>> x = np.linspace(0.0, N * T, N, endpoint=False) # time
    >>> y = 3 * np.sin(50.0 * 2.0 * np.pi * x) + 2 * np.sin(80.0 * 2.0 * np.pi * x)
    >>> signal = y
    >>> f, x_mag = positive_fft(signal, fs,  hann = false, normalization = false, axix = -1)
    >>> freq = np.around(f[np.where(mag > 1)])
    >>> freq
    [50., 80.]
    """

    if len(signal.shape) > 2:
        raise ValueError("Dimension of signal must be less than 3")

    if hann is True:
        signal = signal * np.hanning(signal.shape[axis])

    x = fft(signal, axis=axis)
    x_half = _index_along_axis(x, np.s_[: signal.shape[axis] // 2], axis=axis)
    n = signal.shape[axis]
    f = fftfreq(n, d=1 / fs)
    mid = int(n / 2)
    f = f[:mid]

    # nomalization
    if normalization is True:
        x_mag = np.abs(x_half) / (n / 2)
    else:
        x_mag = np.abs(x_half)

    return f, x_mag

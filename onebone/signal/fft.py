"""The module about fast fourier transform.

- Author: Daeyeop Na, Kangwhi Kim
- Contact: daeyeop.na@onepredict.com, kangwhi.kim@onepredict.com
"""

from typing import Tuple, Union

import numpy as np

from onebone.utils import slice_along_axis


def positive_fft(
    signal: np.ndarray,
    fs: Union[int, float],
    hann: bool = False,
    normalization: bool = False,
    axis: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Positive 1D fourier transformation.

    Parameters
    ------------
    signal : numpy.ndarray
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
    freq : numpy.ndarray
        frequency
        If input shape is [signal_length,], output shape is freq = [signal_length,].
        If input shape is [n, signal_length,], output shape is freq = [signal_length,].
    mag : numpy.ndarray
        magnitude
        If input shape is [signal_length,], output shape is mag = [signal_length,].
        If input shape is [n, signal_length,], output shape is mag = [n, signal_length,].

    Examples
    --------
    >>> n = 400  # array length
    >>> fs = 800  # Sampling frequency
    >>> t = 1 / fs  # Sample interval time
    >>> x = np.linspace(0.0, n * t, n, endpoint=False) # time
    >>> y = 3 * np.sin(50.0 * 2.0 * np.pi * x) + 2 * np.sin(80.0 * 2.0 * np.pi * x)
    >>> signal = y
    >>> freq, mag = positive_fft(signal, fs,  hann = False, normalization = False, axis = -1)
    >>> freq = np.around(freq[np.where(mag > 1)])
    >>> freq
    [50., 80.]
    """

    if hann is True:
        signal = signal * np.hanning(signal.shape[axis])

    n = signal.shape[axis]
    freq = np.fft.fftfreq(n, d=1 / fs)
    freq = np.abs(freq[: n // 2])

    fft_x = np.fft.fft(signal, axis=axis)
    fft_x_half = slice_along_axis(fft_x, np.s_[: n // 2], axis=axis)

    # Normalization
    if normalization is True:
        mag = np.abs(fft_x_half) / n * 2
    else:
        mag = np.abs(fft_x_half)

    return freq, mag

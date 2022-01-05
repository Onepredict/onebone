from typing import Union

import numpy as np
from numpy.fft import fft, fftfreq


def positive_fft(
    signal: np.ndarray,
    fs: Union[int, float],
    hann: bool,
    normalization: bool,
    axis: int = -1,
):

    """
    positive Fourier Transformation
    hanning window

    Parameters
    ------------
    signal : numpy.ndarray of shape (signal_length,), (n, signal_length)
        Original time-domain signal
    hann: bool[True, False]
        hanningwindow
    fs: Union[int, float]
        Sampling rate
    normalization: bool[True, False]
        Normalization after Fourier transform

    Returns
    -------
    output:
    f: numpy.ndarray
        frequency(N-D) expressed in Hz.
    x_mag: numpy.ndarray
        magnitude(1-D) expressed in scala.

        If input shape is [signal_length,], output shape is
        f = [signal_length,], x_mag = [signal_length,].
        If input shape is [n, signal_length,], output shape is
        f = [n, signal_length,], x_mag = [n, signal_length,].
    Examples
    --------
    >>> N = 600 # array length
    >>> fs = 8000  # Sampling frequency
    >>> T = 1 / Fs  # Sample interval time
    >>> x = np.linspace(0.0, N * T, N, endpoint=False) # time
    >>> y = 3 * np.sin(50.0 * 2.0 * np.pi * x) + 2 * np.sin(80.0 * 2.0 * np.pi * x)
    """

    if len(signal.shape) > 2:
        raise ValueError("Dimension of signal must be less than 3")

    if hann is True:
        signal = signal * np.hanning(signal.shape[0])

    x = fft(signal)
    n = signal.shape[0]
    f = fftfreq(n, d=1 / fs)
    mid = int(f.shape[0] / 2)
    f = f[:mid]

    # nomalisation
    if normalization is True:
        x_mag = np.abs(x[:mid]) / (n / 2)
    else:
        x_mag = np.abs(x[:mid])

    return f, x_mag

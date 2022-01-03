from typing import Union

import numpy as np
from numpy.fft import fft, fftfreq

Fs = 2000  # Sampling frequency
T = 1 / Fs  # Sample interval time
te = 0.5  # End of time
t = np.arange(0, te, T)  # Time vector

# Sum of a 50 Hz sinusoid and a 120 Hz sinusoid
noise = np.random.normal(0, 0.05, len(t))
x = 0.6 * np.cos(2 * np.pi * 60 * t + np.pi / 2) + np.cos(2 * np.pi * 120 * t)
y = x + noise


def positiv_fft(signal: np.ndarray, hann: bool, fs: Union[int, float]):

    """
    Fourier Transformation

    Parameters
    ------------
    signal: numpy.ndarray [signal_length,], [n, signal_length]
        Original time-domain signal
    hann: bool[True, False]
        hanningwindow
    fs: Union[int, float]
        Sampling rate

    Returns
    -------
    out: numpy.adarray
        If input shape is [signal_length,], output shape is [signal_length,].
        If input shape is [1, signal_length], output shape is [signal_length,].

    Examples
    --------
    >>> fs = 2000  # Sampling frequency
    >>> T = 1/Fs # Sample interval time
    >>> te= 0.5  # End of time
    >>> t = np.arange(0, te, T)    # Time vector

    >>> noise = np.random.normal(0,0.05,len(t))
    >>> x = 0.6*np.cos(2*np.pi*60*t+np.pi/2) + np.cos(2*np.pi*120*t)
    >>> y=x+noise

    """
    # [signal_length,1], [1, signal_length] -> [signal]
    signal = np.squeeze(signal)

    if len(signal.shape) > 1:
        raise ValueError("Dimension of signal must be less than 2")

    if hann is True:
        signal = signal * np.hanning(signal.shape[0])

    X = fft(signal)
    N = signal.shape[0]
    f = fftfreq(N, d=1 / fs)
    mid = int(f.shape[0] / 2)

    f = f[:mid]
    X_mag = np.abs(X[:mid])

    return f, X_mag

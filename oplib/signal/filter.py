"""A frequency filter to leave only a specific frequency band.

- Author: Kyunghwan Kim
- Contact: kyunghwan.kim@onepredict.io
"""

import numpy as np
from scipy.signal import butter, lfilter


def lowpass_filter(signal: np.ndarray, fs: float, cutoff: float, order: int = 5) -> np.ndarray:
    """
    Butterworth low-pass filter.

    Parameters
    ----------
    signal: np.ndarray [signal_length,]
        Original time-domain signal.
    fs: float
        Sampling rate.
    cutoff: float
        Cutoff frequency.
    order: int
        Order of butterworth filter.

    Returns
    -------
    out: np.ndarray [signal_length,]
        Filtered signal.

    Examples
    --------
    >>> fs = 5000.0
    >>> t = np.linspace(0, 1, int(fs))
    >>> signal = 10.0 * np.sin(2 * np.pi * 20.0 * t)
    >>> signal += signal += 5.0 * np.sin(2 * np.pi * 100.0 * t)

    >>> freq_x = np.fft.rfftfreq(signal.size, 1 / fs)[:-1]
    >>> origin_fft_mag = abs((np.fft.rfft(signal) / signal.size)[:-1] * 2)
    >>> origin_freq = freq_x[np.where(origin_fft_mag > 0.5)]
    >>> origin_freq
    [ 20. 100.]

    >>> filtered_signal = lowpass_filter(signal, fs, cutoff=50)
    >>> filtered_fft_mag = abs((np.fft.rfft(filtered_signal) / signal.size)[:-1] * 2)
    >>> filtered_freq = freq_x[np.where(filtered_fft_mag > 0.5)]
    >>> filtered_freq
    [ 20.]
    >>>
    """
    if len(signal.shape) > 1:
        raise Exception("Shape of signal is more than 1. Signal must be 1d array.")

    nyq = 0.5 * fs
    cutoff = cutoff / nyq
    b, a = butter(order, cutoff, btype="low")
    signal = lfilter(b, a, signal)
    return signal


def highpass_filter(
    signal: np.ndarray, sample_freq: float, cutoff: float, order: int = 5
) -> np.ndarray:
    if len(signal.shape) > 1:
        raise Exception("Shape of signal is more than 1. Signal must be 1d array.")

    nyq = 0.5 * sample_freq
    cutoff = cutoff / nyq
    b, a = butter(order, cutoff, btype="high")
    signal = lfilter(b, a, signal)
    return signal


def bandpass_filter(
    signal: np.ndarray,
    sample_freq: float,
    l_cutoff: float,
    h_cutoff: float,
    order: int = 5,
) -> np.ndarray:
    if len(signal.shape) > 1:
        raise Exception("Shape of signal is more than 1. Signal must be 1d array.")

    nyq = 0.5 * sample_freq
    low = l_cutoff / nyq
    high = h_cutoff / nyq
    b, a = butter(order, [low, high], btype="bandpass")
    signal = lfilter(b, a, signal)
    return signal


def bandstop_filter(
    signal: np.ndarray,
    sample_freq: float,
    l_cutoff: float,
    h_cutoff: float,
    order: int = 5,
) -> np.ndarray:
    if len(signal.shape) > 1:
        raise Exception("Shape of signal is more than 1. Signal must be 1d array.")

    nyq = 0.5 * sample_freq
    low = l_cutoff / nyq
    high = h_cutoff / nyq
    b, a = butter(order, [low, high], btype="bandstop")
    signal = lfilter(b, a, signal)
    return signal

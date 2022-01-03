"""A frequency filter to leave only a specific frequency band.

- Author: Kyunghwan Kim
- Contact: kyunghwan.kim@onepredict.com
"""

from typing import Union

import numpy as np
from scipy.signal import butter, lfilter


def lowpass_filter(
    signal: np.ndarray,
    fs: Union[int, float],
    cutoff: Union[int, float],
    order: int = 5,
    axis: int = -1,
) -> np.ndarray:
    """
    Butterworth lowpass filter.

    Parameters
    ----------
    signal : numpy.ndarray of shape (signal_length,), (n, signal_length,)
        Original time-domain signal.
    fs : Union[int, float]
        Sampling rate.
    cutoff : Union[int, float]
        Cutoff frequency.
    order : int, default=5
        Order of butterworth filter.
    axis : int, default=-1
        The axis of the input data array along which to apply the linear filter.

    Returns
    -------
    out : numpy.ndarray of shape (signal_length,), (n, signal_length,)
        Filtered signal.
        If input shape is [signal_length,], output shape is [signal_length,].
        If input shape is [n, signal_length,], output shape is [n, signal_length,].

    Examples
    --------
    >>> fs = 5000.0
    >>> t = np.linspace(0, 1, int(fs))
    >>> signal = 10.0 * np.sin(2 * np.pi * 20.0 * t)
    >>> signal += 5.0 * np.sin(2 * np.pi * 100.0 * t)

    >>> freq_x = np.fft.rfftfreq(signal.size, 1 / fs)[:-1]
    >>> origin_fft_mag = abs((np.fft.rfft(signal) / signal.size)[:-1] * 2)
    >>> origin_freq = freq_x[np.where(origin_fft_mag > 0.5)]
    >>> origin_freq
    [ 20., 100.]

    >>> filtered_signal = highpass_filter(signal, fs, cutoff=50)
    >>> filtered_fft_mag = abs((np.fft.rfft(filtered_signal) / signal.size)[:-1] * 2)
    >>> filtered_freq = freq_x[np.where(filtered_fft_mag > 0.5)]
    >>> filtered_freq
    [ 20.]
    """
    if not (isinstance(fs, int) | isinstance(fs, float)):
        raise TypeError(f"Argument 'fs' must be of type int or float, not {type(fs)}")
    if not (isinstance(cutoff, int) | isinstance(cutoff, float)):
        raise TypeError(f"Argument 'cutoff' must be of type int or float, not {type(cutoff)}")
    if not isinstance(order, int):
        raise TypeError(f"Argument 'order' must be of type int or float, not {type(order)}")
    if not isinstance(axis, int):
        raise TypeError(f"Argument 'axis' must be of type int or float, not {type(axis)}")

    if len(signal.shape) > 2:
        raise ValueError("Dimension of signal must be less than 3.")

    nyq = 0.5 * float(fs)
    cutoff = cutoff / nyq
    b, a = butter(order, cutoff, btype="low")
    signal = lfilter(b, a, signal, axis=axis)
    return signal


def highpass_filter(
    signal: np.ndarray,
    fs: Union[int, float],
    cutoff: Union[int, float],
    order: int = 5,
    axis: int = -1,
) -> np.ndarray:
    """
    Butterworth highpass filter.

    Parameters
    ----------
    signal : numpy.ndarray of shape (signal_length,), (n, signal_length,)
        Original time-domain signal.
    fs : Union[int, float]
        Sampling rate.
    cutoff : Union[int, float]
        Cutoff frequency.
    order : int, default=5
        Order of butterworth filter.
    axis : int, default=-1
        The axis of the input data array along which to apply the linear filter.

    Returns
    -------
    out : numpy.ndarray of shape (signal_length,), (n, signal_length,)
        Filtered signal.
        If input shape is [signal_length,], output shape is [signal_length,].
        If input shape is [n, signal_length,], output shape is [n, signal_length,].

    Examples
    --------
    >>> fs = 5000.0
    >>> t = np.linspace(0, 1, int(fs))
    >>> signal = 10.0 * np.sin(2 * np.pi * 20.0 * t)
    >>> signal += 5.0 * np.sin(2 * np.pi * 100.0 * t)

    >>> freq_x = np.fft.rfftfreq(signal.size, 1 / fs)[:-1]
    >>> origin_fft_mag = abs((np.fft.rfft(signal) / signal.size)[:-1] * 2)
    >>> origin_freq = freq_x[np.where(origin_fft_mag > 0.5)]
    >>> origin_freq
    [ 20., 100.]

    >>> filtered_signal = lowpass_filter(signal, fs, cutoff=50)
    >>> filtered_fft_mag = abs((np.fft.rfft(filtered_signal) / signal.size)[:-1] * 2)
    >>> filtered_freq = freq_x[np.where(filtered_fft_mag > 0.5)]
    >>> filtered_freq
    [ 100.]
    """
    if not (isinstance(fs, int) | isinstance(fs, float)):
        raise TypeError(f"Argument 'fs' must be of type int or float, not {type(fs)}")
    if not (isinstance(cutoff, int) | isinstance(cutoff, float)):
        raise TypeError(f"Argument 'cutoff' must be of type int or float, not {type(cutoff)}")
    if not isinstance(order, int):
        raise TypeError(f"Argument 'order' must be of type int or float, not {type(order)}")
    if not isinstance(axis, int):
        raise TypeError(f"Argument 'axis' must be of type int or float, not {type(axis)}")

    if len(signal.shape) > 2:
        raise ValueError("Dimension of signal must be less than 3.")

    nyq = 0.5 * fs
    cutoff = cutoff / nyq
    b, a = butter(order, cutoff, btype="high")
    signal = lfilter(b, a, signal, axis=axis)
    return signal


def bandpass_filter(
    signal: np.ndarray,
    fs: Union[int, float],
    l_cutoff: Union[int, float],
    h_cutoff: Union[int, float],
    order: int = 5,
    axis: int = -1,
) -> np.ndarray:
    """
    Butterworth bandpass filter.

    Parameters
    ----------
    signal : numpy.ndarray of shape (signal_length,), (n, signal_length,)
        Original time-domain signal.
    fs : Union[int, float]
        Sampling rate.
    l_cutoff : Union[int, float]
        Low cutoff frequency.
    h_cutoff : Union[int, float]
        High cutoff frequency.
    order : int, default=5
        Order of butterworth filter.
    axis : int, default=-1
        The axis of the input data array along which to apply the linear filter.

    Returns
    -------
    out : numpy.ndarray of shape (signal_length,), (n, signal_length,)
        Filtered signal.
        If input shape is [signal_length,], output shape is [signal_length,].
        If input shape is [n, signal_length,], output shape is [n, signal_length,].

    Examples
    --------
    >>> fs = 5000.0
    >>> t = np.linspace(0, 1, int(fs))
    >>> signal = 10.0 * np.sin(2 * np.pi * 20.0 * t)
    >>> signal += 5.0 * np.sin(2 * np.pi * 100.0 * t)
    >>> signal += 5.0 * np.sin(2 * np.pi * 500.0 * t)

    >>> freq_x = np.fft.rfftfreq(signal.size, 1 / fs)[:-1]
    >>> origin_fft_mag = abs((np.fft.rfft(signal) / signal.size)[:-1] * 2)
    >>> origin_freq = freq_x[np.where(origin_fft_mag > 0.5)]
    >>> origin_freq
    [ 20., 100., 500.]

    >>> filtered_signal = bandpass_filter(signal, fs, l_cutoff=50, h_cutoff=300)
    >>> filtered_fft_mag = abs((np.fft.rfft(filtered_signal) / signal.size)[:-1] * 2)
    >>> filtered_freq = freq_x[np.where(filtered_fft_mag > 0.5)]
    >>> filtered_freq
    [ 100.]
    """
    if not (isinstance(fs, int) | isinstance(fs, float)):
        raise TypeError(f"Argument 'fs' must be of type int or float, not {type(fs)}")
    if not (isinstance(l_cutoff, int) | isinstance(l_cutoff, float)):
        raise TypeError(f"Argument 'l_cutoff' must be of type int or float, not {type(l_cutoff)}")
    if not (isinstance(h_cutoff, int) | isinstance(h_cutoff, float)):
        raise TypeError(f"Argument 'h_cutoff' must be of type int or float, not {type(h_cutoff)}")
    if not isinstance(order, int):
        raise TypeError(f"Argument 'order' must be of type int or float, not {type(order)}")
    if not isinstance(axis, int):
        raise TypeError(f"Argument 'axis' must be of type int or float, not {type(axis)}")

    if len(signal.shape) > 2:
        raise ValueError("Dimension of signal must be less than 3.")

    nyq = 0.5 * fs
    low = l_cutoff / nyq
    high = h_cutoff / nyq
    b, a = butter(order, [low, high], btype="bandpass")
    signal = lfilter(b, a, signal, axis=axis)
    return signal


def bandstop_filter(
    signal: np.ndarray,
    fs: Union[int, float],
    l_cutoff: Union[int, float],
    h_cutoff: Union[int, float],
    order: int = 5,
    axis: int = -1,
) -> np.ndarray:
    """
    Butterworth bandstop filter.

    Parameters
    ----------
    signal : numpy.ndarray of shape (signal_length,), (n, signal_length,)
        Original time-domain signal.
    fs : Union[int, float]
        Sampling rate.
    l_cutoff : Union[int, float]
        Low cutoff frequency.
    h_cutoff : Union[int, float]
        High cutoff frequency.
    order : int, default=5
        Order of butterworth filter.
    axis : int, default=-1
        The axis of the input data array along which to apply the linear filter.

    Returns
    -------
    out : numpy.ndarray of shape (signal_length,), (n, signal_length,)
        Filtered signal.
        If input shape is [signal_length,], output shape is [signal_length,].
        If input shape is [n, signal_length,], output shape is [n, signal_length,].

    Examples
    --------
    >>> fs = 5000.0
    >>> t = np.linspace(0, 1, int(fs))
    >>> signal = 10.0 * np.sin(2 * np.pi * 20.0 * t)
    >>> signal += 5.0 * np.sin(2 * np.pi * 100.0 * t)
    >>> signal += 5.0 * np.sin(2 * np.pi * 500.0 * t)

    >>> freq_x = np.fft.rfftfreq(signal.size, 1 / fs)[:-1]
    >>> origin_fft_mag = abs((np.fft.rfft(signal) / signal.size)[:-1] * 2)
    >>> origin_freq = freq_x[np.where(origin_fft_mag > 0.5)]
    >>> origin_freq
    [ 20., 100., 500.]

    >>> filtered_signal = bandstop_filter(signal, fs, l_cutoff=50, h_cutoff=300)
    >>> filtered_fft_mag = abs((np.fft.rfft(filtered_signal) / signal.size)[:-1] * 2)
    >>> filtered_freq = freq_x[np.where(filtered_fft_mag > 0.5)]
    >>> filtered_freq
    [ 20., 500.]
    """
    if not (isinstance(fs, int) | isinstance(fs, float)):
        raise TypeError(f"Argument 'fs' must be of type int or float, not {type(fs)}")
    if not (isinstance(l_cutoff, int) | isinstance(l_cutoff, float)):
        raise TypeError(f"Argument 'l_cutoff' must be of type int or float, not {type(l_cutoff)}")
    if not (isinstance(h_cutoff, int) | isinstance(h_cutoff, float)):
        raise TypeError(f"Argument 'h_cutoff' must be of type int or float, not {type(h_cutoff)}")
    if not isinstance(order, int):
        raise TypeError(f"Argument 'order' must be of type int or float, not {type(order)}")
    if not isinstance(axis, int):
        raise TypeError(f"Argument 'axis' must be of type int or float, not {type(axis)}")

    if len(signal.shape) > 2:
        raise ValueError("Dimension of signal must be less than 3.")

    nyq = 0.5 * fs
    low = l_cutoff / nyq
    high = h_cutoff / nyq
    b, a = butter(order, [low, high], btype="bandstop")
    signal = lfilter(b, a, signal, axis=axis)
    return signal

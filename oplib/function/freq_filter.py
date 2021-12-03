import numpy as np
from scipy.signal import butter, lfilter


def lowpass_filter(
    signal: np.ndarray, sample_freq: float, cutoff: float, order: float = 5
) -> np.ndarray:
    if len(signal.shape) > 1:
        raise Exception("Shape of signal is more than 1. Signal must be 1d array.")

    nyq = 0.5 * sample_freq
    cutoff = cutoff / nyq
    b, a = butter(order, cutoff, btype="low")
    signal = lfilter(b, a, signal)
    return signal


def highpass_filter(
    signal: np.ndarray, sample_freq: float, cutoff: float, order: float = 5
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
    order: float = 5,
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

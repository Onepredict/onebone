from typing import Union

import numpy as np
from scipy import signal


def snr(
    x: np.ndarray, fs: Union[int, float] = 1000, nperseg: int = 256, noverlap: int = 32
) -> np.ndarray:
    """
    Extract the SNR(Signal-to-Noise-Ratio) feature from the signal using the 'STFT'.
    SNR is the ratio between max power intensity of frequency and power of other frequencies
    at time t in the STFT spectrum.

    .. math::
        P_{signal}(t) = max(|STFT(t,f)|)

    .. math::
        SNR(t) = {P_{signal}(t) \over {\sum_{f}|STFT(t,f)|} - P_{signal}(t)}

    Parameters
    ----------
    x : numpy.ndarray
        1d-signal data. Must be real.
    fs : int or float
        Sampling rate.
    nperseg : int, default=256
        Length of each segment.
    noverlap : int, default=32
        Number of points to overlap between segments.

    Returns
    ----------
    snr : numpy.ndarray
        SNR of the `x`, 1d-array.

    Examples
    --------
    >>> fs = 1000.0
    >>> t = np.linspace(0, 1, int(fs))
    >>> x = 10.0 * np.sin(2 * np.pi * 20.0 * t)
    >>> snr_array = snr_feature(x, fs)

    Notes
    --------
    1. Get a snr array by using snr_feature for one of given signals.
    2. Make the segments of snr array and get the mean of each segment.
    3. Compare the mean of SNR between normal states and fault states. \
        e.g. SNR_fault = np.mean(SNR_fault_array), SNR_normal = np.mean(SNR_normal_array)
    4. Typically, SNR_fault is smaller than SNR_normal.
    """

    if np.iscomplexobj(x):
        raise ValueError("`x` must be real.")

    _, _, power = signal.spectrogram(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    p = np.max(power, axis=0)  # max power intensity over the entire frequency range at time point t
    res = np.sum(power, axis=0) - p  # noise
    snr = p / res
    return snr

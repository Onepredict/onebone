import numpy as np
from scipy import signal


def snr_feature(x: np.ndarray, fs: float) -> np.ndarray:
    """
    Extract the SNR(Signal-to-Noise-Ratio) feature from the signal using the 'STFT'.
    SNR is the ratio between max power intensity of frequency and power of other frequencies 
    at time t in the STFT spetrum.

    Parameters
    ----------
    x : array_like
        Signal data. Must be real.
    fs : float
        Sampling rate.

    Returns
    ----------
    snr : numpy.ndarray
        SNR of the `x`, 1d-array

    Examples
    --------
    >>> fs = 1000.0
    >>> t = np.linspace(0, 1, int(fs))
    >>> x = 10.0 * np.sin(2 * np.pi * 20.0 * t)
    >>> snr_array = snr_feature(x, fs)

    Usage
    --------
    1. Get a snr array by using snr_feature for one of given signals
    2. Get a mean of the snr array
    3. Compare the mean of SNR between normal states and fault states
    e.g. SNR_fault = np.mean(SNR_fault_array), SNR_normal = np.mean(SNR_normal_array)
    4. Typically, SNR_fault is smaller than SNR_normal.
    """
    if np.iscomplexobj(x):
        raise ValueError("`x` must be real.")

    f, t, Sxx = signal.spectrogram(x, fs=fs, nperseg=64, noverlap=8)
    P = np.max(Sxx, axis=0)  # max power intensity over the entire frequency range at time point t
    Res = np.sum(Sxx, axis=0) - P  # noise
    snr = P / Res
    return snr

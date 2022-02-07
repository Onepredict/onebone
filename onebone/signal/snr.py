import numpy as np
from scipy import signal


def snr_feature(x: np.ndarray, fs: float, nperseg: int = 256, noverlap: int = 32) -> np.ndarray:
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
    nperseg : int, optional
        Length of each segment. Default value is 256
    noverlap : int, optional
        Number of points to overlap between segments. Default value is 32

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
    2. Make the segments of snr array and get the mean of each segment.
    3. Compare the mean of SNR between normal states and fault states
    e.g. SNR_fault = np.mean(SNR_fault_array), SNR_normal = np.mean(SNR_normal_array)
    4. Typically, SNR_fault is smaller than SNR_normal.
    """
    if np.iscomplexobj(x):
        raise ValueError("`x` must be real.")

    f, t, Sxx = signal.spectrogram(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    P = np.max(Sxx, axis=0)  # max power intensity over the entire frequency range at time point t
    Res = np.sum(Sxx, axis=0) - P  # noise
    snr = P / Res
    return snr

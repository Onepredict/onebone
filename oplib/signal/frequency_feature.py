"""Frequency domain feature.

- Author: Kangwhi Kim
- Contact: kangwhi.kim@onepredict.com
"""

from typing import Tuple

import numpy as np
from scipy.fft import fft, fftfreq


def _get_amp_and_freq(
    x: np.ndarray, fs: float = None, freq_range: Tuple = None
) -> Tuple[np.ndarray, np.ndarray]:
    # Do FFT.
    amp = fft(x)
    n = amp.size

    # Set 'fs' for normalized mean frequency.
    if fs is None:
        fs = 1
    # Return the FFT sample frequencies.
    freq = fftfreq(n, d=1 / fs)

    # Get the oneside of FFT results.
    amp = amp[: n // 2]
    freq = freq[: n // 2]

    # Get frequencies and amplitudes of FFT samples within the frequency range.
    if freq_range is None:
        low_f = freq[0]
        high_f = freq[-1]
    else:
        low_f = freq_range[0]
        high_f = freq_range[-1]

    low_idx = np.where(freq >= low_f)[0][0]
    high_idx = np.where(freq <= high_f)[0][-1]

    amp = amp[low_idx : high_idx + 1]
    freq = freq[low_idx : high_idx + 1]

    return amp, freq


def mnf(x: np.ndarray, fs: float = None, freq_range: Tuple = None) -> float:
    """Compute the mean frequency.
    Mean frequency has a similar definiton as the central frequency."""
    # TODO: Doscrings
    # TODO: Exception for types, availability of freq_range

    # Get the amplitudes and FFT sample frequencies.
    amp, freq = _get_amp_and_freq(x, fs, freq_range)

    # Get the MNF(mean frequency).
    mnf = np.sum(freq * np.abs(amp)) / np.sum(np.abs(amp))

    return mnf


def mdf(x: np.ndarray, fs: float = None, freq_range: Tuple = None) -> float:
    """Compute the median frequency."""
    # Get the amplitudes and FFT sample frequencies.
    amp, freq = _get_amp_and_freq(x, fs, freq_range)

    # Get the MDF(median frequency).
    cumsum_amp = np.cumsum(amp)
    mdf_idx = np.where(cumsum_amp <= cumsum_amp[-1] / 2)[0][-1]
    mdf = freq[mdf_idx]

    return mdf


def vcf(x: np.ndarray, fs: float = None, freq_range: Tuple = None) -> float:
    """Compute the variance of central frequency(mean frequency)."""
    # Get the amplitudes and FFT sample frequencies.
    amp, freq = _get_amp_and_freq(x, fs, freq_range)

    # Get the MNF(mean frequency).
    cf = mnf(x, fs, freq_range)

    # Get the VCF(variance of central frequency).
    vcf = np.sum(((freq - cf) ** 2) * np.abs(amp)) / np.sum(np.abs(amp))

    return vcf

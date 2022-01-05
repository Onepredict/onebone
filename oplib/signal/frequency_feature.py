"""Frequency domain feature.

- Author: Kangwhi Kim
- Contact: kangwhi.kim@onepredict.com
"""

from typing import Tuple, Union

import numpy as np
from scipy.fft import fft, fftfreq


def _index_along_axis(x: np.ndarray, s: slice, axis: int):
    """Index under certain conditions along the axis you specify."""
    x = x.copy()  # shallow copy
    if axis == -1:
        lower_ndim, upper_ndim = len(x.shape[:axis]), 0
    else:
        lower_ndim, upper_ndim = len(x.shape[:axis]), len(x.shape[axis + 1 :])
    indices = (
        lower_ndim
        * np.s_[
            :,
        ]
        + (s,)
        + upper_ndim
        * np.s_[
            :,
        ]
    )
    x = x[indices]

    return x


def _get_amp_and_freq(
    x: np.ndarray, fs: float = None, freq_range: Tuple = None, axis: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the amplitudes and FFT sample frequencies through positive FFT.
    And you can get results within a specific frequency range.
    """
    # Set default parameter
    if fs is None:
        fs = 1  # `fs`` for normalized mean frequency.
    if freq_range is None:
        freq_range = (0, fs / 2)

    # Check inputs
    if not isinstance(x, np.ndarray):
        raise TypeError("'x' must be array.")
    if len(x.shape) >= 3:
        raise ValueError("'x' has less than 3 dimensions.")

    if not isinstance(fs, (int, float)):
        raise TypeError("'fs' must be integer or float.")

    if not isinstance(freq_range, Tuple):
        raise TypeError("'freq_range' must be tuple.")
    if len(freq_range) != 2:
        raise ValueError("'freq_range' requires two elements.")
    if not (isinstance(freq_range[0], (int, float)) & isinstance(freq_range[-1], (int, float))):
        raise TypeError("The elements of 'freq_range' must be integer or float.")
    if not (freq_range[0] < freq_range[-1]):
        raise ValueError("The first element of 'freq_range' must be lower than the second element.")

    # Do FFT.
    amp = np.abs(fft(x, axis=axis))
    n = amp.shape[axis]

    # Return the FFT sample frequencies.
    freq = fftfreq(n, d=1 / fs)

    # Get the oneside of FFT results along axis.
    amp = _index_along_axis(amp, np.s_[: n // 2], axis)
    freq = freq[: n // 2]

    # Get frequencies and amplitudes of FFT samples within the frequency range.
    low_f = freq_range[0]
    high_f = freq_range[-1]

    freq_range_indices = np.where((freq >= low_f) & (freq <= high_f))[0]
    if len(freq_range_indices) == 0:
        raise ValueError("The frequency range is not valid.")
    low_idx = freq_range_indices[0]
    high_idx = freq_range_indices[-1]

    amp = _index_along_axis(amp, np.s_[low_idx : high_idx + 1], axis=axis)
    freq = freq[low_idx : high_idx + 1]

    # Make the dimensions of 'amp' equal to the dimensions of 'freq'.
    if axis == -1:
        lower_dims, upper_dims = amp.shape[:axis], ()
    else:
        lower_dims, upper_dims = amp.shape[:axis], amp.shape[axis + 1 :]
    freq_shape = len(lower_dims) * (1,) + (amp.shape[axis],) + len(upper_dims) * (1,)
    freq = freq.reshape(freq_shape)
    amp_shape = lower_dims + (1,) + upper_dims
    freq = np.zeros(amp_shape) + freq  # Numpy broadcasting

    return amp, freq


def mnf(
    x: np.ndarray,
    fs: float = None,
    freq_range: Tuple = None,
    axis: int = -1,
    keepdims: bool = False,
) -> Union[float, np.ndarray]:
    """Compute the mean frequency.
    Mean frequency has a similar definiton as the central frequency."""
    # Get the amplitudes and FFT sample frequencies.
    amp, freq = _get_amp_and_freq(x, fs, freq_range, axis)

    # Get the MNF(mean frequency) along the `axis`.
    mnf = np.sum(freq * amp, axis=axis, keepdims=keepdims) / np.sum(
        amp, axis=axis, keepdims=keepdims
    )

    return mnf


def _search_median_frequency(amp, freq, axis, keepdims):
    # Get the 1-D frequency.
    if axis == -1:
        lower_dims, upper_dims = freq.shape[:axis], ()
    else:
        lower_dims, upper_dims = freq.shape[:axis], freq.shape[axis + 1 :]
    freq_shape = (
        len(lower_dims) * (0,)
        + np.s_[
            :,
        ]
        + len(upper_dims) * (0,)
    )
    freq = freq[freq_shape]

    # Get the MDF(median frequency).
    cumsum_a = np.cumsum(amp)
    mdf = freq[cumsum_a >= cumsum_a[-1] / 2][0]

    # When `keepdims` is True, return the result of the list type.
    if keepdims:
        return [mdf]
    else:
        return mdf


def mdf(
    x: np.ndarray,
    fs: float = None,
    freq_range: Tuple = None,
    axis: int = -1,
    keepdims: bool = False,
) -> float:
    """Compute the median frequency."""
    # Get the amplitudes and FFT sample frequencies.
    amp, freq = _get_amp_and_freq(x, fs, freq_range, axis)

    # Get the MDF(median frequency) along the `axis`.
    mdf = np.apply_along_axis(_search_median_frequency, axis, amp, *(freq, axis, keepdims))

    return mdf


def vcf(
    x: np.ndarray,
    fs: float = None,
    freq_range: Tuple = None,
    axis: int = -1,
    keepdims: bool = False,
) -> float:
    """Compute the variance of central frequency(mean frequency)."""
    # Get the amplitudes and FFT sample frequencies.
    amp, freq = _get_amp_and_freq(x, fs, freq_range, axis)

    # Get the MNF(mean frequency).
    cf = mnf(x, fs, freq_range, axis, keepdims=True)

    # Get the VCF(variance of central frequency) along the `axis`.
    vcf = np.sum(((freq - cf) ** 2) * amp, axis=axis, keepdims=keepdims) / np.sum(
        amp, axis=axis, keepdims=keepdims
    )

    return vcf

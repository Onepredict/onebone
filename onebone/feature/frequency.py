"""Frequency domain feature.

- Author: Kangwhi Kim
- Contact: kangwhi.kim@onepredict.com
"""

from typing import Tuple, Union

import numpy as np
from scipy.fft import fftfreq

from onebone.utils import slice_along_axis


def _get_amp_and_freq_for_oneside_spectrum(
    amp: np.ndarray, fs: Union[int, float] = 1, freq_range: Tuple = None, axis: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the amplitudes and FFT sample frequencies of the one-side amplitude spectrum.
    """
    # Set default parameter
    if freq_range is None:
        freq_range = (0, fs / 2)

    # Check inputs
    if not isinstance(amp, np.ndarray):
        raise TypeError("'amp' must be array.")
    if len(amp.shape) >= 3:
        raise ValueError("'amp' has less than 3 dimensions.")
    if np.iscomplexobj(amp):
        raise ValueError("The elements of 'amp' must be real.")

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

    # Get the size of amplitudes.
    n = amp.shape[axis]

    # Get the FFT sample frequencies.
    freq = fftfreq(n, d=1 / fs)

    # Get the oneside of FFT results along axis.
    amp = slice_along_axis(amp, np.s_[: n // 2], axis)
    freq = freq[: n // 2]

    # Get frequencies and amplitudes of FFT samples within the frequency range.
    low_f = freq_range[0]
    high_f = freq_range[-1]

    freq_range_indices = np.where((freq >= low_f) & (freq <= high_f))[0]
    if len(freq_range_indices) == 0:
        raise ValueError("The frequency range is not valid.")
    low_idx = freq_range_indices[0]
    high_idx = freq_range_indices[-1]

    amp = slice_along_axis(amp, np.s_[low_idx : high_idx + 1], axis=axis)
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

    return freq, amp


def mnf(
    amp: np.ndarray,
    fs: Union[int, float] = 1,
    freq_range: Tuple = None,
    axis: int = -1,
    keepdims: bool = False,
) -> Union[float, np.ndarray]:
    """
    Compute the mean frequency.

    .. math:: {MNF = {\sum_{j=1}^M f_{j}A_{j} \over \sum_{j=1}^M A_{j}}}^{[1]}

    Where :math:`f_{j}` is the frequency value of spectrum at the bin :math:`j`,
    :math:`A_{j}` is the amplitude value of spectrum at the frequency bin :math:`j`,
    and :math:`M` is the length of frequency bin.

    Parameters
    ----------
    amp : numpy.ndarray of shape (signal_length,), (n, signal_length)
        The amplitudes of a spectrum of time-domain signals.
    fs : int or float, default=1
        Sample rate. The sample rate is the number of samples per unit time.
        If `fs` is 1, then `mnf` is the normalized frequency; (0 ~ 1).
    freq_range : tuple, default=None
        Frequency range, specified as a two-element tuple of real values.
        If `freq_range` is None, then `mnf` uses the entire bandwidth of the input signal.
        That is, `freq_range` is (0, `fs` / 2).
    axis : int, default=-1
        Axis along which `mnf` is performed.
        The default, `axis`=-1, will calculate the `mnf` along last axis of `x`.
        If `axis` is negative, it counts from the last to the first axis.
    keepdims : bool, default=False
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one.

    Returns
    -------
    mnf : float or numpy.ndarray
        Mean frequency.
        If `fs` is 1, then `mnf` has units of cycle/sample.
        But, if you specify the `fs`, then `mnf` has the same units as `fs`. E.g. cycle/sec.

    References
    ----------
    .. [1] Angkoon Phinyomark, Sirinee Thongpanja, Huosheng Hu,
       Pornchai Phukpattaranont and Chusak Limsakul (October 17th 2012).
       The Usefulness of Mean and Median Frequencies in Electromyography Analysis,
       Computational Intelligence in Electromyography Analysis -
       A Perspective on Current Applications and Future Challenges, Ganesh R. Naik, IntechOpen,
       DOI: 10.5772/50639. Available from: https://www.intechopen.com/chapters/40123

    Examples
    --------
    >>> import numpy as np
    >>> fs = 100
    >>> t = np.arange(0, 1, 1 / fs)
    >>> x1 = np.sin(2 * np.pi * 10 * t)
    >>> x2 = np.sin(2 * np.pi * 20 * t)
    >>> x3 = np.sin(2 * np.pi * 30 * t)
    >>> x = x1 + x2 + x3
    >>> amp = np.abs(np.fft.fft(x))
    >>> mnf(amp, fs)
    20
    """
    # Get the amplitudes and FFT sample frequencies.
    freq, amp = _get_amp_and_freq_for_oneside_spectrum(amp, fs, freq_range, axis)

    # Get the MNF(mean frequency) along the `axis`.
    mnf = np.sum(freq * amp, axis=axis, keepdims=keepdims) / np.sum(
        amp, axis=axis, keepdims=keepdims
    )

    return mnf


def _get_mdf_of_1D_signal(amp: np.ndarray, freq: np.ndarray, keepdims: bool = False):
    """Get the median frequency for the 1-D signal"""
    # Get the MDF(median frequency).
    cumsum_a = np.cumsum(amp)
    mdf = freq[cumsum_a >= cumsum_a[-1] / 2][0]

    # When `keepdims` is True, return the result of the list type.
    if keepdims:
        return [mdf]
    else:
        return mdf


def mdf(
    amp: np.ndarray,
    fs: Union[int, float] = 1,
    freq_range: Tuple = None,
    axis: int = -1,
    keepdims: bool = False,
) -> float:
    """
    Compute the median frequency.

    .. math:: {\sum_{j=1}^{MDF} A_{j} = \sum_{j=MDF}^{M} A_{j} = {1 \over 2}\sum_{j=1}^M P_{j}}^{[1]}

    Where :math:`A_{j}` is the amplitude value of spectrum at the frequency bin :math:`j`,
    and :math:`M` is the length of frequency bin.

    Parameters
    ----------
    amp : numpy.ndarray of shape (signal_length,), (n, signal_length)
        The amplitudes of a spectrum of time-domain signals.
    fs : int or float, default=1
        Sample rate. The sample rate is the number of samples per unit time.
        If `fs` is 1, then `mdf` is the normalized frequency; (0 ~ 1).
    freq_range : tuple, default=None
        Frequency range, specified as a two-element tuple of real values.
        If `freq_range` is None, then `mdf` uses the entire bandwidth of the input signal.
        That is, `freq_range` is (0, `fs` / 2).
    axis : int, default=-1
        Axis along which `mdf` is performed.
        The default, `axis`=-1, will calculate the `mdf` along last axis of `x`.
        If `axis` is negative, it counts from the last to the first axis.
    keepdims : bool, default=False
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one.

    Returns
    -------
    mdf : float or numpy.ndarray
        Median frequency.
        If `fs` is 1, then `mdf` has units of cycle/sample.
        But, if you specify the `fs`, then `mdf` has the same units as `fs`. E.g. cycle/sec.

    References
    ----------
    .. [1] Angkoon Phinyomark, Sirinee Thongpanja, Huosheng Hu,
       Pornchai Phukpattaranont and Chusak Limsakul (October 17th 2012).
       The Usefulness of Mean and Median Frequencies in Electromyography Analysis,
       Computational Intelligence in Electromyography Analysis -
       A Perspective on Current Applications and Future Challenges, Ganesh R. Naik, IntechOpen,
       DOI: 10.5772/50639. Available from: https://www.intechopen.com/chapters/40123

    Examples
    --------
    >>> import numpy as np
    >>> fs = 100
    >>> t = np.arange(0, 1, 1 / fs)
    >>> x1 = np.sin(2 * np.pi * 10 * t)
    >>> x2 = np.sin(2 * np.pi * 20 * t)
    >>> x3 = np.sin(2 * np.pi * 30 * t)
    >>> x = x1 + x2 + x3
    >>> amp = np.abs(np.fft.fft(x))
    >>> mdf(amp, fs)
    20
    """
    # Get the amplitudes and FFT sample frequencies.
    freq, amp = _get_amp_and_freq_for_oneside_spectrum(amp, fs, freq_range, axis)

    # Get the 1-D sample frequencies.
    if axis == -1:
        lower_dims, upper_dims = freq.shape[:axis], ()
    else:
        lower_dims, upper_dims = freq.shape[:axis], freq.shape[axis + 1 :]
    freq_shape = len(lower_dims) * (0,) + (np.s_[:],) + len(upper_dims) * (0,)
    freq = freq[freq_shape]

    # Get the MDF(median frequency) along the `axis`.
    mdf = np.apply_along_axis(_get_mdf_of_1D_signal, axis, amp, *(freq, keepdims))

    return mdf


def vcf(
    amp: np.ndarray,
    fs: Union[int, float] = 1,
    freq_range: Tuple = None,
    axis: int = -1,
    keepdims: bool = False,
) -> float:
    """
    Compute the variance of central frequency(mean frequency).

    .. math:: {VCF = {1 \over \sum_{j=1}^M A_{j}}{\sum_{j=1}^M A_{j}(f_{j} - MNF)^2}}^{[1]}

    Where :math:`f_{j}` is the frequency value of spectrum at the bin :math:`j`,
    :math:`A_{j}` is the amplitude value of spectrum at the frequency bin :math:`j`,
    :math:`M` is the length of frequency bin,
    :math:`MNF` is the mean frequency.

    Parameters
    ----------
    amp : numpy.ndarray of shape (signal_length,), (n, signal_length)
        The amplitudes of a spectrum of time-domain signals.
    fs : int or float, default=1
        Sample rate. The sample rate is the number of samples per unit time.
        If `fs` is 1, then `vcf` is the normalized frequency; (0 ~ 1).
    freq_range : tuple, default=None
        Frequency range, specified as a two-element tuple of real values.
        If `freq_range` is None, then `vcf` uses the entire bandwidth of the input signal.
        That is, `freq_range` is (0, `fs` / 2).
    axis : int, default=-1
        Axis along which `vcf` is performed.
        The default, `axis`=-1, will calculate the `vcf` along last axis of `x`.
        If `axis` is negative, it counts from the last to the first axis.
    keepdims : bool, default=False
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one.

    Returns
    -------
    vcf : float or numpy.ndarray
        Median frequency.
        If `fs` is 1, then `vcf` has units of cycle/sample.
        But, if you specify the `fs`, then `vcf` has the same units as `fs`. E.g. cycle/sec.

    References
    ----------
    .. [1] Angkoon Phinyomark, Sirinee Thongpanja, Huosheng Hu,
       Pornchai Phukpattaranont and Chusak Limsakul (October 17th 2012).
       The Usefulness of Mean and Median Frequencies in Electromyography Analysis,
       Computational Intelligence in Electromyography Analysis -
       A Perspective on Current Applications and Future Challenges, Ganesh R. Naik, IntechOpen,
       DOI: 10.5772/50639. Available from: https://www.intechopen.com/chapters/40123

    Examples
    --------
    >>> import numpy as np
    >>> fs = 100
    >>> t = np.arange(0, 1, 1 / fs)
    >>> x1 = np.sin(2 * np.pi * 10 * t)
    >>> x2 = np.sin(2 * np.pi * 20 * t)
    >>> x3 = np.sin(2 * np.pi * 30 * t)
    >>> x = x1 + x2 + x3
    >>> amp = np.abs(np.fft.fft(x))
    >>> vcf(x, fs)
    66.666
    """
    # Get the MNF(mean frequency).
    cf = mnf(amp, fs, freq_range, axis, keepdims=True)

    # Get the amplitudes and FFT sample frequencies.
    freq, amp = _get_amp_and_freq_for_oneside_spectrum(amp, fs, freq_range, axis)

    # Get the VCF(variance of central frequency) along the `axis`.
    vcf = np.sum(((freq - cf) ** 2) * amp, axis=axis, keepdims=keepdims) / np.sum(
        amp, axis=axis, keepdims=keepdims
    )

    return vcf

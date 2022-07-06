"""A frequency filter to leave only a specific frequency band.
   and a filter that replaces outlier values in data with other values.

- Author: Kyunghwan Kim, Sunjin Kim
- Contact: kyunghwan.kim@onepredict.com, sunjin.kim@onepredict.com
"""

from typing import Tuple, Union

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
    1D Butterworth lowpass filter.

    Parameters
    ----------
    signal : numpy.ndarray
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
    out : numpy.ndarray
        Filtered signal.

    Examples
    --------
    Apply the filter to 1d signal. And then check the frequency component of the filtered signal.

    >>> fs = 5000.0
    >>> t = np.linspace(0, 1, int(fs))
    >>> signal = 10.0 * np.sin(2 * np.pi * 20.0 * t)
    >>> signal += 5.0 * np.sin(2 * np.pi * 100.0 * t)
    >>> signal.shape
    (5000,)
    >>> freq_x = np.fft.rfftfreq(signal.size, 1 / fs)[:-1]
    >>> origin_fft_mag = abs((np.fft.rfft(signal) / signal.size)[:-1] * 2)
    >>> origin_freq = freq_x[np.where(origin_fft_mag > 0.5)]
    >>> origin_freq
    [ 20., 100.]
    >>> filtered_signal = lowpass_filter(signal, fs, cutoff=50)
    >>> filtered_fft_mag = abs((np.fft.rfft(filtered_signal) / signal.size)[:-1] * 2)
    >>> filtered_freq = freq_x[np.where(filtered_fft_mag > 0.5)]
    >>> filtered_freq
    [ 20.]

    Apply the filter to 2d signal (axis=0).

    >>> fs = 5000.0
    >>> t = np.linspace(0, 1, int(fs))
    >>> signal = 10.0 * np.sin(2 * np.pi * 20.0 * t)
    >>> signal += 5.0 * np.sin(2 * np.pi * 100.0 * t)
    >>> signal = np.stack([signal, signal]).T
    >>> signal.shape
    (5000, 2)
    >>> filtered_signal = lowpass_filter(signal, fs, cutoff=50, axis=0)
    >>> filtered_fft_mag = abs((np.fft.rfft(filtered_signal[:, 0]) / signal.size)[:-1] * 2)
    >>> filtered_freq = freq_x[np.where(filtered_fft_mag > 0.5)]
    >>> filtered_freq
    [ 20.]
    """
    if not (isinstance(fs, int) | isinstance(fs, float)):
        raise TypeError(f"Argument 'fs' must be of type int or float, not {type(fs).__name__}")
    if not (isinstance(cutoff, int) | isinstance(cutoff, float)):
        raise TypeError(
            f"Argument 'cutoff' must be of type int or float, not {type(cutoff).__name__}"
        )
    if not isinstance(order, int):
        raise TypeError(
            f"Argument 'order' must be of type int or float, not {type(order).__name__}"
        )
    if not isinstance(axis, int):
        raise TypeError(f"Argument 'axis' must be of type int or float, not {type(axis).__name__}")

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
    1D Butterworth highpass filter.

    Parameters
    ----------
    signal : numpy.ndarray
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
    out : numpy.ndarray
        Filtered signal.

    Examples
    --------
    Apply the filter to 1d signal. And then check the frequency component of the filtered signal.

    >>> fs = 5000.0
    >>> t = np.linspace(0, 1, int(fs))
    >>> signal = 10.0 * np.sin(2 * np.pi * 20.0 * t)
    >>> signal += 5.0 * np.sin(2 * np.pi * 100.0 * t)
    >>> signal.shape
    (5000,)
    >>> freq_x = np.fft.rfftfreq(signal.size, 1 / fs)[:-1]
    >>> origin_fft_mag = abs((np.fft.rfft(signal) / signal.size)[:-1] * 2)
    >>> origin_freq = freq_x[np.where(origin_fft_mag > 0.5)]
    >>> origin_freq
    [ 20., 100.]
    >>> filtered_signal = highpass_filter(signal, fs, cutoff=50)
    >>> filtered_fft_mag = abs((np.fft.rfft(filtered_signal) / signal.size)[:-1] * 2)
    >>> filtered_freq = freq_x[np.where(filtered_fft_mag > 0.5)]
    >>> filtered_freq
    [ 100.]

    Apply the filter to 2d signal (axis=0).

    >>> fs = 5000.0
    >>> t = np.linspace(0, 1, int(fs))
    >>> signal = 10.0 * np.sin(2 * np.pi * 20.0 * t)
    >>> signal += 5.0 * np.sin(2 * np.pi * 100.0 * t)
    >>> signal = np.stack([signal, signal]).T
    >>> signal.shape
    (5000, 2)
    >>> filtered_signal = highpass_filter(signal, fs, cutoff=50, axis=0)
    >>> filtered_fft_mag = abs((np.fft.rfft(filtered_signal[:, 0]) / signal.size)[:-1] * 2)
    >>> filtered_freq = freq_x[np.where(filtered_fft_mag > 0.5)]
    >>> filtered_freq
    [ 100.]
    """
    if not (isinstance(fs, int) | isinstance(fs, float)):
        raise TypeError(f"Argument 'fs' must be of type int or float, not {type(fs).__name__}")
    if not (isinstance(cutoff, int) | isinstance(cutoff, float)):
        raise TypeError(
            f"Argument 'cutoff' must be of type int or float, not {type(cutoff).__name__}"
        )
    if not isinstance(order, int):
        raise TypeError(
            f"Argument 'order' must be of type int or float, not {type(order).__name__}"
        )
    if not isinstance(axis, int):
        raise TypeError(f"Argument 'axis' must be of type int or float, not {type(axis).__name__}")

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
    1D Butterworth bandpass filter.

    Parameters
    ----------
    signal : numpy.ndarray
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
    out : numpy.ndarray
        Filtered signal.
        If input shape is [signal_length,], output shape is [signal_length,].
        If input shape is [n, signal_length,], output shape is [n, signal_length,].

    Examples
    --------
    Apply the filter to 1d signal. And then check the frequency component of the filtered signal.

    >>> fs = 5000.0
    >>> t = np.linspace(0, 1, int(fs))
    >>> signal = 10.0 * np.sin(2 * np.pi * 20.0 * t)
    >>> signal += 5.0 * np.sin(2 * np.pi * 100.0 * t)
    >>> signal += 5.0 * np.sin(2 * np.pi * 500.0 * t)
    >>> signal.shape
    (5000,)
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

    Apply the filter to 2d signal (axis=0).

    >>> fs = 5000.0
    >>> t = np.linspace(0, 1, int(fs))
    >>> signal = 10.0 * np.sin(2 * np.pi * 20.0 * t)
    >>> signal += 5.0 * np.sin(2 * np.pi * 100.0 * t)
    >>> signal = np.stack([signal, signal]).T
    >>> signal.shape
    (5000, 2)
    >>> filtered_signal = bandpass_filter(signal, fs, l_cutoff=50, h_cutoff=300, axis=0)
    >>> filtered_fft_mag = abs((np.fft.rfft(filtered_signal[:, 0]) / signal.size)[:-1] * 2)
    >>> filtered_freq = freq_x[np.where(filtered_fft_mag > 0.5)]
    >>> filtered_freq
    [ 100.]
    """
    if not (isinstance(fs, int) | isinstance(fs, float)):
        raise TypeError(f"Argument 'fs' must be of type int or float, not {type(fs).__name__}")
    if not (isinstance(l_cutoff, int) | isinstance(l_cutoff, float)):
        raise TypeError(
            f"Argument 'l_cutoff' must be of type int or float, not {type(l_cutoff).__name__}"
        )
    if not (isinstance(h_cutoff, int) | isinstance(h_cutoff, float)):
        raise TypeError(
            f"Argument 'h_cutoff' must be of type int or float, not {type(h_cutoff).__name__}"
        )
    if not isinstance(order, int):
        raise TypeError(
            f"Argument 'order' must be of type int or float, not {type(order).__name__}"
        )
    if not isinstance(axis, int):
        raise TypeError(f"Argument 'axis' must be of type int or float, not {type(axis).__name__}")

    nyq = 0.5 * fs
    low = l_cutoff / nyq
    high = h_cutoff / nyq
    b, a = butter(order, [low, high], btype="bandpass")
    signal = lfilter(b, a, signal, axis=axis)
    return signal


def bandpass_filter_ideal(
    signal: np.ndarray,
    fs: Union[int, float],
    l_cutoff: Union[int, float],
    h_cutoff: Union[int, float],
) -> np.ndarray:
    """
    .. warning:: This method **may cause distortion of signal**. \
        Generally, this operates well on signals extracted in low resolution. \
        In order to check the distortion of signals, \
        it is recommended to monitor the linear transition of phase.

    1D ideal bandpass filter.

    Parameters
    ----------
    signal : numpy.ndarray
        Original time-domain signal.
    fs : Union[int, float]
        Sampling rate.
    l_cutoff : Union[int, float]
        Low cutoff frequency.
    h_cutoff : Union[int, float]
        High cutoff frequency.

    Returns
    -------
    out : numpy.ndarray
        Filtered signal.
        Input shape is [signal_length,] and output shape is [signal_length,].

    Examples
    --------
    Apply the filter to 1d signal. And then check the frequency component of the filtered signal.

    >>> fs = 5000.0
    >>> t = np.linspace(0, 1, int(fs))
    >>> signal = 10.0 * np.sin(2 * np.pi * 20.0 * t)
    >>> signal += 5.0 * np.sin(2 * np.pi * 100.0 * t)
    >>> signal += 5.0 * np.sin(2 * np.pi * 500.0 * t)
    >>> signal.shape
    (5000,)
    >>> freq_x = np.fft.rfftfreq(signal.size, 1 / fs)[:-1]
    >>> origin_fft_mag = abs((np.fft.rfft(signal) / signal.size)[:-1] * 2)
    >>> origin_freq = freq_x[np.where(origin_fft_mag > 0.5)]
    >>> origin_freq
    [ 20., 100., 500.]
    >>> filtered_signal = bandpass_filter_ideal(signal, fs, l_cutoff=50, h_cutoff=300)
    >>> filtered_fft_mag = abs((np.fft.rfft(filtered_signal) / signal.size)[:-1] * 2)
    >>> filtered_freq = freq_x[np.where(filtered_fft_mag > 0.5)]
    >>> filtered_freq
    [ 100.]
    """
    if not (isinstance(fs, int) | isinstance(fs, float)):
        raise TypeError(f"Argument 'fs' must be of type int or float, not {type(fs).__name__}")
    if not (isinstance(l_cutoff, int) | isinstance(l_cutoff, float)):
        raise TypeError(
            f"Argument 'l_cutoff' must be of type int or float, not {type(l_cutoff).__name__}"
        )
    if not (isinstance(h_cutoff, int) | isinstance(h_cutoff, float)):
        raise TypeError(
            f"Argument 'h_cutoff' must be of type int or float, not {type(h_cutoff).__name__}"
        )
    if len(signal.shape) > 1:
        raise TypeError(f"Argument 'len(signal.shape)' must be 1, not {len(signal.shape)}")

    n = len(signal)
    t = n / fs
    k = np.arange(n)

    freq_full = k / t

    yfull = np.fft.fft(signal)
    band2low = fs - h_cutoff
    band2high = fs - l_cutoff
    band1low_idx = np.where(freq_full >= l_cutoff)[0][0]
    band1high_idx = np.where(freq_full >= h_cutoff)[0][0]
    band2low_idx = np.where(freq_full >= band2low)[0][0]
    band2high_idx = np.where(freq_full >= band2high)[0][0]
    band1_idx = np.arange(band1low_idx, band1high_idx)
    band2_idx = np.arange(band2low_idx, band2high_idx)
    band_idx = np.vstack([band1_idx, band2_idx]).T
    full_idx = np.arange(n)
    notch_idx = np.setdiff1d(full_idx, band_idx)
    filt_yfull = yfull.copy()
    filt_yfull[notch_idx] = 0
    filter_signal = np.real(np.fft.ifft(filt_yfull))
    return filter_signal


def bandstop_filter(
    signal: np.ndarray,
    fs: Union[int, float],
    l_cutoff: Union[int, float],
    h_cutoff: Union[int, float],
    order: int = 5,
    axis: int = -1,
) -> np.ndarray:
    """
    1D Butterworth bandstop filter.

    Parameters
    ----------
    signal : numpy.ndarray
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
    out : numpy.ndarray
        Filtered signal.
        If input shape is [signal_length,], output shape is [signal_length,].
        If input shape is [n, signal_length,], output shape is [n, signal_length,].

    Examples
    --------
    Apply the filter to 1d signal. And then check the frequency component of the filtered signal.

    >>> fs = 5000.0
    >>> t = np.linspace(0, 1, int(fs))
    >>> signal = 10.0 * np.sin(2 * np.pi * 20.0 * t)
    >>> signal += 5.0 * np.sin(2 * np.pi * 100.0 * t)
    >>> signal += 5.0 * np.sin(2 * np.pi * 500.0 * t)
    >>> signal.shape
    (5000,)
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

    Apply the filter to 2d signal (axis=0).

    >>> fs = 5000.0
    >>> t = np.linspace(0, 1, int(fs))
    >>> signal = 10.0 * np.sin(2 * np.pi * 20.0 * t)
    >>> signal += 5.0 * np.sin(2 * np.pi * 100.0 * t)
    >>> signal = np.stack([signal, signal]).T
    >>> signal.shape
    (5000, 2)
    >>> filtered_signal = bandstop_filter(signal, fs, l_cutoff=50, h_cutoff=300, axis=0)
    >>> filtered_fft_mag = abs((np.fft.rfft(filtered_signal[:, 0]) / signal.size)[:-1] * 2)
    >>> filtered_freq = freq_x[np.where(filtered_fft_mag > 0.5)]
    >>> filtered_freq
    [ 20., 500.]
    """
    if not (isinstance(fs, int) | isinstance(fs, float)):
        raise TypeError(f"Argument 'fs' must be of type int or float, not {type(fs).__name__}")
    if not (isinstance(l_cutoff, int) | isinstance(l_cutoff, float)):
        raise TypeError(
            f"Argument 'l_cutoff' must be of type int or float, not {type(l_cutoff).__name__}"
        )
    if not (isinstance(h_cutoff, int) | isinstance(h_cutoff, float)):
        raise TypeError(
            f"Argument 'h_cutoff' must be of type int or float, not {type(h_cutoff).__name__}"
        )
    if not isinstance(order, int):
        raise TypeError(
            f"Argument 'order' must be of type int or float, not {type(order).__name__}"
        )
    if not isinstance(axis, int):
        raise TypeError(f"Argument 'axis' must be of type int or float, not {type(axis).__name__}")

    nyq = 0.5 * fs
    low = l_cutoff / nyq
    high = h_cutoff / nyq
    b, a = butter(order, [low, high], btype="bandstop")
    signal = lfilter(b, a, signal, axis=axis)
    return signal


def hampel_filter(x: np.ndarray, window_size: int, n_sigma: float = 3) -> Tuple[np.ndarray, list]:
    """
    A hampel filter removes outliers.
    Estimate the median and standard deviation of each sample using
    MAD(Median Absolute Deviation) in the window range set by the user.
    If the MAD > 3 * sigma condition is satisfied,
    the value is replaced with the median value.

    .. math::
        m_i = median(x_{i-k_{left}}, x_{i-k_{left}+1}, ..., x_{i+k_{right}-1}, x_{i+k_{right}})
    .. math::
        MAD_i = median(|x_{i-k}-m_{i}|,...,|x_{i+k}-m_{i}|)
    .. math::
        {\sigma}_i = {\kappa} * MAD_i

    Where :math:`k_{left}` and :math:`k_{right}` are the number of neighbors on the left and right sides,
    respectively, based on x_i (:math:`k_{left} + k_{right}` = window samples).
    :math:`m_i` is Local median, :math:`MAD_i` is median absolute deviation
    which is the residuals (deviations) from the data's median.
    :math:`{\sigma}_i` is the MAD may be used similarly to how one would use the deviation for the average.
    In order to use the MAD as a consistent estimator
    for the estimation of the standard deviation :math:`{\sigma}`, one takes :math:`{\kappa} * MAD_i`.
    :math:`{\kappa}` is a constant scale factor, which depends on the distribution.
    For normally distributed data :math:`{\kappa}` is taken to be :math:`{\kappa}` = 1.4826


    Parameters
    ----------
    x : numpy.ndarray
        1d-timeseries data.
        The shape of x must be (signal_length,) .
    window_size : int
        Lenght of the sliding window.
        Only integer types are available,
        and the window size must be adjusted according to your data.
    n_sigma : float, defalut=3
        Coefficient of standard deviation.

    Returns
    -------
    filtered_x : numpy.ndarray
        A value from which outlier or NaN has been removed by the filter.
    index : list
        Returns the index corresponding to outlier.

    References
    ----------
    [1] Pearson, Ronald K., et al. "Generalized hampel filters."
    EURASIP Journal on Advances in Signal Processing 2016.1 (2016): 1-18.
    DOI: 10.1186/s13634-016-0383-6

    Examples
    --------
    >>> fs = 1000.0
    >>> t = np.linspace(0, 1, int(fs))
    >>> y = np.sin(2 * np.pi * 10.0 * t)
    >>> np.put(y, [13, 124, 330, 445, 651, 775, 978], 3)
    >>> plt.plot(y) # noise_signal

    .. image:: https://bit.ly/3CWKVaw
        :width: 600

    >>> filtered_signal = hampel_filter.hampel_filter(y, window_size=5)[0]
    >>> plt.plot(filtered_signal) # filtered_signal

    .. image:: https://bit.ly/3in3Jq1
        :width: 600

    """

    # Check inputs
    if not isinstance(x, np.ndarray):
        raise TypeError("'x' must be np.ndarray")
    if not isinstance(window_size, int):
        raise TypeError("'window_size' must be integer")
    if not isinstance(n_sigma, (int, float)):
        raise TypeError("'n_sigma' must be int or float")

    copy_x = x.copy()
    oulier_index = []

    # Define sliding window
    indexer = np.arange(window_size)[None, :] + np.arange(len(copy_x) - window_size)[:, None]

    # Define window median
    window_median = np.median(copy_x[indexer], axis=1)
    window_median_array = np.repeat(window_median, window_size, axis=0).reshape(
        np.shape(copy_x[indexer])[0], window_size
    )

    # Get estimated_sigma (mad * k)
    k = 1.4826
    estimated_sigma = k * np.median(np.abs(copy_x[indexer] - window_median_array), axis=1)
    # Scale factor for Gaussian distribution
    # The factor 1.4826 makes the MAD scale estimate an unbiased estimate
    # of the standard deviation for Gaussian data.
    # Get comparative value

    value = np.abs(copy_x[indexer][:, 0] - window_median)

    filtered_x = np.where(value > n_sigma * estimated_sigma, window_median, copy_x[indexer][:, 0])

    # Get index
    oulier_index = np.where(value <= n_sigma * estimated_sigma, None, indexer[:, 0])
    oulier_index = list(filter(None, oulier_index))

    return filtered_x, oulier_index

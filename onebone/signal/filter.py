"""A frequency filter to leave only a specific frequency band.

- Author: Kyunghwan Kim
- Contact: kyunghwan.kim@onepredict.com
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


"""hampel_filter.
- Author: Sunjin Kim
- Contact: sunjin.kim@onepredict.com
"""


def hampel_filter(x: np.ndarray, window_size: int, n_sigmas: float = 3) -> Tuple[np.ndarray, list]:
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
    ----------
    filtered_series : numpy.ndarray
        A value from which Outlier or NaN has been removed by the filter.
    index : list
        Returns the index corresponding to Outlier.

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
    >>> print('noise_signal')
    .. image:: https://bit.ly/3JitQu0 #nopa
        :width: 600
    >>> filtered_signal = hampel_filter.hampel_filter(y, window_size=2)[0]
    >>> print('filtered_signal_window=5')
    .. image:: https://bit.ly/3MX92KV #nopa
        :width: 600
    >>> filtered_signal = hampel_filter.hampel_filter(y, window_size=3)[0]
    >>> print('filtered_signal_window=10')
    .. image:: https://bit.ly/3JlBion #nopa
        :width: 600
    """

    k = 1.4826  # scale factor for Gaussian distribution
    # The factor 1.4826 makes the MAD scale estimate an unbiased estimate
    # of the standard deviation for Gaussian data.

    # Check inputs
    if not isinstance(x, np.ndarray):
        raise TypeError("'series' must be np.ndarray")
    if not isinstance(window_size, int):
        raise TypeError("'window_size' must be integer")
    if not isinstance(n_sigmas, (int, float)):
        raise TypeError("'n_sigmas' must be int or float")

    copy_series = x.copy()
    oulier_index = []

    ## define sliding window
    indexer = np.arange(window_size)[None, :] + np.arange(len(copy_series) - window_size)[:, None]

    ## define window median
    window_median = np.median(copy_series[indexer], axis=1)
    window_median_array = np.repeat(window_median, window_size, axis=0).reshape(
        np.shape(copy_series[indexer])[0], window_size
    )

    ## get mad * k
    k_mad = k * np.median(np.abs(copy_series[indexer] - window_median_array), axis=1)

    ## get comparative value
    value = np.abs(copy_series[indexer][:, 0] - window_median)

    filtered_series = np.where(value > n_sigmas * k_mad, window_median, copy_series[indexer][:, 0])

    ## get index
    oulier_index = np.where(value <= n_sigmas * k_mad, None, indexer[:, 0])
    oulier_index = list(filter(None, oulier_index))

    return filtered_series, oulier_index

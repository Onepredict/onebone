"""Condition metrics for gear.

- Author: Kyunghwan Kim, Kangwhi Kim
- Contact: kyunghwan.kim@onepredict.com, kangwhi.kim@onepredict.com
"""

from typing import Tuple, Union

import numpy as np
from scipy.stats import moment


def na4(
    x_tsa: np.ndarray,
    prev_info: Tuple[int, float],
    fs: Union[int, float],
    rpm: float,
    freq_list: Tuple[float] = None,
    n_harmonics: int = 2,
) -> Tuple[float, Tuple[int, float]]:
    """
    Calculate NA4 metric.

    NA4 indicates the onset of damage and continues to react to the damage
    as it spreads and increases in magnitude.

    .. math::
        {NA4 = {N\sum_{i=1}^N (r_{i}-\\bar{r})^4 \over \left\{\\frac{1}{M}
        \sum_{j=1}^M \left[\sum_{i=1}^N (r_{ij}-\\bar{r_j})^2\\right] \\right\}^2}
        = {({\mu}_4)_M \over {\\frac{1}{M}\sum_{j=1}^M ({\mu}_2)_j}}}^{[1]}

    Where :math:`r` is residual signal, :math:`\\bar{r}` is mean value of residual signal,
    :math:`N` is total number of data points in time record,
    :math:`M` is current time record in run ensemble,
    :math:`i` is data point number in time record,
    :math:`j` is time record number in run ensemble,
    :math:`{\mu}_2` is the 2th central moment of :math:`r`,
    and :math:`{\mu}_4` is the 4th central moment of :math:`r`.

    Parameters
    ----------
    x : numpy.ndarray of shape (length of `x`,)
        Time-synchronous-averaged signal (recommended)
    prev_info : tuple of (int, float)
        The information for the 'previous time record number' in run ensemble.
        The first element of `prev_info` is the 'previous time record number'(M-1).
        The second element of `prev_info` is the average of each 2th central moment
        of (M-1) previous residual signals. If the current time record number is 1, the `prev_info` is (0, 0).
    fs : int or float
        Sampling rate
    rpm : float
        Revolution per minute. The unit of 'rpm' is 'rev/min'.
    freq_list : None or tuple of floats, default=None
        The frequencies of gear-meshing components.
        They are filtered from the 'x' signal.
    n_harmonics : int, default=2
        A positive integer specifying the number of shaft
        and gear meshing frequency harmonics to remove.

    Returns
    -------
    na4 : float
        A metric to not only detect the onset of damage,
        but also to continue to react to the damage as it increases.
    cur_info : tuple of (int, float)
        The information for the 'current time record number' in run ensemble.
        The first element of `cur_info` is the 'current time record number'(M).
        The second element of `cur_info` is the average of each 2th central moment of M residual signals.

    References
    ----------
    .. [1] Zakrajsek, James & Handschuh, Robert & Decker, Harry. (1994).
           Application of fault detection techniques to spiral bevel gear fatigue data.
           Available from: https://ntrs.nasa.gov/citations/19940020010

    Examples
    --------
    >>> rpm = 180                                    # Revolution per minute
    >>> fs = 50e3                                    # Sampling rate
    >>> t = np.arange(0, (1 / 3) - 1 / fs, 1 / fs)   # Sample times
    >>> freq_list = (51, 153)                        # Gear mesh frequencies
    >>> f = (rpm/60,) + freq_list                    # Frequencies of signals
    >>> prev_info = (0, 0)         # The information for the 'previous time record number' in run ensemble.
    >>> n_harmonics = 2
    >>> na4_list = []

        # Assume that the gear condition is getting worse.
    >>> for k in range(1, 11):
            # Motor shaft rotation and harmonic
            shaft_signal = np.sin(2 * np.pi * f[0] * t) + np.sin(2 * np.pi * 2 * f[0] * t)
            # Gear mesh vibration and harmonic for a pair of gears
            gm1_signal = 3 * np.sin(2 * np.pi * f[1] * t) + 3 * np.sin(2 * np.pi * 2 * f[1] * t)
            # Gear mesh vibration and harmonic for a pair of gears
            gm2_signal = 4 * np.sin(2 * np.pi * f[2] * t) + 4 * np.sin(2 * np.pi * 2 * f[2] * t)
            # Fault component signal
            fault_signal = 2 * (k / 6) * np.sin(2 * np.pi * 10 * f[0] * t)
            # New signal is the sum of gm1_signal, gm2_signal, and fault_signal.
            new_signal = shaft_signal + gm1_signal + gm2_signal + fault_signal

            # Calculate NA4
            na4_, cur_info = na4(new_signal, prev_info, fs, rpm, freq_list, n_harmonics)
            prev_info = cur_info
            na4_list.append(na4_)

    >>> print(na4_list)
    [1.536982703280907, 3.857738227803903, 5.590250485891509, 6.835250547872656, 7.755227746173137,
     8.457654233606695, 9.009683995782796, 9.454160950683573, 9.819352835339927, 10.12454304712786]
    """
    # Set default parameters
    if freq_list is None:
        freq_list = ()

    # Check inputs
    if not isinstance(x_tsa, np.ndarray):
        raise TypeError("`x_tsa` must be array.")
    if len(x_tsa.shape) >= 2:
        raise ValueError("`x_tsa` has less than 2 dimensions.")
    if not isinstance(prev_info, Tuple):
        raise TypeError("`prev_info` must be tuple.")
    if len(prev_info) != 2:
        raise ValueError("`prev_info` requires two elements.")
    if not (isinstance(prev_info[0], (int, float)) & isinstance(prev_info[-1], (int, float))):
        raise TypeError("The elements of `prev_info` must be integer or float.")
    if not (prev_info[0] >= 0):
        raise ValueError(
            "'The average of each 2th central moment of the residual signals' must be more than zero."
        )
    if not (prev_info[-1] >= 0):
        raise ValueError("'The time record number' must be more than zero.")
    if not isinstance(fs, (int, float)):
        raise TypeError("`fs` must be integer or float.")
    if not isinstance(rpm, (int, float)):
        raise TypeError("`rpm` must be integer or float.")
    if not isinstance(freq_list, Tuple):
        raise TypeError("`freq_list` must be tuple or None.")
    for e in freq_list:
        if not isinstance(e, (int, float)):
            raise TypeError("The elements of `freq_list` must be integer or float.")
    if not isinstance(n_harmonics, int):
        raise TypeError("`n_harmonics` must be integer.")

    # Add the `rps` to the filtering target frequency list.
    rps = rpm / 60
    freq_list = [f for f in freq_list]
    freq_list.append(rps)

    # Filter the range of a frequency in freq_list.
    n = x_tsa.size
    amp = np.fft.rfft(
        x_tsa,
    )
    freq = np.fft.rfftfreq(n, d=1 / fs)
    freq_band = rps  # Set the bandwidth of a center of the frequency to `rps`.

    for freq_center in freq_list:
        for k in range(1, n_harmonics + 1):
            harmonic_freq = freq_center * k
            filtered_indices = np.where(np.abs(freq - harmonic_freq) <= freq_band / 2)
            amp[filtered_indices] = 0

    # Get residual signal.
    res = np.fft.irfft(amp)

    # Get the previous information.
    prev_m, prev_avg_2th_moments = prev_info

    # Get the current information.
    cur_2th_moment = moment(res, 2)
    cur_m = prev_m + 1
    cur_avg_2th_moments = (prev_avg_2th_moments * prev_m + cur_2th_moment) / cur_m
    cur_info = (cur_m, cur_avg_2th_moments)

    # Caculate the NA4.
    na4 = (
        moment(res, 4) / (cur_avg_2th_moments) ** 2
    )  # Use the average variance up to the number of signals.
    return na4, cur_info

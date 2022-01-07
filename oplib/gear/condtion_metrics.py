"""Condition metrics for gear.

- Author: Kyunghwan Kim, Kangwhi Kim
- Contact: kyunghwan.kim@onepredict.com, kangwhi.kim@onepredict.com
"""

from typing import List, Union

import numpy as np
from scipy.stats import moment

from oplib.signal import bandstop_filter


def na4(
    x,
    fs: Union[int, float],
    rpm: float,
    orderlist: List[float],
    n_harmonics: int = 2,
    n_rotations: int = 1,
):
    """

    .. math::

    Parameters
    ----------
    x : np.array of shape ()

    n_harmonics : int, default=2
        A positive integer specifying the number of shaft
        and gear meshing frequency harmonics to remove.
    n_rotations : int, default=1
        A positive integer specifying the number of shaft rotations in the TSA signal X.

    Returns
    -------

    Examples
    --------
    """
    # Get residual signal
    ## gmf 는 각 기어에 대한 mesh frequency
    rps = rpm / 60
    gmf: List[float] = rps * np.array(orderlist)

    freq_tolerance = 0.0369

    ## 각 gmf의 highest harmonics (n개)를 filtering (bandstop)
    residual_signal = x
    for freq_center in gmf:
        for n in range(1, n_harmonics + 1):
            harmonic_freq = freq_center * n
            # NOTE: Consider freq_band argument
            freq_band = harmonic_freq * freq_tolerance
            l_cutoff = harmonic_freq - freq_band / 2
            h_cutoff = harmonic_freq + freq_band / 2
            residual_signal = bandstop_filter(residual_signal, fs, l_cutoff, h_cutoff, order=2)
    ## rotaional speed의 highest harmonics를 filtering
    for n in range(1, n_rotations + 1):
        harmonic_freq = rps * n
        freq_band = harmonic_freq * freq_tolerance
        l_cutoff = harmonic_freq - freq_band / 2
        h_cutoff = harmonic_freq + freq_band / 2
        residual_signal = bandstop_filter(residual_signal, fs, l_cutoff, h_cutoff, order=2)

    # Caculate NA4
    # Metrics based on ensemble of residual TSA signals

    import matplotlib.pyplot as plt
    from scipy.stats import kurtosis

    freq_x = np.fft.rfftfreq(x.size, 1 / fs)[:-1]
    freq_res = np.fft.rfftfreq(residual_signal.size, 1 / fs)[:-1]
    x_mag = abs((np.fft.rfft(x) / x.size)[:-1] * 2)
    res_mag = abs((np.fft.rfft(residual_signal) / residual_signal.size)[:-1] * 2)

    # plt.plot(freq_x, x_mag)
    # plt.plot(freq_res, res_mag)
    # plt.show()

    v = moment(residual_signal, 2)  # Variance at current time, k
    na4 = moment(residual_signal, 4) / np.mean(v) ** 2  # use average variance up to k
    plt.hist(residual_signal, 200, alpha=0.3)

    return na4, kurtosis(residual_signal)

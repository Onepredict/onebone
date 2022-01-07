"""Condition metrics for gear.

- Author: Kyunghwan Kim, Kangwhi Kim
- Contact: kyunghwan.kim@onepredict.com, kangwhi.kim@onepredict.com
"""

from typing import Tuple, Union

import numpy as np
from scipy.stats import moment


def na4(
    tsa_x_list: Tuple[np.ndarray, ...],
    fs: Union[int, float],
    rpm: float,
    freq_list: Tuple[float] = None,
    n_harmonics: int = 2,
):
    """

    .. math::

    Parameters
    ----------
    x : np.array of shape ()

    n_harmonics : int, default=2
        A positive integer specifying the number of shaft
        and gear meshing frequency harmonics to remove.

    Returns
    -------

    Examples
    --------
    """
    # Default hyperparameters
    if freq_list is None:
        freq_list = []
    else:
        freq_list = [f for f in freq_list]

    # Get residual signal
    rps = rpm / 60

    # Filtering할 주파수 성분
    freq_list.append(rps)

    # 각 gmf의 highest harmonics (n개)를 filtering (bandstop)
    v, na4_list = [], []
    for x in tsa_x_list:
        n = x.size
        # TODO: oplib fft로 교체 고려
        amp = np.fft.rfft(x)
        freq = np.fft.rfftfreq(n, d=1 / fs)
        freq_band = rps

        for freq_center in freq_list:
            for n in range(1, n_harmonics + 1):
                harmonic_freq = freq_center * n
                filtered_indices = np.where(np.abs(freq - harmonic_freq) <= freq_band / 2)
                amp[filtered_indices] = 0
        residual_signal = np.fft.irfft(amp)

        # Caculate NA4
        # Metrics based on ensemble of residual TSA signals
        # TODO: NA4가 res의 kurtosis가 아닐 수 있음
        v.append(moment(residual_signal, 2))  # Variance at current time, k
        na4 = moment(residual_signal, 4) / np.mean(v) ** 2  # use average variance up to k
        na4_list.append(na4)
        # print(moment(residual_signal, 4), np.mean(v))

    return na4_list

"""Condition metrics for gear.

- Author: Kibum Kim
- Contact: kibum.kim@onepredict.com
"""

from typing import Tuple, Union

import numpy as np


def phase_alignment(y: np.ndarray, fs: Union[int, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute phase alignment(PA) of a set of 1D signals

    .. math:: {PA_{f} = \mid {1 \over n} \sum_j{e^{iw_{j,f}}} \mid }

    Where :math:`PA_{f}` is phase alignment value on frequency :math:`f`,
    :math:`n` is the number of signals, and :math:`j` is the index of each signals.
    :math:`w_{j, f}` denotes phase of singal :math:`j` at frequency :math:`f`.


    This function computes mean vector of unit phase vectors of 1D signals.
    This process is repeated for every frequency unit.

    Parameters
    ----------
    y : numpy.ndarray of shape (n, signal_length)
        n denotes the number of signals, and each signal shold be 1D time domain.
    fs : int or float, default=1
        Sample rate. The sample rate is the number of samples per unit time.

    Returns
    -------
    freq : numpy.ndarray
        frequency array
    phase_alignment : numpy.ndarray
        phase alignment value of each frequency
    """
    if not isinstance(y, np.ndarray):
        raise TypeError(f"Argument 'y' must be of type numpy.ndarray, not {type(y).__name__}")
    if not isinstance(fs, (int, float)):
        raise TypeError("`fs` must be integer or float.")

    freq = np.fft.fftfreq(y.shape[-1], d=1 / fs)[: y.shape[-1] // 2]
    density = np.fft.fft(y)[:, : y.shape[-1] // 2]
    normed_density = density / np.abs(density)
    phase_alignment = np.abs(np.mean(normed_density, axis=0))

    return freq, phase_alignment

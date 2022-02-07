""" Track and extract a instaneous frequency profile from vibration signal

- Author: Kangwhi Kim
- Contact: kangwhi.kim@onepredict.com
"""

from typing import Union

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import stft


def _track_local_maxima(
    f: np.ndarray,
    t: np.ndarray,
    zxx: np.ndarray,
    f_start: Union[int, float],
    f_tol: Union[int, float],
) -> np.ndarray:
    """Track the maximum frequency components over time in a local frequency range."""
    zxx = np.abs(zxx)
    n = t.size
    inst_freq = np.zeros(n)
    inst_freq[0] = f_start
    f_max = f_start
    for i in range(1, n):
        f_range_indices = np.where(np.abs(f - f_max) <= f_tol)
        if f_range_indices[0].size == 0:
            raise ValueError(
                "Frequency cannot be tracked within a certain range `f_tol`. \
                 Set `f_tol` or `nperseg` to a higher value than the current value."
            )
        f_range = f[f_range_indices]
        zxx_maxima_indices = np.argmax(zxx[f_range_indices, i])
        f_max = f_range[zxx_maxima_indices]
        inst_freq[i] = f_max

    return inst_freq


def estimate_if(
    x: np.ndarray,
    fs: Union[int, float],
    f_start: Union[int, float],
    f_tol: Union[int, float],
    filter_bw: Union[int, float],
    window: str = "hann",
    nperseg: int = None,
    noverlap: int = None,
    **kwargs
) -> np.ndarray:

    # Check inputs
    if not isinstance(x, np.ndarray):
        raise TypeError("`x` must be array.")
    if len(x.shape) >= 2:
        raise ValueError("`x` has less than 2 dimensions.")
    if not isinstance(fs, (int, float)):
        raise TypeError("`fs` must be integer or float.")
    if not isinstance(fs, (int, float)):
        raise TypeError("`f_start` must be integer or float.")
    if not isinstance(fs, (int, float)):
        raise TypeError("`f_tol` must be integer or float.")
    if not isinstance(fs, (int, float)):
        raise TypeError("`filter_bw` must be integer or float.")

    # Get the size of the signal.
    n = x.size
    # Compute the Short Time Fourier Transform (STFT).
    f, t, zxx = stft(x, fs, window, nperseg, noverlap, **kwargs)

    # Extract the frequency components along time using the two-step method.
    freq_components = _track_local_maxima(f, t, zxx, f_start, f_tol)

    # Make the size of frequency components equal to the size of the signal.
    freq_components = interp1d(np.linspace(0, 1, t.size), freq_components)
    freq_components = freq_components(np.linspace(0, 1, n))

    # Convert frequency components into phase components.
    phase_components = 2 * np.pi * np.cumsum(freq_components) / fs

    # Filter components excluding the above frequency components.
    x_pc = x * np.exp(-1j * phase_components)
    fft_x_pc = np.fft.fft(x_pc)
    indices_filtered = np.ceil((filter_bw / 2) / (fs / n)).astype(int)
    fft_x_pc[indices_filtered:-indices_filtered] = 0
    xf = np.fft.ifft(2 * fft_x_pc) * np.exp(1j * phase_components)

    # Get the instaneous frequency of signal.
    phase = np.unwrap(np.angle(xf))
    inst_freq = np.diff(phase) / (1 / fs) / (2 * np.pi)

    return inst_freq

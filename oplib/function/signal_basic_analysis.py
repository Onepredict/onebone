from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from convert_to_x import do_fft
from freq_filter import bandpass_filter
from scipy.signal import stft
from scipy.stats import kurtosis


def plot_signal(signal: np.ndarray, sr: float, is_angle: bool = False) -> None:
    """Plot the signal.

    Parameters
    ----------
    signal : ndarray
        One-dimensional signal
    sr : float
        Sampling rate
    is_angle : bool, default=False
        Whether the input data is in angle domain.
    """
    time = np.arange(signal.size) * 1 / sr
    plt.plot(time, signal)
    plt.title("Signal")
    if is_angle is True:
        plt.xlabel("Angle[rad]")
    else:
        plt.xlabel("Time[s]")


def plot_fft(
    signal: np.ndarray, sr: float, is_angle: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Do Fast Fourier Transfrom (is_angle=True인 경우, Order Analysis 표현)

    Parameters
    ----------
    signal : ndarray
        One-dimensional signal
    sr : float
        Sampling rate
    is_angle : bool, default=False
        Whether the input data is in angle domain.

    Returns
    -------
    fft_freq : ndarray
        Array of the sample frequencies.
    fft_amp : ndarray
        Array of the amplitude.
    """
    fft_freq, fft_amp = do_fft(signal, sr)
    if is_angle is True:
        fft_freq *= 2 * np.pi

    plt.plot(fft_freq, fft_amp)
    plt.ylabel("Amp")
    if is_angle is True:
        plt.title("Order Analysis")
        plt.xlabel("Order[cycle/rev]")
    else:
        plt.title("FFT Magnitude")
        plt.xlabel("Frequeny[Hz]")

    return fft_freq, fft_amp


def plot_stft(
    signal: np.ndarray, sr: float, is_angle: bool = False, stft_kwarg=None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Short Time Fourier Transform 수행 (is_angle=True인 경우, Order Spectra 표현)

    Parameters
    ----------
    signal : ndarray
        One-dimensional signal
    sr : float
        Sampling rate
    is_angle : bool, default=False
        Whether the input data is in angle domain.
    stft_kwarg : dict
        hyperparameters of scipy.signal.stft function

    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    t : ndarray
        Array of segment times.
    Zxx : ndarray
        STFT of x. By default, the last axis of Zxx corresponds to the segment times.

    """
    if stft_kwarg is None:
        stft_kwarg = {}

    f, t, Zxx = stft(signal, sr, **stft_kwarg)
    if is_angle is True:
        f *= 2 * np.pi  # [cycle/rad] to [order]
        t /= 2 * np.pi  # [rad] to [cycle]
    plt.pcolormesh(t, f, np.log(np.abs(Zxx)), shading="gouraud")
    if is_angle is True:
        plt.title("Order Spectra")
        plt.xlabel("Revolution[rev]")
        plt.ylabel("Order[cycle/rev]")
    else:
        plt.title("STFT Magnitude")
        plt.xlabel("Time[s]")
        plt.ylabel("Frequeny[Hz]")

    return f, t, Zxx


def show():
    """Show the plot"""
    return plt.show()


def plot_spectrual_kurotosis(
    signal: np.ndarray,
    sr: float,
    n_row: int = None,
    is_angle: bool = False,
) -> Tuple[np.ndarray, float, float]:
    """Plot spectrual kurtosis

    Parameters
    ----------
    signal : ndarray
        One-dimensional signal
    sr : float
        Sampling rate
    n_row : int
        Number of Level
    is_angle : bool, default=False
        Whether the input data is in angle domain.

    Returns
    -------
    sk_matrix : ndarray
        Matrix of spectrual kurtosis
    max_kurt_center_freq : float
        Center frequency at maximum kurtosis
    max_kurt_bandwidth : float
        Bandwidth at maximum kurtosis
    """
    max_freq = sr / 2
    if n_row is None:
        n_row = int(np.log2(max_freq) + 1)
    n_col = 2 ** (n_row - 1)
    sk_matrix = np.zeros([n_row, n_col])
    old_kurt = 0
    for row in range(n_row):
        n_seg = 2 ** row
        indices_length = n_col / n_seg
        seg_indices = np.arange(n_seg + 1) * indices_length
        seg_indices = seg_indices.astype(int)
        bandwidth = max_freq / n_seg
        center_freqs = np.arange(1, 2 * n_seg, 2) * bandwidth / 2

        for i, center_freq in enumerate(center_freqs):
            freq_low = center_freq - bandwidth / 2
            freq_low = max(freq_low, 1e-6)
            freq_high = center_freq + bandwidth / 2
            freq_high = min(freq_high, max_freq - 1e-6)
            idx_low = seg_indices[i]
            idx_high = seg_indices[i + 1]
            filtered_signal = bandpass_filter(signal, sr, freq_low, freq_high, 1)
            kurt = kurtosis(filtered_signal)
            sk_matrix[row, idx_low:idx_high] = kurt
            if kurt > old_kurt:
                max_kurt = kurt
                max_kurt_level = row
                max_kurt_center_freq = center_freq
                max_kurt_bandwidth = bandwidth
                if is_angle is True:
                    max_kurt_center_freq *= 2 * np.pi
                    max_kurt_bandwidth *= 2 * np.pi
                optimal_window_length = n_seg
                old_kurt = kurt

    # Plot spectural kurtosis
    coor_level = np.arange(n_row)
    coor_center_freq = np.arange(1, 2 * n_col, 2) * (max_freq / n_col) / 2
    if is_angle is True:
        coor_center_freq *= 2 * np.pi

    fig, ax = plt.subplots(1, 1)
    pcm = ax.pcolormesh(coor_center_freq, coor_level, sk_matrix, cmap="viridis", shading="nearest")
    fig.colorbar(pcm, ax=ax, label="Spectral Kurtosis")

    if is_angle is True:
        ax.set_title(
            f"Max kurtosis = {max_kurt:.2f} at level {max_kurt_level}, "
            + f"Optimal Window Length = {optimal_window_length}\n"
            + f"Center Order = {max_kurt_center_freq:.2f}, "
            + f"Bandwidth = {max_kurt_bandwidth:.2f}"
        )
        ax.set_xlabel("Order[cycle/rev]")
    else:
        ax.set_title(
            f"Max kurtosis = {max_kurt:.2f} at level {max_kurt_level}, "
            + f"Optimal Window Length = {optimal_window_length}\n"
            + f"Center Frequency = {max_kurt_center_freq:.2f} Hz, "
            + f"Bandwidth = {max_kurt_bandwidth:.2f} Hz"
        )
        ax.set_xlabel("Frequency[Hz]")

    ax.set_ylabel("Level")
    ax.invert_yaxis()
    plt.show()

    return sk_matrix, max_kurt_center_freq, max_kurt_bandwidth

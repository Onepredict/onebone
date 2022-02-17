""" Track and extract a instantaneous frequency(IF) profile from vibration signal

- Author: Kangwhi Kim
- Contact: kangwhi.kim@onepredict.com
"""

from typing import Tuple, Union

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import stft


def _track_local_maxima(
    f: np.ndarray,
    t: np.ndarray,
    coeff_matrix: np.ndarray,
    f_start: Union[int, float],
    f_tol: Union[int, float],
) -> np.ndarray:
    """Track the maximum frequency components over time in a local frequency range."""
    amp_matrix = np.abs(coeff_matrix)
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
        maxima_amp_indices_matrix = np.argmax(amp_matrix[f_range_indices, i])
        f_max = f_range[maxima_amp_indices_matrix]
        inst_freq[i] = f_max

    return inst_freq


def two_step_if(
    x: np.ndarray,
    fs: Union[int, float],
    f_start: Union[int, float],
    f_tol: Union[int, float],
    filter_bw: Union[int, float],
    window: Union[str, Tuple, np.ndarray] = "hann",
    nperseg: int = 256,
    noverlap: int = None,
    **kwargs
) -> np.ndarray:
    """
    Track and extract a instantaneous frequency(IF) profile from vibration signal, based on Two-step method.

    .. note:: If you have a tachometer pulse signal, use `tacho_to_rpm` function.

    `two_step_if` uses the local maxima technique :math:`{}^{[1]}` for IF estimation, as follows;

    .. math::
        f_{max}(t) = \\underset{f}{Argmax}{\\left|{X(t,f)}\\right|}^2,\
        \\quad\\mathrm{for}\\; f \\in \\mathit{\\Delta}f_t
        \\\\
        \\\\
        \\mathit{\\Delta}f_t \\subset \\left\\{f_{max}(t-d\\tau)-\\delta f,\
        \\; f_{max}(t-d\\tau)+\\delta f\\right\\}

    where, :math:`{\\delta f}` is the given frequency tolerance for maxima detection,
    :math:`X(t,f)` is the STFT of signal :math:`x(t)`
    computed for frequency values in set :math:`\\mathit{\\Delta}f_t`,
    specified from the previous estimate :math:`f_{max}(t-d\\tau)`.
    \\tau is time defined as a window position.
    Note that for t=0, :math:`\\mathit{\\Delta}f_t` should be given by the user.


    Parameters
    ----------
    x : numpy.ndarray of shape (length of `x`, )
        A vibration 1-D signal.
    fs : int or float
        Sampling rate. [Hz]
    f_start : int or float
        Starting frequency point for the IF estimation. [Hz]
    f_tol : int or float
        Frequency tolerance for maxima detection. [Hz]
    filter_bw : int or float
        frequency bandwidth for filtration. [Hz]
    window : str or tuple or numpy.ndarray, default="hann"
        Desired window to use for `scipy.signal.stft`.
        If window is a string or tuple, it is passed to get_window to generate the window values,
        which are DFT-even by default. See get_window for a list of windows and required parameters.
        If window is array_like it will be used directly as the window
        and its length must be nperseg. Defaults to a Hann window.
    nperseg : int, default=256
        Length of each segment for `scipy.signal.stft`.
    noverlap : int, default=None
        Number of points to overlap between segments.
        If None, noverlap = nperseg // 2. Defaults to None.
        When specified, the COLA constraint must be met;
        i.e. (x.shape[axis] - nperseg) % (nperseg-noverlap) == 0.
        For more information, see Notes of `scipy.signal.stft`.
    **kwargs : dict
        Additional parameters for `scipy.signal.stft`.


    Returns
    -------
    inst_freq : numpy.ndarray of shape (`x.size` - 1,)
        A instantaneous frequency(IF) profile.
        For improved results try to manipulate `f_tol` and `f_tol` parameters.
        You might also change spectrogram options.


    References
    ----------
    .. [1] Jacek Urbanek, Tomasz Barszcz, Jerome Antoni. (2013).
           A two-step procedure for estimation of instantaneous rotational speed with large fluctuations.
           https://doi.org/10.1016/j.ymssp.2012.05.009.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    Generate a test signal, sin wave whose frequency is modulated around 3kHz, corrupted by white noise.

    >>> fs = 1e4
    >>> n = 1e5
    >>> time = np.arange(n) / fs
    >>> mod = 500 * np.cos(2 * np.pi * 0.1 * time)
    >>> carrier = 3 * np.sin(2 * np.pi * 3e3 * time + mod)
    >>> x = carrier + np.random.rand(carrier.size) / 5  # test signal

    Extract the instantaneous frequency from the signal.

    >>> inst_freq = two_step_if(x, fs, f_start=3e3, f_tol=50, filter_bw=5,
    window='hann', nperseg=4096, noverlap=3985)

    Plot the instantaneous frequency(IF) profile.

    >>> time = np.arange(x.size) / fs
    >>> time = time[:-1]
    >>> plt.plot(time, inst_freq)
    >>> plt.title('Estimated the instantaneous frequency(IF) profile')
    >>> plt.ylabel('Frequency [Hz]')
    >>> plt.xlabel('Time [sec]')
    >>> plt.show()
    """

    # Check inputs
    if not isinstance(x, np.ndarray):
        raise TypeError("`x` must be array.")
    if len(x.shape) >= 2:
        raise ValueError("`x` must have 1 dimension.")
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
    f, t, coeff_matrix = stft(x, fs, window, nperseg, noverlap, **kwargs)

    # Estimate the preliminary instantaneous frequency using the two-step method.
    pre_inst_freq = _track_local_maxima(f, t, coeff_matrix, f_start, f_tol)

    # Make the size of preliminary instantaneous frequency equal to the size of the signal.
    freq_comp_interp = interp1d(np.linspace(0, 1, t.size), pre_inst_freq)
    freq_comp = freq_comp_interp(np.linspace(0, 1, n))

    # Convert the preliminary instantaneous frequency into the instaneous phase.
    inst_phase = 2 * np.pi * np.cumsum(freq_comp) / fs

    # Filter the signal around the harmonic component of interest in a narrow angular frequency band.
    x_pc = x * np.exp(-1j * inst_phase)
    fft_x_pc = np.fft.fft(x_pc)
    indices_filtered = np.ceil((filter_bw / 2) / (fs / n)).astype(int)
    fft_x_pc[indices_filtered:-indices_filtered] = 0
    xf = np.fft.ifft(2 * fft_x_pc) * np.exp(1j * inst_phase)

    # Get the instantaneous frequency of signal.
    phase = np.unwrap(np.angle(xf))
    inst_freq = np.diff(phase) / (1 / fs) / (2 * np.pi)

    return inst_freq

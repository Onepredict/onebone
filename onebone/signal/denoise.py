"""Signal denoising method.

- Author: Kyunghwan Kim
- Contact: kyunghwan.kim@onepredict.com
"""

import numpy as np
import pywt


def wavelet_denoising(
    signal: np.ndarray, wavelet: str, axis: int = -1, level: int = None
) -> np.ndarray:
    """
    Denoise signal using Discrete Wavelet Transform(DWT).

        1. Multilevel Wavelet Decomposition.
        2. Identify a thresholding technique.
        3. Threshold and Reconstruct.

    Parameters
    ----------
    signal : numpy.ndarray
        Input signal.
    wavelet : str
        Wavelet name.
        See this `page <http://wavelets.pybytes.com/>`_.

    axis : int, default=-1
        Axis over which to compute the DWT.
    level : int, default=None
        If level is None, then it will be calculated using
        the ``pywt.dwt_max_level`` function.

    Returns
    -------
    out : numpy.ndarray
        Denoised signal.

    Examples
    --------
    Apply the filter to 1d signal.

    >>> signal = np.array([10.0] * 10 + [0.0] * 10)
    >>> signal += np.random.random(signal.shape)

    .. image:: https://user-images.githubusercontent.com/86699249/157188089-5b000e5b-fcca-4e37-aed5-c6f49aecc50a.png  # noqa
        :width: 300

    >>> wavelet = "db1"
    >>> denoised_signal = wavelet_denoising(signal, wavelet, level=2)

    .. image:: https://user-images.githubusercontent.com/86699249/157188361-9fa41f1a-bac0-4ddd-822d-d9aa8c3616bf.png  # noqa
        :width: 300

    """
    coeff = pywt.wavedec(signal, wavelet, axis=axis, level=level)

    # Universal threshold
    threshold = np.sqrt(2 * np.log(signal.size)) * np.median(np.abs(coeff[-1])) / 0.6745

    coeff[1:] = (pywt.threshold(c, value=threshold, mode="soft") for c in coeff[1:])
    denoised_signal = pywt.waverec(coeff, wavelet, axis=axis)
    return denoised_signal

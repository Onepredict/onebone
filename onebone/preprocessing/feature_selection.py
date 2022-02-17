"""Feature Selection methods.

- Author: Junha Jeon
- Contact: junha.jeon@onepredict.com
"""

import numpy as np
import scipy.signal as signal


def fs_crosscorrelation(x: np.ndarray, refer: np.ndarray, output_col_num: int) -> np.ndarray:
    """
    .. note:: This method uses `scipy.signal.correlate`_.
    .. _scipy.signal.correlate:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html

    Reduce the dimension of input data by removing the signals
    which have small cross correlation with reference signal.


    Parameters
    ----------
    x : numpy.ndarray of shape (data_length, n_features)
        The data.
    refer : numpy.ndarray of shape (data_length,)
        The reference data.
    output_col_num : int
        Number of columns after dimension reduction.

    Returns
    -------
    x_tr : numpy.ndarray of shape (data_length, n_features)
        The data after dimension reduction.

    Examples
    --------
    >>> t = np.linspace(0, 1, 1000)
    >>> a = 1.0 * np.sin(2 * np.pi * 30.0 * t)
    >>> b = 5.0 * np.sin(2 * np.pi * 30.0 * t)
    >>> x = np.stack([a, b], axis=1)
    >>> x.shape
    (1000, 2)
    >>> refer = 1.0 * np.sin(2 * np.pi * 10.0 * t)
    >>> x_dimreduced = fs_crosscorrelation(x, refer, output_col_num=1)
    >>> x_dimreduced.shape
    (1000, 1)
    """
    if len(x.shape) != 2:
        raise ValueError("'x' must have 2 dimensions")
    if output_col_num > x.shape[1]:
        raise ValueError("'output_col_num' must be smaller than x.shape[1]")
    max_abs_ccorr_list = []
    for col_num in range(x.shape[1]):
        ccorr = signal.correlate(refer, x[:, col_num])
        max_abs_ccorr = np.absolute(ccorr).max()
        max_abs_ccorr_list.append(max_abs_ccorr)

    selected_col = np.array(max_abs_ccorr_list).argsort()[::-1][:output_col_num]
    return x[:, selected_col]

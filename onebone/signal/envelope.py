"""Extract envelope.

- Author: Kangwhi Kim
- Contact: kangwhi.kim@onepredict.com
"""

import numpy as np
from scipy.signal import hilbert


def envelope_hilbert(x, axis: int = -1) -> np.ndarray:

    """
    Extract the envelope from the signal using the 'Hilbert transform'.

    Parameters
    ----------
    x : array_like
        Signal data. Must be real.
    axis : int, default=-1
        Axis along which to do the transformation.

    Returns
    -------
    y : numpy.ndarray
        Envelope of the `x`, of each 1-D array along `axis`
    """
    if np.iscomplexobj(x):
        raise ValueError("`x` must be real.")
    if not isinstance(axis, int):
        raise TypeError(f"`axis` must be integer, not {type(axis).__name__}.")
    y = np.abs(hilbert(x, axis=axis))

    return y

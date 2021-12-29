"""Extract envelope.

- Author: Kangwhi Kim
- Contact: kangwhi.kim@onepredict.com
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import hilbert


def envelope_analytic(x: ArrayLike, axis: int = -1) -> np.ndarray:

    """
    Extract the envelope from the signal using the 'Hilbert transform'.

    The transformation is done along the last axis by default.

    Parameters
    ----------
    x : array_like
        Signal data.  Must be real.
    axis : int, optional
        Axis along which to do the transformation.  Default: -1.

    Returns
    ----------
    y: numpy.ndarray
        Envelope of the `x`, of each 1-D array along `axis`
    """
    if np.iscomplexobj(x):
        raise ValueError("'x' must be real.")
    if not isinstance(axis, int):
        raise TypeError(f"'axis' must be integer, not {type(axis).__name__}.")
    y = np.abs(hilbert(x, axis=axis))

    return y

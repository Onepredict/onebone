import numpy as np


def slice_along_axis(arr: np.ndarray, s: slice, axis: int) -> np.ndarray:
    """
    Slice the values of the array within a certain range on the axis.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array.
    s : slice
        Range on the `axis`.
    axis : int
        Axis

    Returns
    -------
    arr_out : numpy.ndarray
        Sliced input array.
    """
    arr_out = arr.copy()  # shallow copy
    if axis == -1:
        lower_ndim, upper_ndim = len(arr_out.shape[:axis]), 0
    else:
        lower_ndim, upper_ndim = len(arr_out.shape[:axis]), len(arr_out.shape[axis + 1 :])
    indices = lower_ndim * (np.s_[:],) + (s,) + upper_ndim * (np.s_[:],)
    arr_out = arr_out[indices]

    return arr_out

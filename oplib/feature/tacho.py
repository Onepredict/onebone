"""Convert tacho to angle or RPM.

- Author: Kangwhi Kim
- Contact: kangwhi.kim@onepredict.com
"""

from typing import Tuple, Union

import numpy as np
from scipy.interpolate import interp1d


def _get_indices_when_value_jump(x: np.ndarray, gap: float) -> np.ndarray:
    """Find sample indices in which the sample value rises rapidly."""
    diff_x = np.diff(x)
    indices = np.where(diff_x >= gap)[0] + 1
    return indices


def _get_time_and_angles_of_each_turn(
    x: np.ndarray,
    fs: Union[int, float],
    state_levels_trh: Union[int, float],
    indices_trh: int,
    pulses_per_rev: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get time and angles of which the samples are rising."""
    t = np.arange(x.size) / fs
    event_indices = _get_indices_when_value_jump(x, state_levels_trh)
    event_indices = event_indices[_get_indices_when_value_jump(event_indices, indices_trh)]

    tp = t[event_indices]
    angles = np.arange(tp.size) * (2 * np.pi) / pulses_per_rev

    return tp, angles


def tacho_to_angle(
    x: np.ndarray,
    fs: Union[int, float],
    state_levels_trh: Union[int, float],
    indices_trh: int = 2,
    pulses_per_rev: int = 1,
    output_fs: Union[int, float] = None,
    fit_type: str = "linear",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract angle signal from tachometer pulses.

    Parameters
    ----------
    x : numpy.ndarray
        Tachometer pulse signal(1-D).
    fs : int or float
        Sample rate.
    state_levels_trh : int or float
        The difference between state levels used to identify pulses.
        (The state levels used to identify pulses.)
    indices_trh : int, default=2
        The difference between indices of the first samples of high-level-state of pulses.
    pulses_per_rev : int, default=1
        Number of tachometer pulses per revolution.
    output_fs : int or float, default=None
        Output sample rate. When the default is None, the `output_fs` is the `fs`.
    fit_type : str, default="linear"
        Fitting method

    Returns
    -------
    angle : numpy.ndarray
        Rotational angle(1-D).
    t : numpy.ndarray
        Time(1-D) expressed in seconds.
    tp : numpy.ndarray
        Pulse locations(1-D) expressed in seconds.

    Examples
    --------
    >>> x = np.array([0,1,0,1,0,0,1,0,0,1])
    >>> fs = 10
    >>> state_levels_trh = 0.5

    >>> angle, t, tp = tacho_to_angle(x, fs, state_levels_trh)

    >>> angle
    array([-6.28318531e+00, -4.18879020e+00, -2.09439510e+00,  1.16262283e-15,
           2.09439510e+00,  4.18879020e+00,  6.28318531e+00,  8.37758041e+00,
           1.04719755e+01,  1.25663706e+01])
    >>> t
    array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    >>> tp
    array([0.3, 0.6, 0.9])
    """
    # Set default parameter
    if output_fs is None:
        output_fs = fs

    # Check inputs
    if not isinstance(x, np.ndarray):
        raise TypeError("`x` must be array.")
    if len(x.shape) >= 2:
        raise ValueError("`x` has less than 2 dimensions.")

    if not (isinstance(fs, int) | isinstance(fs, float)):
        raise TypeError("`fs` must be integer or float.")

    if not (isinstance(state_levels_trh, int) | isinstance(state_levels_trh, float)):
        raise TypeError("`state_levels_trh` must be integer or float.")

    if not isinstance(indices_trh, int):
        raise TypeError("`indices_trh` must be integer.")

    if not isinstance(pulses_per_rev, int):
        raise TypeError("`pulses_per_rev` must be integer.")

    if not (isinstance(output_fs, int) | isinstance(output_fs, float)):
        raise TypeError("`output_fs must` be integer or float.")

    if not isinstance(fit_type, str):
        raise TypeError("`fit_type` must be string.")

    # Resample time for the sampling rate of `output_fs`
    inc = 1 / output_fs  # incrementals of time
    end_time = (x.size - 1) / fs
    t = np.arange(0, end_time + inc, inc)

    # Get time and angles of each turn
    tp, tp_angles = _get_time_and_angles_of_each_turn(
        x, fs, state_levels_trh, indices_trh, pulses_per_rev
    )

    # Return zero if there is no rising edge
    if tp.size == 0:
        angle = np.zeros(t.size)
        return angle, t, tp

    # Get angle profile using interpolation between each turn
    # TODO: which value
    if fit_type == "linear":
        angle_profile = interp1d(
            tp, tp_angles, kind="linear", bounds_error=False, fill_value="extrapolate"
        )
    elif fit_type == "cubic":
        angle_profile = interp1d(
            tp, tp_angles, kind="cubic", bounds_error=False, fill_value="extrapolate"
        )

    # Get angles for resampled time
    angle = angle_profile(t)

    return angle, t, tp


def tacho_to_rpm(
    x: np.ndarray,
    fs: Union[int, float],
    state_levels_trh: Union[int, float],
    indices_trh: int = 2,
    pulses_per_rev: int = 1,
    output_fs: Union[int, float] = None,
    fit_type: str = "linear",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract RPM signal from tachometer pulses.

    Parameters
    ----------
    x : numpy.ndarray
        Tachometer pulse signal(1-D).
    fs : int or float
        Sample rate.
    state_levels_trh : int or float
        The difference between state levels used to identify pulses.
        (The state levels used to identify pulses.)
    indices_trh : int, default=2
        The difference between indices of the first samples of high-level-state of pulses.
    pulses_per_rev : int, default=1
        Number of tachometer pulses per revolution.
    output_fs : int or float, default=None
        Output sample rate. When the default is None, the `output_fs` is the `fs`.
    fit_type : str, default="linear"
        Fitting method.

    Returns
    -------
    rpm : numpy.ndarray
        Rotational speed(1-D).
    t : numpy.ndarray
        Time(1-D) expressed in seconds.
    tp : numpy.ndarray
        Pulse locations(1-D) expressed in seconds.

    Examples
    --------
    >>> x = np.array([0,1,0,1,0,0,1,0,0,1])
    >>> fs = 10
    >>> state_levels_trh = 0.5

    >>> rpm, t, tp = tacho_to_rpm(x, fs, state_levels_trh)

    >>> rpm
    array([200., 200., 200., 200., 200., 200., 200., 200., 200., 200.])
    >>> t
    array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    >>> tp
    array([0.3, 0.6, 0.9])
    """
    # Set default parameter
    if output_fs is None:
        output_fs = fs

    # Check inputs
    if not isinstance(x, np.ndarray):
        raise TypeError("`x` must be array.")
    if len(x.shape) >= 2:
        raise ValueError("`x` has less than 2 dimensions.")

    if not (isinstance(fs, int) | isinstance(fs, float)):
        raise TypeError("`fs` must be integer or float.")

    if not (isinstance(state_levels_trh, int) | isinstance(state_levels_trh, float)):
        raise TypeError("`state_levels_trh` must be integer or float.")

    if not isinstance(indices_trh, int):
        raise TypeError("`indices_trh` must be integer.")

    if not isinstance(pulses_per_rev, int):
        raise TypeError("`pulses_per_rev` must be integer.")

    if not (isinstance(output_fs, int) | isinstance(output_fs, float)):
        raise TypeError("`output_fs` must be integer or float.")

    if not isinstance(fit_type, str):
        raise TypeError("`fit_type` must be string.")

    # Resample time for the sampling rate of `output_fs`
    inc = 1 / output_fs  # incrementals of time
    end_time = (x.size - 1) / fs
    t = np.arange(0, end_time + inc, inc)

    # Get time and angles of each turn
    tp, tp_angles = _get_time_and_angles_of_each_turn(
        x, fs, state_levels_trh, indices_trh, pulses_per_rev
    )

    # Return zero if there is no rising edge
    if tp.size == 0:
        angle = np.zeros(t.size)
        return angle, t, tp

    # Calculate the RPM for each rotation section.
    rpm_t = tp[:-1] + np.diff(tp) / 2
    rpm = np.diff(tp_angles) / np.diff(tp) / (2 * np.pi) * 60

    # Get angle profile using interpolation between each turn
    # TODO: which value
    if fit_type == "linear":
        rpm_profile = interp1d(
            rpm_t, rpm, kind="linear", bounds_error=False, fill_value="extrapolate"
        )
    elif fit_type == "cubic":
        rpm_profile = interp1d(
            rpm_t, rpm, kind="cubic", bounds_error=False, fill_value="extrapolate"
        )

    # Get RPM for resampled time
    rpm = rpm_profile(t)

    return rpm, t, tp

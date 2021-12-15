import os
import sys
from typing import Tuple

import numpy as np
from scipy.interpolate import interp1d

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.exception import ArrayShapeError, DataError  # noqa


def _find_rising_indices(x: np.ndarray, gap: float) -> np.ndarray:
    diff_x = np.diff(x)
    indices = np.where(diff_x >= gap)[0] + 1
    return indices


def _get_pulse_location(
    x: np.ndarray, fs: int, state_levels_trh: float, indices_trh: int, purses_per_rev: int
):
    t = np.arange(x.size) / fs
    event_indices = _find_rising_indices(x, state_levels_trh)
    event_indices = event_indices[_find_rising_indices(event_indices, indices_trh)]

    tp = t[event_indices]
    angles = np.arange(tp.size) * (2 * np.pi) / purses_per_rev

    return tp, angles


def tacho_to_angle(
    x,
    fs,
    state_levels_trh,
    indices_trh=2,
    purses_per_rev=1,
    output_fs=None,
    fit_type="linear",
):
    """
    Extract angle signal from tachometer pulses

    Parameters
    ----------
    x: numpy.ndarray
        Tachometer pulse signal(1-D).
    fs: int
        Sample rate.
    state_levels_trh: float
        The difference between state levels used to identify pulses.
        (The state levels used to identify pulses.)
    indices_trh: int
        The difference between indices of the first samples of high-level-state of pulses.
        The default is '2' in order to identify pulses.
    purses_per_rev: int
        Number of tachometer pulses per revolution.
    output_fs: int
        Output sample rate.
    fit_type: str
        Fitting method.

    Returns
    -------
    angle: numpy.ndarray
        Rotational angle(1-D).
    t: numpy.ndarray
        Time(1-D) expressed in seconds.
    tp: numpy.ndarray
        Pulse locations(1-D) expressed in seconds.

    Examples
    --------
    >>> x = np.array([0,1,0,1,0,0,1,0,0,1])
    >>> fs = 10
    >>> state_levels_trh = 0.5

    >>> angle, t, tp = tacho_to_angle(x, fs, state_levels_trh)

    >>> angle
    array([           nan,            nan,            nan, 1.16262283e-15,
       2.09439510e+00, 4.18879020e+00, 6.28318531e+00, 8.37758041e+00,
       1.04719755e+01, 1.25663706e+01])
    >>> t
    array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    >>> tp
    array([0.3, 0.6, 0.9])
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("X must be array.")

    if len(x.shape) >= 2:
        raise ArrayShapeError("The array dimensions must be 1-dimension.")

    if len(set(x)) == 1:
        raise DataError("There is no pulse.")

    if output_fs is None:
        output_fs = fs

    tp, tp_angles = _get_pulse_location(x, fs, state_levels_trh, indices_trh, purses_per_rev)

    if fit_type == "linear":
        angle_profile = interp1d(
            tp, tp_angles, kind="linear", bounds_error=False, fill_value=np.nan
        )
    elif fit_type == "cubic":
        angle_profile = interp1d(tp, tp_angles, kind="cubic", bounds_error=False, fill_value=np.nan)

    origin_t = np.arange(x.size) / fs
    t = np.arange(origin_t[0], origin_t[-1] + 1 / output_fs, 1 / output_fs)
    angle = angle_profile(t)

    return angle, t, tp


def tacho_to_rpm(
    x: np.ndarray,
    fs: int,
    state_levels_trh: float,
    indices_trh: int = 2,
    purses_per_rev: int = 1,
    output_fs: int = None,
    fit_type: str = "linear",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract RPM signal from tachometer pulses

    Parameters
    ----------
    x: numpy.ndarray
        Tachometer pulse signal(1-D).
    fs: int
        Sample rate.
    state_levels_trh: float
        The difference between state levels used to identify pulses.
        (The state levels used to identify pulses.)
    indices_trh: int
        The difference between indices of the first samples of high-level-state of pulses.
        The default is '2' in order to identify pulses.
    purses_per_rev: int
        Number of tachometer pulses per revolution.
    output_fs: int
        Output sample rate.
    fit_type: str
        Fitting method.

    Returns
    -------
    rpm: numpy.ndarray
        Rotational speed(1-D).
    t: numpy.ndarray
        Time(1-D) expressed in seconds.
    tp: numpy.ndarray
        Pulse locations(1-D) expressed in seconds.

    Examples
    --------
    >>> x = np.array([0,1,0,1,0,0,1,0,0,1])
    >>> fs = 10
    >>> state_levels_trh = 0.5

    >>> rpm, t, tp = tacho_to_rpm(x, fs, state_levels_trh)

    >>> rpm
    array([ nan,  nan,  nan,  nan,  nan, 200., 200., 200.,  nan,  nan])
    >>> t
    array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    >>> tp
    array([0.3, 0.6, 0.9])
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("X must be array.")

    if len(x.shape) >= 2:
        raise ArrayShapeError("The array dimensions must be 1-dimension.")

    if len(set(x)) == 1:
        raise DataError("There is no pulse.")

    if output_fs is None:
        output_fs = fs

    tp, tp_angles = _get_pulse_location(x, fs, state_levels_trh, indices_trh, purses_per_rev)
    rpm_t = tp[:-1] + np.diff(tp) / 2
    rpm = np.diff(tp_angles) / np.diff(tp) / (2 * np.pi) * 60

    if fit_type == "linear":
        rpm_profile = interp1d(rpm_t, rpm, kind="linear", bounds_error=False, fill_value=np.nan)
    elif fit_type == "cubic":
        rpm_profile = interp1d(rpm_t, rpm, kind="cubic", bounds_error=False, fill_value=np.nan)

    origin_t = np.arange(x.size) / fs
    t = np.arange(origin_t[0], origin_t[-1] + 1 / output_fs, 1 / output_fs)
    rpm = rpm_profile(t)

    return rpm, t, tp

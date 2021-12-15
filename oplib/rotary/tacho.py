import os
import sys

import numpy as np
from scipy.interpolate import interp1d

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.exception import ArrayShapeError, DataError  # noqa


def _find_rising_indices(x: np.ndarray, gap: float) -> np.ndarray:
    """1-D Array에서 이전과 다음 샘플값 차이가 설정값 이상인 index를 산출

    Parameters
    ----------
    x: numpy.ndarray
        1차원 data
    gap: float
        간극 설정값

    Returns
    -------
    indices: numpy.ndarray
        조건에 해당하는 Input data의 인덱스값 배열
    """
    diff_x = np.diff(x)
    indices = np.where(diff_x >= gap)[0] + 1
    return indices


def _get_pulse_location(x, fs, state_levels_gap, high_level_indices_gap, purses_per_rev):
    t = np.arange(x.size) / fs
    event_indices = _find_rising_indices(x, state_levels_gap)
    event_indices = event_indices[_find_rising_indices(event_indices, high_level_indices_gap)]

    tp = t[event_indices]
    angles = np.arange(tp.size) * (2 * np.pi) / purses_per_rev

    return tp, angles


def tacho_to_angle(
    x,
    fs,
    state_levels_gap,
    high_level_indices_gap=2,
    purses_per_rev=1,
    output_fs=None,
    fit_type="linear",
):
    if not isinstance(x, np.ndarray):
        raise TypeError("X must be array.")

    if len(x.shape) >= 3:
        raise ArrayShapeError(3)

    if len(set(x)) == 1:
        raise DataError("There is no tacho pulse.")

    if output_fs is None:
        output_fs = fs

    # get 'time of pulse location'(tp)
    tp, tp_angles = _get_pulse_location(
        x, fs, state_levels_gap, high_level_indices_gap, purses_per_rev
    )

    if fit_type == "linear":
        angle_profile = interp1d(
            tp, tp_angles, kind="linear", bounds_error=False, fill_value=np.nan
        )
    elif fit_type == "cubic":
        angle_profile = interp1d(tp, tp_angles, kind="cubic", bounds_error=False, fill_value=np.nan)

    origin_t = np.arange(x.size) / fs
    t = np.arange(origin_t[0], origin_t[-1], 1 / output_fs)
    angle = angle_profile(t)

    return angle, t, tp


def tacho_to_rpm(
    x,
    fs,
    state_levels_gap,
    high_level_indices_gap=2,
    purses_per_rev=1,
    output_fs=None,
    fit_type="linear",
):
    if not isinstance(x, np.ndarray):
        raise TypeError("X must be array.")

    if len(x.shape) >= 3:
        raise ArrayShapeError(3)

    if output_fs is None:
        output_fs = fs

    # get 'time of pulse location'(tp)
    tp, tp_angles = _get_pulse_location(
        x, fs, state_levels_gap, high_level_indices_gap, purses_per_rev
    )

    rpm_t = tp[:-1] + np.diff(tp) / 2
    rpm = np.diff(tp_angles) / np.diff(tp) / (2 * np.pi) * 60

    if fit_type == "linear":
        rpm_profile = interp1d(rpm_t, rpm, kind="linear", bounds_error=False, fill_value=np.nan)
    elif fit_type == "cubic":
        rpm_profile = interp1d(rpm_t, rpm, kind="cubic", bounds_error=False, fill_value=np.nan)

    origin_t = np.arange(x.size) / fs
    t = np.arange(origin_t[0], origin_t[-1], 1 / output_fs)
    rpm = rpm_profile(t)

    return rpm, t, tp

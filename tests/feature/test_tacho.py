"""Test code for tacho.py.

- Author: Kangwhi Kim
- Contact: kangwhi.kim@onepredict.com
"""


import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.interpolate import interp1d

from oplib.feature import tacho_to_angle, tacho_to_rpm

# TEST_PARAMS = [rpm_level, profile_type, tacho_fs, state_levels, state_levels_trh]
TEST_PARAMS = [
    [3600, "sine", 25600, (0, 1), 0.5],
]

# THRESHOLD is allowable error
THRESHOLD = [0.03]


def _generate_tacho_signal(rpm_level, profile_type, tacho_fs, state_levels=None, pulse_per_rev=1):
    # Set default parameter
    if state_levels is None:
        state_levels = (0, 1)

    # Generate a RPM profile
    if profile_type == "linear":
        t = np.linspace(0, 3, 3000)
        rpm = rpm_level * t
    elif profile_type == "sine":
        t = np.linspace(0, 3, 3000)
        rpm = rpm_level * np.sin(2 * np.pi * t) + rpm_level
    elif profile_type == "cosine":
        t = np.linspace(0, 3, 3000)
        rpm = rpm_level * np.cos(2 * np.pi * t) + rpm_level
    elif profile_type == "random":
        t = np.linspace(0, 3, 3000)
        rpm = rpm_level * np.random.randn(3000)
        rpm -= np.min(rpm)
    else:
        raise ValueError(f"'{profile_type}' is not a valid value for profile_type")

    rpm_profile = interp1d(t, rpm, kind="linear")

    # Generate angle profile
    w = rpm / 60 * (2 * np.pi)  # angle velocity
    angles = np.cumsum(
        0.5 * np.diff(t) * (w[1:] + w[:-1])
    )  # TODO: It will be replaced with intergrate-function.
    angles = np.concatenate([np.zeros(1), angles])
    angle_profile = interp1d(t, angles, kind="linear", bounds_error=False, fill_value=0)

    # Generate tacho signal
    signal_t = np.arange(t[0], t[-1], 1 / tacho_fs)
    signal = np.ones(signal_t.size) * state_levels[0]

    angles = angle_profile(signal_t)
    angle_per_pulse = 2 * np.pi / pulse_per_rev
    pulse_count = angles // angle_per_pulse

    if set([0, 1]) | set(np.diff(pulse_count)) != set([0, 1]):
        raise ValueError(msg="The tacho sampling rate is too low.")

    rising_indices = np.where(np.diff(pulse_count) == 1)[0] + 1
    signal[rising_indices] = state_levels[-1]

    return signal, angle_profile, rpm_profile


def _relative_error(
    threshold, tacho_signal, tacho_fs, state_levels_trh, angle_profile, rpm_profile, is_plot=False
):
    # Estimate RPM
    angle_estimated, t, tp = tacho_to_angle(
        tacho_signal,
        tacho_fs,
        state_levels_trh,
    )
    rpm_estimated, _, _ = tacho_to_rpm(
        tacho_signal,
        tacho_fs,
        state_levels_trh,
    )

    # Relative error
    angle_true = angle_profile(t)  # True value of rotational angle
    angle_estimated += angle_true[
        np.where(angle_estimated >= 0)[0][0]
    ]  # Compensate for the phase difference between true angle and estimated angle
    rpm_true = rpm_profile(t)  # True value of RPM

    re_angle = abs(
        (np.sum(angle_true) - np.sum(angle_estimated)) / np.sum(angle_true)
    )  # Relative error for rotational angle
    re_rpm = abs(
        (np.sum(rpm_true) - np.sum(rpm_estimated)) / np.sum(rpm_true)
    )  # Relative error for RPM

    if is_plot is True:
        _, axes = plt.subplots(1, 2)
        axes[0].set_title("Angle Profile")
        axes[0].set_xlabel("Time[s]")
        axes[0].set_ylabel("Angle[rad]")
        axes[0].plot(t, angle_true, c="b", label="real angle")
        axes[0].scatter(t, angle_estimated, s=0.1, c="g", label="estimated angle")
        axes[0].scatter(tp, angle_profile(tp), s=30, c="r", marker="x", label="pulse rising")
        axes[0].legend()
        axes[1].set_title("RPM Profile")
        axes[1].set_xlabel("Time[s]")
        axes[1].set_ylabel("RPM[rev/min]")
        axes[1].plot(t, rpm_true, c="b", label="real rpm")
        axes[1].scatter(t, rpm_estimated, s=0.1, c="g", label="estimated rpm")
        axes[1].scatter(tp, rpm_profile(tp), s=30, c="r", marker="x", label="pulse rising")
        axes[1].legend()
        plt.show()

    assert (
        re_angle <= threshold
    ), f"Wrong Angle: angle-relative error: {re_angle}, threshold: {threshold}"
    assert re_rpm <= threshold, f"Wrong RPM: rpm-relative error: {re_rpm}, threshold: {threshold}"
    print(f"Angle-relative error: {re_angle}\tRPM-relative error: {re_rpm}")


def _check_array_shape(tacho_signal, tacho_fs, state_levels_trh):
    with pytest.raises(ValueError):
        tacho_to_angle(tacho_signal, tacho_fs, state_levels_trh)
        tacho_to_rpm(tacho_signal, tacho_fs, state_levels_trh)


def test_tacho(is_plot=False):
    for (rpm_level, profile_type, tacho_fs, state_levels, state_levels_trh), threshold in zip(
        TEST_PARAMS, THRESHOLD
    ):
        tacho_signal, angle_profile, rpm_profile = _generate_tacho_signal(
            rpm_level, profile_type, tacho_fs, state_levels
        )
        _relative_error(
            threshold, tacho_signal, tacho_fs, state_levels_trh, angle_profile, rpm_profile, is_plot
        )
        invalid_signal = np.stack([tacho_signal] * 2)
        _check_array_shape(invalid_signal, tacho_fs, state_levels_trh)


if __name__ == "__main__":
    test_tacho(is_plot=False)

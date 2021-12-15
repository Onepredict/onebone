import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from oplib.rotary.tacho import tacho_to_angle, tacho_to_rpm
from oplib.utils.exception import ValueLevelError

# TEST_PARAMS = [rpm_level, profile_type, tacho_fs]
TEST_PARAMS = [
    [3600, "sine", 25600],
]

# THRESHOLD is allowable error
THRESHOLD = [0.01]


def _generate_tacho_signal(t, rpm, tacho_fs, state_levels=None, pulse_per_rev=1):
    if min(rpm) < 0:
        raise ValueLevelError("RPM must be more than zero.")

    if state_levels is None:
        state_levels = (0, 1)

    # generate angular velocity profile
    rpm_profile = interp1d(t, rpm, kind="linear")
    t = np.linspace(t[0], t[-1], t.size * 10)  # resample for intergration
    rpm = rpm_profile(t)
    w = rpm / 60 * (2 * np.pi)  # angle velocity

    # generate angle profile
    angles = np.cumsum(
        0.5 * np.diff(t) * (w[1:] + w[:-1])
    )  # this line will be replaced with intergrate-function.
    angles = np.concatenate([np.zeros(1), angles])
    angle_profile = interp1d(t, angles, kind="linear", bounds_error=False, fill_value=np.nan)

    # generate tacho signal
    signal_t = np.arange(t[0], t[-1], 1 / tacho_fs)
    signal = np.ones(signal_t.size) * state_levels[0]

    angles = angle_profile(signal_t)
    angle_per_pulse = 2 * np.pi / pulse_per_rev
    pulse_count = angles // angle_per_pulse

    if set([0, 1]) | set(np.diff(pulse_count)) != set([0, 1]):
        raise ValueLevelError(msg="The tacho sampling rate is too low.")

    rising_indices = np.where(np.diff(pulse_count) == 1)[0] + 1
    signal[rising_indices] = state_levels[-1]

    return signal, angle_profile, rpm_profile


def _relative_error(threshold, rpm_level, profile_type, tacho_fs, is_plot=False):
    if profile_type == "linear":
        real_t = np.linspace(0, 3, 1000)
        real_rpm = rpm_level * real_t
    elif profile_type == "sine":
        real_t = np.linspace(0, 3, 1000)
        real_rpm = rpm_level * np.sin(2 * np.pi * real_t) + rpm_level
    elif profile_type == "cosine":
        real_t = np.linspace(0, 3, 1000)
        real_rpm = rpm_level * np.cos(2 * np.pi * real_t) + rpm_level
    elif profile_type == "random":
        real_t = np.linspace(0, 3, 1000)
        real_rpm = rpm_level * np.random.randn(1000)
        real_rpm -= np.min(real_rpm)
    else:
        raise ValueError(f"'{profile_type}' is not a valid value for profile_type")

    # generate tacho signal
    state_levels = (0, 1)
    tacho_signal, angle_profile, rpm_profile = _generate_tacho_signal(
        real_t, real_rpm, tacho_fs, state_levels
    )

    # estimate rpm
    state_levels_trh = 0.5
    angle, t, tp = tacho_to_angle(
        tacho_signal,
        tacho_fs,
        state_levels_trh,
    )
    rpm, _, _ = tacho_to_rpm(
        tacho_signal,
        tacho_fs,
        state_levels_trh,
    )

    # relative error
    a1 = angle_profile(t)[~np.isnan(angle)]
    a2 = angle[~np.isnan(angle)]
    a2 += a1[0]  # compensate for the phase difference between real angle and estimated angle
    r1 = rpm_profile(t)[~np.isnan(rpm)]
    r2 = rpm[~np.isnan(rpm)]
    are = abs((np.sum(a1 ** 2) - np.sum(a1 * a2)) / np.sum(a1 ** 2))  # angle-relative error
    rre = abs((np.sum(r1 ** 2) - np.sum(r1 * r2)) / np.sum(r1 ** 2))  # rpm-relative error

    if is_plot is True:
        _, axes = plt.subplots(1, 2)
        axes[0].set_title("Angle Profile")
        axes[0].set_xlabel("Time[s]")
        axes[0].set_ylabel("Angle[rad]")
        axes[0].plot(t, angle_profile(t), c="b", label="real angle")
        axes[0].scatter(t, angle + a1[0], s=0.1, c="g", label="estimated angle")
        axes[0].scatter(tp, angle_profile(tp), s=30, c="r", marker="x", label="pulse rising")
        axes[0].legend()
        axes[1].set_title("RPM Profile")
        axes[1].set_xlabel("Time[s]")
        axes[1].set_ylabel("RPM[rev/min]")
        axes[1].plot(t, rpm_profile(t), c="b", label="real rpm")
        axes[1].scatter(t, rpm, s=0.1, c="g", label="estimated rpm")
        axes[1].scatter(tp, rpm_profile(tp), s=30, c="r", marker="x", label="pulse rising")
        axes[1].legend()
        plt.show()

    assert are <= threshold, f"Wrong Angle: angle-relative error: {are}, threshold: {threshold}"
    assert rre <= threshold, f"Wrong RPM: rpm-relative error: {rre}, threshold: {threshold}"
    print(f"Angle-relative error: {are}\tRPM-relative error: {rre}")


def test_tacho(is_plot: bool = False):
    for (rpm_level, profile_type, tacho_fs), threshold in zip(TEST_PARAMS, THRESHOLD):
        _relative_error(threshold, rpm_level, profile_type, tacho_fs, is_plot=is_plot)


if __name__ == "__main__":
    tacho_to_rpm(np.array([0, 1, 0, 1, 0, 0, 1, 0, 0, 1]), 10, 0.5)[0]
    test_tacho(False)

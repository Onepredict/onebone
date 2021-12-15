import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from oplib.rotary.tacho import tacho_to_angle, tacho_to_rpm
from oplib.utils.exception import ValueLevelError

# TEST_PARAMS = [
#     [5000.0, 0.5, 0.02, 10.0],
# ]

# # [kurtosis]
# EXPECTED_ANSWER = [
#     [-1.4999999999999998],
# ]


def _generate_tacho_signal(t, rpm, tacho_fs, state_levels=None, pulse_per_rev=1):
    if min(rpm) < 0:
        raise ValueLevelError("RPM must be more than zero.")

    if state_levels is None:
        state_levels = (0, 1)

    # generate angular velocity profile
    w = rpm / 60 * (2 * np.pi)  # angle velocity
    w_profile = interp1d(t, w, kind="linear")
    t = np.linspace(t[0], t[-1], t.size * 5)  # resample for intergration
    w = w_profile(t)

    # generate angle profile
    angles = np.cumsum(
        0.5 * np.diff(t) * (w[1:] + w[:-1])
    )  # this line will be replaced with intergrate-function.
    angle_profile = interp1d(t[1:], angles, kind="linear")

    # generate tacho signal
    signal_t = np.arange(t[1], t[-1], 1 / tacho_fs)
    signal = np.ones(signal_t.size) * state_levels[0]

    angles = angle_profile(signal_t)
    angle_per_pulse = 2 * np.pi / pulse_per_rev
    pulse_count = angles // angle_per_pulse

    if set([0, 1]) | set(np.diff(pulse_count)) != set([0, 1]):
        raise ValueLevelError(msg="The tacho sampling rate is too low.")

    rising_indices = np.where(np.diff(pulse_count) == 1)[0] + 1
    signal[rising_indices] = state_levels[-1]

    return signal


if __name__ == "__main__":
    real_t = np.linspace(0, 10, 100)
    real_rpm = 1000 * np.random.randn(100) + 3600
    tacho_fs = 25600
    state_levels = (0, 1)
    pulse_per_rev = 1
    state_levels_gap = 0.5
    high_level_indices_gap = 2
    output_fs = tacho_fs

    tacho_signal = _generate_tacho_signal(real_t, real_rpm, tacho_fs, state_levels, pulse_per_rev)
    angle, t, tp = tacho_to_angle(
        tacho_signal, tacho_fs, state_levels_gap, high_level_indices_gap, pulse_per_rev, output_fs
    )
    rpm, _, _ = tacho_to_rpm(
        tacho_signal, tacho_fs, state_levels_gap, high_level_indices_gap, pulse_per_rev, output_fs
    )

    plt.plot(t, angle, label="estimated angle")
    plt.plot(t, rpm, label="estimated rpm")
    plt.plot(real_t, real_rpm, label="real rpm")
    plt.legend()
    plt.show()

    plt.plot(np.arange(tacho_signal.size) / tacho_fs, tacho_signal)
    plt.scatter(
        tp, np.ones(tp.size) * np.mean(tacho_signal), s=1, c="r", marker="x", label="pulse location"
    )
    ""

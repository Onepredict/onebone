import matplotlib.pyplot as plt
import numpy as np
from find_indices import find_jump_indices
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from scipy.signal import hilbert


class TachoToRPM(object):
    def __init__(self, time, tacho_signal, teeth):
        self.time = time
        self.tacho_signal = tacho_signal
        self.teeth = teeth

    def make_angle_and_rpm_profile(self, jump_signal_gap, interval_between_tacho=10, kind="linear"):
        event_indices = find_jump_indices(self.tacho_signal, jump_signal_gap)
        event_indices = event_indices[find_jump_indices(event_indices, interval_between_tacho)]

        self.event_times = self.time[event_indices]
        self.event_angles = np.arange(self.event_times.size) * (2 * np.pi) / self.teeth
        self.angle_profile = interp1d(
            self.event_times, self.event_angles, kind=kind, bounds_error=False, fill_value=np.nan
        )

        avg_between_event_times = self.event_times[:-1] + np.diff(self.event_times) / 2
        angle_vel = np.diff(self.event_angles) / np.diff(self.event_times)
        rpm = angle_vel / (2 * np.pi) * 60
        self.rpm_profile = interp1d(
            avg_between_event_times, rpm, kind=kind, bounds_error=False, fill_value=np.nan
        )

        return self

    def plot(self, figsize=None):
        _, axes = plt.subplots(2, 1, figsize=figsize)
        axes[0].plot(self.time, self.angle_profile(self.time), label="Interpolation")
        axes[0].scatter(self.event_times, self.event_angles, c="r", s=0.5, label="Original")
        axes[0].set_title("Time-angle Profile")
        axes[0].set_xlabel("Time[s]")
        axes[0].set_ylabel("Angle[rad]")
        axes[0].legend()
        axes[1].plot(self.event_times, self.rpm_profile(self.event_times))
        axes[1].set_title("Time-RPM Profile")
        axes[1].set_xlabel("Time[s]")
        axes[1].set_ylabel("RPM[rev/min]")
        plt.show()


def resample_signal(signal, x1, x1_to_x2_profile, kind="linear", is_plot=False, figsize=None):
    x2 = x1_to_x2_profile(x1)
    x2_nan_indices = np.isnan(x2)
    x2 = x2[~x2_nan_indices]
    cut_signal = signal[~x2_nan_indices]
    x2_to_sig_profile = interp1d(x2, cut_signal, kind=kind)
    x2 = np.linspace(x2[0], x2[-1], cut_signal.size)
    resampled_signal = x2_to_sig_profile(x2)
    sr = (cut_signal.size - 1) / (x2[-1] - x2[0])

    if is_plot is True:
        _, axes = plt.subplots(2, 1, figsize=figsize)
        axes[0].plot(x1, signal)
        axes[0].set_title("Original signal")
        axes[0].set_xlabel("x1")
        axes[0].set_ylabel("Amplitude")
        axes[1].set_title("Resampled signal")
        axes[1].plot(x2, resampled_signal)
        axes[1].set_xlabel("x2")
        axes[1].set_ylabel("Amplitude")
        plt.show()

    return x2, resampled_signal, sr


def do_fft(signal, sr):
    amp = np.abs(fft(signal)) / signal.size * 2
    freq = fftfreq(signal.size, 1 / sr)
    idxes = np.where(freq >= 0)
    amp = amp[idxes]
    freq = freq[idxes]
    return freq, amp


def convert_envelope_signal(signal: np.ndarray):
    envelope_signal = np.abs(hilbert(signal))
    return envelope_signal

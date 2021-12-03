import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import matplotlib.pyplot as plt
import signal_basic_analysis as sba
from convert_to_x import TachoToRPM, resample_signal
from freq_filter import bandpass_filter, lowpass_filter
from local_maxima_tracking import track_local_maxima
from scipy.interpolate import interp1d
from scipy.signal import hilbert

# Parameter
sr = 25600
root = Path(r"C:\Users\kangwhi\Desktop\data\Planetary_variable")
conditions = ["normal", "fault_5"]
avg_ang_vel = 21.7
u_gmf = 23.373
avg_gmf = u_gmf * avg_ang_vel

# vibration signal load
condition = conditions[0]
df = pd.read_pickle(root / f"sine_rpm_{condition}.pickle")
signal = df["Vib"].to_numpy(dtype=np.float64)
time = np.linspace(0, signal.size / sr, signal.size)

# tacho signal load
tacho_data = df["Enc"].to_numpy(dtype=np.float32)
tacho_time = np.linspace(0, tacho_data.size / sr, tacho_data.size)

# get the angle-profile and rpm-profile from tacho signal
ttr = TachoToRPM(tacho_time, tacho_data, teeth=60)
ttr.make_angle_and_rpm_profile(2, 5)
ttr.plot()

# examine stft
sba.plot_stft(signal, sr)
sba.show()

# extract frequency profiles
cmap = plt.get_cmap("Reds")
track_param = [25, 50, 100, 150, 200, 250, 300]
color_param = np.arange(30, 240, 30)
cmap = plt.get_cmap("Reds")
for bwd, color in zip(track_param, color_param):
    time_component, freq_component = track_local_maxima(signal, sr, 238, bwd, 1)
    plt.plot(time_component, freq_component, color=cmap(color), label=f"f_bandwidth = {bwd}")
plt.title("instantaneous phase profile")
plt.xlabel("Time[s]")
plt.ylabel("RPM[rev/min]")
plt.legend()
plt.show()

# select 'm'th harmonic frequency component and extract phase component
m = 2
time_component, freq_component = track_local_maxima(signal, sr, 238, 50, 20)
# sba.plot_signal(time_component, freq_component)
# sba.show()
phase_component = np.cumsum(2 * np.pi * freq_component[1:] * np.diff(time_component))
# plt.plot(time_component[1:], phase_component)
# plt.title('time-angle profile')
# plt.xlabel('Time[s]')
# plt.ylabel('Angle[rad]')
# plt.show()


# angular resampling
angle_profile = interp1d(
    time_component[1:], phase_component, kind="linear", bounds_error=False, fill_value=np.nan
)
angle, resampled_signal, spr = resample_signal(signal, time, angle_profile)

# order analysis
# sba.plot_fft(resampled_signal, spr, True)
# sba.show()

# apply bandpass filter to resampled signal
signal_filtered = bandpass_filter(resampled_signal, spr, 0.99 / (2 * np.pi), 1.01 / (2 * np.pi), 4)
sba.plot_fft(resampled_signal, spr, True)
sba.show()

# reverse angular resampling
time_profile = interp1d(
    phase_component, time_component[1:], kind="linear", bounds_error=False, fill_value=np.nan
)
time_filtered, signal_filtered, sr_filtered = resample_signal(signal_filtered, angle, time_profile)
# sba.plot_fft(signal_filtered, sr_filtered)
# sba.show()

# extract IF(instantaneous frequency)
z = hilbert(signal_filtered)
phase = np.unwrap(np.angle(z))
phase /= m * u_gmf
inst_phase = np.diff(phase) / np.diff(time_filtered) / (2 * np.pi) * 60
# sba.plot_fft(inst_phase, sr_filtered)
# sba.show()

filtered_inst_phase = lowpass_filter(inst_phase, sr_filtered, 0.5, 2)
# plt.plot(time_filtered[:-1], filtered_inst_phase)
# plt.title('instantaneous phase profile')
# plt.xlabel('Time[s]')
# plt.ylabel('RPM[rev/min]')
# plt.show()

# calculate relative error
error = np.abs(
    (ttr.rpm_profile(time_filtered[1:]) - filtered_inst_phase) / ttr.rpm_profile(time_filtered[1:])
)
error = np.sum(error[~np.isnan(error)]) / error.size
print("relative error: ", error)

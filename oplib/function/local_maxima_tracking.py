import numpy as np
from scipy.signal import stft


def track_local_maxima(signal, sr, f_init, f_bandwidth, df, **stft_kwargs):
    nperseg = int(sr / df + 1)
    f, t, Zxx = stft(signal, sr, nperseg=nperseg, **stft_kwargs)
    Zxx = np.abs(Zxx)
    inst_freq = np.zeros(t.size)
    for i in range(t.size):
        if i == 0:
            f_max = f_init
            inst_freq[i] = f_max
            continue
        interested_f_range_indices = np.where(np.abs(f - f_max) < f_bandwidth / 2)
        f_range = f[interested_f_range_indices]
        f_maxima_indices = np.argmax(Zxx[interested_f_range_indices, i])
        f_max = f_range[f_maxima_indices]
        inst_freq[i] = f_max

    return t, inst_freq

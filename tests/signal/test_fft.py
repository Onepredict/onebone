from typing import Callable

import numpy as np

from oplib.signal import positiv_fft

# import pytest


def _generate_signal(fs: float):
    # Generate signal
    T = 1 / fs
    t = np.arange(0, 0.5, T)
    noise = np.random.normal(0, 0.05, len(t))
    x = 0.6 * np.cos(2 * np.pi * 60 * t + np.pi / 2) + np.cos(2 * np.pi * 120 * t)
    signal = x + noise
    return signal


def check_signal(fft_: Callable, input_arg: float, hann: bool, expected_return: np.ndarray):
    fs = input_arg
    signal = _generate_signal(fs)
    f_, mag_ = fft_(signal, hann, fs)
    Freq_ = f_[np.where(mag_ > 10)]
    assert np.all(np.equal(Freq_, expected_return)), (
        f"Wrong return: The expected return is {expected_return}, " + f"but output is {Freq_}"
    )


def check_array_shape(fft_: Callable, input_arg: float, hann: bool, expected_return: np.ndarray):
    fs = input_arg
    signal = _generate_signal(fs)
    changed_signal1 = signal.reshape(1, -1)
    changed_signal2 = signal.reshape(-1, 1)
    f_1, mag_1 = positiv_fft(changed_signal1, hann, fs)
    f_2, mag_2 = positiv_fft(changed_signal2, hann, fs)
    Freq_ = f_1[np.where(mag_1 > 10)]
    assert np.all(np.equal(Freq_, expected_return)), (
        f"Wrong return: The expected return is {expected_return}, " + f"but output is {Freq_}"
    )
    # with pytest.raises(ValueError) as ex:
    # positiv_fft(changed_signal1, hann, fs)
    # assert str(ex.value) == "Dimension of signal must be less than 2."


def test_fft():
    check_signal(positiv_fft, 2000, False, np.array([60.0, 120.0]))
    check_array_shape(positiv_fft, 2000, False, np.array([60.0, 120.0]))


if __name__ == "__main__":
    test_fft()

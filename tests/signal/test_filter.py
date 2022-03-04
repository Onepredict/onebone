"""Test frequency filter.

- Author: Kyunghwan Kim
- Contact: kyunghwan.kim@onepredict.com
"""

from typing import Callable, Tuple

import numpy as np

from onebone.signal import (
    bandpass_filter,
    bandstop_filter,
    highpass_filter,
    lowpass_filter,
)


def _generate_sin_signal(fs: float):
    # Generate signal
    t = np.linspace(0, 1, int(fs))
    signal = 2.0 * np.sin(2 * np.pi * 10.0 * t)
    signal += 10.0 * np.sin(2 * np.pi * 20.0 * t)
    signal += 5.0 * np.sin(2 * np.pi * 100.0 * t)
    signal += 1.0 * np.sin(2 * np.pi * 200.0 * t)
    signal += 1.0 * np.sin(2 * np.pi * 500.0 * t)
    return signal


def check_1d_signal(filter_: Callable, input_args: Tuple[float, int], expected_return: np.ndarray):
    fs = input_args[0]
    signal = _generate_sin_signal(fs)
    filtered_signal = filter_(signal, *input_args)

    # Check frequency
    freq_x = np.fft.rfftfreq(signal.size, 1 / fs)[:-1]
    filtered_fft_mag = abs((np.fft.rfft(filtered_signal) / signal.size)[:-1] * 2)
    filtered_freq = freq_x[np.where(filtered_fft_mag > 0.5)]

    # print(origin_freq, filtered_freq)
    assert np.all(np.equal(filtered_freq, expected_return)), (
        f"Wrong return: The expected return is {expected_return}, "
        + f"but output is {filtered_freq}"
    )


def check_2d_signal(filter_: Callable, input_args: Tuple[float, int], expected_return: np.ndarray):
    fs = input_args[0]
    signal = _generate_sin_signal(fs)

    signal_2d = np.stack([signal] * 2).T

    filtered_signal = filter_(signal_2d, *input_args, axis=0)

    # Check frequency
    freq_x = np.fft.rfftfreq(signal.size, 1 / fs)[:-1]
    for idx in range(2):
        filtered_fft_mag = abs((np.fft.rfft(filtered_signal[:, idx]) / signal.size)[:-1] * 2)
        filtered_freq = freq_x[np.where(filtered_fft_mag > 0.5)]

        assert np.all(np.equal(filtered_freq, expected_return)), (
            f"Wrong return: The expected return is {expected_return}, "
            + f"but output is {filtered_freq}"
        )


def check_3d_signal(filter_: Callable, input_args: Tuple[float, int], expected_return: np.ndarray):
    fs = input_args[0]
    origin_signal = _generate_sin_signal(fs)

    # signal shape [2, 3, data_length]
    changed_signal = np.stack([origin_signal] * 3, axis=0)
    changed_signal = np.stack([changed_signal] * 2, axis=0)
    changed_signal = np.transpose(changed_signal, (2, 0, 1))

    filtered_signal = filter_(changed_signal, *input_args, axis=0)

    # Check frequency
    freq_x = np.fft.rfftfreq(origin_signal.size, 1 / fs)[:-1]
    for i in range(2):
        for j in range(3):
            filtered_fft_mag = abs(
                (np.fft.rfft(filtered_signal[:, i, j]) / origin_signal.size)[:-1] * 2
            )
            filtered_freq = freq_x[np.where(filtered_fft_mag > 0.5)]

            assert np.all(np.equal(filtered_freq, expected_return)), (
                f"Wrong return: The expected return is {expected_return}, "
                + f"but output is {filtered_freq}"
            )


def test_lowpass_filter():
    check_1d_signal(lowpass_filter, (5000.0, 50), np.array([10.0, 20.0]))
    check_2d_signal(lowpass_filter, (5000.0, 50), np.array([10.0, 20.0]))
    check_3d_signal(lowpass_filter, (5000.0, 50), np.array([10.0, 20.0]))


def test_highpass_filter():
    check_1d_signal(highpass_filter, (5000.0, 50), np.array([100.0, 200.0, 500.0]))
    check_2d_signal(highpass_filter, (5000.0, 50), np.array([100.0, 200.0, 500.0]))
    check_3d_signal(highpass_filter, (5000.0, 50), np.array([100.0, 200.0, 500.0]))


def test_bandpass_filter():
    check_1d_signal(bandpass_filter, (5000.0, 50, 300), np.array([100.0, 200.0]))
    check_2d_signal(bandpass_filter, (5000.0, 50, 300), np.array([100.0, 200.0]))
    check_3d_signal(bandpass_filter, (5000.0, 50, 300), np.array([100.0, 200.0]))


def test_bandstop_filter():
    check_1d_signal(bandstop_filter, (5000.0, 50, 300), np.array([10.0, 20.0, 500.0]))
    check_2d_signal(bandstop_filter, (5000.0, 50, 300), np.array([10.0, 20.0, 500.0]))
    check_3d_signal(bandstop_filter, (5000.0, 50, 300), np.array([10.0, 20.0, 500.0]))


if __name__ == "__main__":
    test_lowpass_filter()
    test_highpass_filter()
    test_bandpass_filter()
    test_bandstop_filter()

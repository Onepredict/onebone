"""Test frequency filter.

- Author: Kyunghwan Kim
- Contact: kyunghwan.kim@onepredict.io
"""

from typing import Callable, Tuple

import numpy as np

from oplib.signal import lowpass_filter


def _generate_sin_signal(fs: float):
    # Generate signal
    t = np.linspace(0, 1, int(fs))
    signal = 2.0 * np.sin(2 * np.pi * 10.0 * t)
    signal += 10.0 * np.sin(2 * np.pi * 20.0 * t)
    signal += 5.0 * np.sin(2 * np.pi * 100.0 * t)
    signal += 1.0 * np.sin(2 * np.pi * 200.0 * t)
    signal += 1.0 * np.sin(2 * np.pi * 500.0 * t)
    return signal


def check_normal_signal(
    filter_: Callable, input_args: Tuple[float, int], expected_return: np.ndarray
):
    fs, cutoff = input_args
    signal = _generate_sin_signal(fs)
    filtered_signal = filter_(signal, fs, cutoff)

    # Check frequency
    freq_x = np.fft.rfftfreq(signal.size, 1 / fs)[:-1]
    # origin_fft_mag = abs((np.fft.rfft(signal) / signal.size)[:-1] * 2)
    # origin_freq = freq_x[np.where(origin_fft_mag > 0.5)]

    filtered_fft_mag = abs((np.fft.rfft(filtered_signal) / signal.size)[:-1] * 2)
    filtered_freq = freq_x[np.where(filtered_fft_mag > 0.5)]

    # print(origin_freq, filtered_freq)
    assert np.all(
        np.equal(filtered_freq, expected_return)
    ), f"Wrong return: The expected return is {expected_return}, but output is {filtered_freq}"


def check_error_case(filter_: Callable):
    pass


def test_lowpass_filter():
    check_normal_signal(
        lowpass_filter,
        (5000.0, 50),
        np.array(
            [
                10.0,
                20.0,
            ]
        ),
    )
    check_error_case(lowpass_filter)


if __name__ == "__main__":
    test_lowpass_filter()

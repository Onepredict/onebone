from typing import Callable, Tuple

import numpy as np
import pytest

from oplib.signal import positive_fft


def generate_signal(fs: float):
    N = 600
    # sample spacing
    T = 1.0 / fs
    x = np.linspace(0.0, N * T, N, endpoint=False)
    signal = 3 * np.sin(50.0 * 2.0 * np.pi * x) + 2 * np.sin(80.0 * 2.0 * np.pi * x)
    return signal


def check_signal(fft: Callable, input_args: Tuple[float, bool, bool], expected_return: np.ndarray):
    fs = input_args[0]
    signal = generate_signal(fs)
    f, mag = fft(signal, *input_args)
    freq = np.around(f[np.where(mag[0] > 1)])
    assert np.all(np.equal(freq, expected_return)), (
        f"Wrong return: The expected return is {expected_return}, " + f"but output is {freq}"
    )


def check_array_shape(fft: Callable, input_args: Tuple[float, bool, bool]):
    fs = input_args[0]
    signal = generate_signal(fs)
    changed_signal = np.stack([signal] * 3, axis=0)
    changed_signal = np.stack([changed_signal] * 2, axis=0)
    # case 1: signal shape [2, 3, data_length] -> ArrayShapeError
    with pytest.raises(ValueError) as ex:
        fft(changed_signal, *input_args)
    assert str(ex.value) == "Dimension of signal must be less than 3"


def check_hann(fft: Callable, input_args: Tuple[float, bool, bool], expected_return: np.ndarray):
    fs = input_args[0]
    signal = generate_signal(fs)
    f, mag = fft(signal, *input_args)
    freq = f[np.where(mag[0] > 2)]
    assert np.all(np.equal(freq, expected_return)), (
        f"Wrong return: The expected return is {expected_return}, " + f"but output is {freq}"
    )


def test_fft():
    check_signal(positive_fft, (800.0, False, True), np.array([49.0, 51.0, 80.0]))
    check_array_shape(positive_fft, (800.0, False, True))
    check_hann(positive_fft, (800.0, True, True), np.array([]))


if __name__ == "__main__":
    test_fft()

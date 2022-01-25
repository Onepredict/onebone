from typing import Callable, Tuple

import numpy as np
import pytest

from onebone.signal import positive_fft


def generate_signal(fs: float):
    n = 400
    # sample spacing
    t = 1.0 / fs
    x = np.linspace(0.0, n * t, n, endpoint=False)
    signal = 3 * np.sin(50.0 * 2.0 * np.pi * x) + 2 * np.sin(80.0 * 2.0 * np.pi * x)
    return signal


def check_1d_signal(
    fft: Callable, input_args: Tuple[float, bool, bool], expected_return: np.ndarray
):
    fs = input_args[0]
    signal = generate_signal(fs)
    f, mag = fft(signal, *input_args)
    freq = np.around(f[np.where(mag > 2)])
    assert np.all(np.equal(freq, expected_return)), (
        f"Wrong return: The expected return is {expected_return}, " + f"but output is {freq}"
    )


def check_2d_signal_axis_zero(
    fft: Callable, input_args: Tuple[float, bool, bool, int], expected_return: np.ndarray
):
    fs = input_args[0]
    signal = generate_signal(fs)
    signal_2d = np.stack([signal] * 2)
    signal_2d = signal_2d.T
    f, mag = fft(signal_2d, *input_args)
    # mag.shape = (n,1)
    freq = np.around(f[np.where(mag[:, 0] > 2)])
    assert np.all(np.equal(freq, expected_return)), (
        f"Wrong return: The expected return is {expected_return}, " + f"but output is {freq}"
    )


def check_2d_signal_axis_one(
    fft: Callable, input_args: Tuple[float, bool, bool, int], expected_return: np.ndarray
):
    fs = input_args[0]
    signal = generate_signal(fs)
    signal_2d = np.stack([signal] * 2)
    f, mag = fft(signal_2d, *input_args)
    # mag.shape = (1,n)
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


def test_fft():
    check_1d_signal(positive_fft, (800.0, False, False), np.array([50, 80.0]))
    check_2d_signal_axis_zero(positive_fft, (800.0, False, False, 0), np.array([50, 80.0]))
    check_2d_signal_axis_one(positive_fft, (800.0, False, True, 1), np.array([50, 80.0]))
    check_array_shape(positive_fft, (800.0, False, True))


if __name__ == "__main__":
    test_fft()

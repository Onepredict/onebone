from typing import Callable, Tuple

import numpy as np
import pytest

from oplib.signal import positive_fft


def _generate_signal(fs: float):
    N = 600
    # sample spacing
    T = 1.0 / fs
    x = np.linspace(0.0, N * T, N, endpoint=False)
    signal = 3 * np.sin(50.0 * 2.0 * np.pi * x) + 2 * np.sin(80.0 * 2.0 * np.pi * x)
    return signal


def check_signal(fft_: Callable, input_args: Tuple[float, bool, bool], expected_return: np.ndarray):
    fs = input_args[0]
    signal = _generate_signal(fs)
    f_, mag_ = fft_(signal, *input_args)
    freq_ = np.around(f_[np.where(mag_ > 1)])
    assert np.all(np.equal(freq_, expected_return)), (
        f"Wrong return: The expected return is {expected_return}, " + f"but output is {freq_}"
    )


def check_array_shape(fft_: Callable, input_args: Tuple[float, bool, bool]):
    fs = input_args[0]
    signal = _generate_signal(fs)
    changed_signal = np.stack([signal] * 3, axis=0)
    changed_signal = np.stack([changed_signal] * 2, axis=0)
    # case 1: signal shape [2, 3, data_length] -> ArrayShapeError
    with pytest.raises(ValueError) as ex:
        fft_(changed_signal, *input_args)
    assert str(ex.value) == "Dimension of signal must be less than 3"


def check_hann_(fft_: Callable, input_args: Tuple[float, bool, bool], expected_return: np.ndarray):
    fs = input_args[0]
    signal = _generate_signal(fs)
    f_, mag_ = fft_(signal, *input_args)
    freq_ = f_[np.where(mag_ > 2)]
    assert np.all(np.equal(freq_, expected_return)), (
        f"Wrong return: The expected return is {expected_return}, " + f"but output is {freq_}"
    )


def test_fft():
    check_signal(positive_fft, (800.0, False, True), np.array([49.0, 51.0, 80.0]))
    check_array_shape(positive_fft, (800.0, False, True))
    check_hann_(positive_fft, (800.0, True, True), np.array([]))


if __name__ == "__main__":
    test_fft()

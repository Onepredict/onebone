from typing import Callable

import numpy as np
import pytest

from onebone.signal import snr_feature


def generate_signal(fs: float):
    t = np.linspace(0, 1, int(fs))
    x = 10.0 * np.sin(2 * np.pi * 20.0 * t)
    return x


def check_bad_args():
    x = np.array([1.0 + 0.0j])
    fs = 1000.0
    with pytest.raises(ValueError) as ex:
        snr_feature(x, fs)
    assert ex.value.args[0] == "`x` must be real."


def check_snr(snr_feature: Callable, fs: float, expected_return: np.float64):
    x = generate_signal(fs)
    snr = snr_feature(x, fs)
    output = np.round(np.mean(snr), 2)
    assert np.all(np.equal(output, expected_return)), (
        f"Wrong return: The expected return is {expected_return}, " + f"but output is {output}"
    )


def test_snr():
    check_bad_args()
    check_snr(snr_feature, 1000.0, 3.49)


if __name__ == "__main__":
    test_snr()

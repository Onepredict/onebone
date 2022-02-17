import numpy as np
import pytest

from onebone.feature import snr


def generate_signal():
    t = np.linspace(0, 1, 1000)
    x = 10.0 * np.sin(2 * np.pi * 20.0 * t)
    return x


def check_bad_args():
    x = np.array([1.0 + 0.0j])
    fs = 1000.0
    with pytest.raises(ValueError) as ex:
        snr(x, fs)
    assert ex.value.args[0] == "`x` must be real."


def check_snr():
    x = generate_signal()
    _snr = snr(x, 1000)
    output = np.round(np.mean(_snr), 2)
    assert np.all(np.equal(output, 7.16)), (
        f"Wrong return: The expected return is {7.16}, " + f"but output is {output}"
    )


def test_snr():
    check_bad_args()
    check_snr()


if __name__ == "__main__":
    test_snr()

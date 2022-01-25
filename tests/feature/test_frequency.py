"""Test code for frequency_feature.py.

- Author: Kangwhi Kim
- Contact: kangwhi.kim@onepredict.com
"""

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from onebone.feature import mdf, mnf, vcf


def _generate_sin_signal(amp, f, fs=None):
    if fs is None:
        fs = f * 2.56  # Nyquist frequency
    t = np.arange(0, 1, 1 / fs)
    x = amp * np.sin(2 * np.pi * f * t)
    return x


def check_mnf_1d_array():
    # Get the 1-D signal `x`.
    fs = 100
    x1 = _generate_sin_signal(1, 10, fs)
    x2 = _generate_sin_signal(2, 20, fs)
    x3 = _generate_sin_signal(1, 30, fs)
    x_1d = x1 + x2 + x3

    # Get the 1-D amplitudes of the spectrum
    amp_1d = np.abs(np.fft.fft(x_1d))

    # Check the output when `fs` is None.
    output = mnf(amp_1d)
    assert_almost_equal(output, 0.2)

    # Check the output when `fs` is not None.
    output = mnf(amp_1d, fs)
    assert_almost_equal(output, 20)

    # Check the output when `freq_range` is not None.
    output = mnf(amp_1d, fs, freq_range=(10, 20))
    assert_almost_equal(output, 50 / 3)


def check_mnf_2d_arrays():
    # Get the 2-D signal `x`.
    fs = 100
    x1 = _generate_sin_signal(1, 10, fs)
    x2 = _generate_sin_signal(2, 20, fs)
    x3 = _generate_sin_signal(1, 30, fs)
    x_1d = x1 + x2 + x3
    x_2d = np.stack([x_1d, x_1d], axis=0)

    # Get the 2-D amplitudes of the spectrum
    amp_2d = np.abs(np.fft.fft(x_2d))

    # Check the output when `axis` is 0.
    output = mnf(amp_2d.T, fs=fs, axis=0)
    assert_array_almost_equal(output, np.ones(2) * 20)
    output = mnf(amp_2d.T, fs=fs, freq_range=(10, 20), axis=0)
    assert_array_almost_equal(output, np.ones(2) * 50 / 3)
    output = mnf(amp_2d.T, fs=fs, freq_range=(10, 20), axis=0, keepdims=True)
    assert_array_almost_equal(output, np.ones((1, 2)) * 50 / 3)

    # Check the output when `axis` is 1.
    output = mnf(amp_2d, fs=fs, axis=1)
    assert_array_almost_equal(output, np.ones(2) * 20)
    output = mnf(amp_2d, fs=fs, freq_range=(10, 20), axis=1)
    assert_array_almost_equal(output, np.ones(2) * 50 / 3)
    output = mnf(amp_2d, fs=fs, freq_range=(10, 20), axis=1, keepdims=True)
    assert_array_almost_equal(output, np.ones((2, 1)) * 50 / 3)


def check_mdf_1d_array():
    # Get the 1-D signal `x`.
    fs = 100
    x1 = _generate_sin_signal(1, 10, fs)
    x2 = _generate_sin_signal(2, 20, fs)
    x3 = _generate_sin_signal(1, 30, fs)
    x_1d = x1 + x2 + x3

    # Get the 1-D amplitudes of the spectrum
    amp_1d = np.abs(np.fft.fft(x_1d))

    # Check the output when `fs` is None.
    output = mdf(amp_1d)
    assert_almost_equal(output, 0.2, decimal=3)

    # Check the output when `fs` is not None.
    output = mdf(amp_1d, fs)
    assert_almost_equal(output, 20, decimal=3)

    # Check the output when `freq_range` is not None.
    output = mdf(amp_1d, fs, freq_range=(10, 15))
    assert_almost_equal(output, 10)


def check_mdf_2d_arrays():
    # Get the 2-D signal `x`.
    fs = 100
    x1 = _generate_sin_signal(1, 10, fs)
    x2 = _generate_sin_signal(2, 20, fs)
    x3 = _generate_sin_signal(1, 30, fs)
    x_1d = x1 + x2 + x3
    x_2d = np.stack([x_1d, x_1d], axis=0)

    # Get the 2-D amplitudes of the spectrum
    amp_2d = np.abs(np.fft.fft(x_2d))

    # Check the output when `axis` is 0.
    output = mdf(amp_2d.T, fs=fs, axis=0)
    assert_array_almost_equal(output, np.ones(2) * 20)
    output = mdf(amp_2d.T, fs=fs, freq_range=(10, 15), axis=0)
    assert_array_almost_equal(output, np.ones(2) * 10)
    output = mdf(amp_2d.T, fs=fs, freq_range=(10, 15), axis=0, keepdims=True)
    assert_array_almost_equal(output, np.ones((1, 2)) * 10)

    # Check the output when `axis` is 1.
    output = mdf(amp_2d, fs=fs, axis=1)
    assert_array_almost_equal(output, np.ones(2) * 20)
    output = mdf(amp_2d, fs=fs, freq_range=(10, 15), axis=1)
    assert_array_almost_equal(output, np.ones(2) * 10)
    output = mdf(amp_2d, fs=fs, freq_range=(10, 15), axis=1, keepdims=True)
    assert_array_almost_equal(output, np.ones((2, 1)) * 10)


def check_vcf_1d_array():
    # Get the 1-D signal `x`.
    fs = 100
    x1 = _generate_sin_signal(1, 10, fs)
    x2 = _generate_sin_signal(2, 20, fs)
    x3 = _generate_sin_signal(1, 30, fs)
    x_1d = x1 + x2 + x3

    # Get the 1-D amplitudes of the spectrum
    amp_1d = np.abs(np.fft.fft(x_1d))

    # Check the output when `fs` is None.
    output = vcf(amp_1d)
    assert_almost_equal(output, 0.005)

    # Check the output when `fs` is not None.
    output = vcf(amp_1d, fs)
    assert_almost_equal(output, 50)

    # Check the output when `freq_range` is not None.
    output = vcf(amp_1d, fs, freq_range=(10, 15))
    assert_almost_equal(output, 0)


def check_vcf_2d_arrays():
    # Get the 2-D signal `x`.
    fs = 100
    x1 = _generate_sin_signal(1, 10, fs)
    x2 = _generate_sin_signal(2, 20, fs)
    x3 = _generate_sin_signal(1, 30, fs)
    x_1d = x1 + x2 + x3
    x_2d = np.stack([x_1d, x_1d], axis=0)

    # Get the 2-D amplitudes of the spectrum
    amp_2d = np.abs(np.fft.fft(x_2d))

    # Check the output when `axis` is 0.
    output = vcf(amp_2d.T, fs=fs, axis=0)
    assert_array_almost_equal(output, np.ones(2) * 50)
    output = vcf(amp_2d.T, fs=fs, freq_range=(10, 15), axis=0)
    assert_array_almost_equal(output, np.zeros(2))
    output = vcf(amp_2d.T, fs=fs, freq_range=(10, 15), axis=0, keepdims=True)
    assert_array_almost_equal(output, np.zeros((1, 2)))

    # Check the output when `axis` is 1.
    output = vcf(amp_2d, fs=fs, axis=1)
    assert_array_almost_equal(output, np.ones(2) * 50)
    output = vcf(amp_2d, fs=fs, freq_range=(10, 15), axis=1)
    assert_array_almost_equal(output, np.zeros(2))
    output = vcf(amp_2d, fs=fs, freq_range=(10, 15), axis=1, keepdims=True)
    assert_array_almost_equal(output, np.zeros((2, 1)))


def test_mnf():
    check_mnf_1d_array()
    check_mnf_2d_arrays()


def test_mdf():
    check_mdf_1d_array()
    check_mdf_2d_arrays()


def test_vcf():
    check_vcf_1d_array()
    check_vcf_2d_arrays()


if __name__ == "__main__":
    test_mnf()
    test_mdf()
    test_vcf()

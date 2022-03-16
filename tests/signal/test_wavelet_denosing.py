"""Test Wavelet Denosing."""
import numpy as np
from numpy.testing import assert_array_almost_equal

from onebone.signal import wavelet_denoising


def test_1d_signal(is_plot: bool = False):
    np.random.seed(123)

    signal = np.array([10.0] * 10 + [0.0] * 10)
    signal += np.random.random(signal.shape)
    wavelet = "db1"
    denoised_signal = wavelet_denoising(signal, wavelet, level=2)
    expected_answer = np.array(
        [
            10.440194, 10.440194, 10.440194, 10.440194, 10.702042, 10.702042,
            10.702042, 10.702042, 10.038253, 10.038253, 0.934385, 0.934385,
            0.408572, 0.408572, 0.408572, 0.408572, 0.355331, 0.355331,
            0.355331, 0.355331
        ]  # fmt: skip
    )
    assert_array_almost_equal(denoised_signal, expected_answer)

    if is_plot:
        import matplotlib.pyplot as plt

        plt.plot(signal)
        plt.show()
        plt.close()

        plt.plot(denoised_signal)
        plt.show()
        plt.close()


def test_2d_signal():
    np.random.seed(123)

    signal = np.array([10.0] * 10 + [0.0] * 10)
    signal += np.random.random(signal.shape)
    signal_2d = np.stack([signal] * 2).T

    wavelet = "db1"
    denoised_signal = wavelet_denoising(signal_2d, wavelet, axis=0, level=2)
    for idx in range(2):
        expected_answer = np.array(
            [
                10.440194, 10.440194, 10.440194, 10.440194, 10.702042, 10.702042,
                10.702042, 10.702042, 9.994573, 9.994573, 0.978066, 0.978066,
                0.408572, 0.408572, 0.408572, 0.408572, 0.355331, 0.355331,
                0.355331, 0.355331
            ]  # fmt: skip
        )
        assert_array_almost_equal(denoised_signal[:, idx], expected_answer)


if __name__ == "__main__":
    test_1d_signal(is_plot=True)
    test_2d_signal()

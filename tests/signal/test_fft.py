import numpy as np

from onebone.signal import positive_fft


def generate_signal():
    n = 400
    # sample spacing
    t = 1 / 800
    x = np.linspace(0.0, n * t, n, endpoint=False)
    signal = 3 * np.sin(50.0 * 2.0 * np.pi * x) + 2 * np.sin(80.0 * 2.0 * np.pi * x)
    return signal


def check_1d_signal():
    fs = 800
    signal = generate_signal()
    freq, mag = positive_fft(signal, fs=fs, hann=False, normalization=False)
    target_freq = np.around(freq[np.where(mag > 2)])

    expected_return = np.array([50, 80.0])
    assert np.all(np.equal(target_freq, expected_return)), (
        f"Wrong return: The expected return is {expected_return}, " + f"but output is {freq}"
    )


def check_2d_signal_axis_zero():
    fs = 800
    signal = generate_signal()
    signal_2d = np.stack([signal] * 2)
    signal_2d = signal_2d.T

    freq, mag = positive_fft(signal_2d, fs=fs, hann=False, normalization=False, axis=0)
    target_freq = np.around(freq[np.where(mag[:, 0] > 2)])

    expected_return = np.array([50, 80.0])
    assert np.all(np.equal(target_freq, expected_return)), (
        f"Wrong return: The expected return is {expected_return}, " + f"but output is {freq}"
    )


def check_2d_signal_axis_one():
    fs = 800
    signal = generate_signal()
    signal_2d = np.stack([signal] * 2)

    freq, mag = positive_fft(signal_2d, fs=fs, hann=False, normalization=True, axis=1)
    target_freq = np.around(freq[np.where(mag[0, :] > 1)])

    expected_return = np.array([50, 80.0])
    assert np.all(np.equal(target_freq, expected_return)), (
        f"Wrong return: The expected return is {expected_return}, " + f"but output is {freq}"
    )


def check_3d_signal():
    fs = 800
    signal = generate_signal()
    signal_2d = np.stack([signal] * 2)
    signal_3d = np.stack([signal_2d] * 2)

    freq, mag = positive_fft(signal_3d, fs=fs, hann=False, normalization=True, axis=2)
    expected_return = np.array([50, 80.0])

    for i in range(2):
        for j in range(2):
            target_freq = np.around(freq[np.where(mag[i, j] > 1)])
            assert np.all(np.equal(target_freq, expected_return)), (
                f"Wrong return: The expected return is {expected_return}, "
                + f"but output is {freq}"
            )


def test_fft():
    check_1d_signal()
    check_2d_signal_axis_zero()
    check_2d_signal_axis_one()
    check_3d_signal()


if __name__ == "__main__":
    test_fft()

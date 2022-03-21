import numpy as np

from onebone.feature import phase_alignment


def generate_signal() -> np.ndarray:
    n = 3000
    t = 1 / 1000

    x = np.linspace(0.0, n * t, n, endpoint=False)
    signal = 3 * np.sin(50.0 * 2.0 * np.pi * x) + 2 * np.sin(80.0 * 2.0 * np.pi * x)
    ys = np.array([signal + np.random.uniform(low=-1, high=1, size=(n,)) for i in range(1000)])

    return ys


def test_phase_alignment():
    generate_signal()
    ys = generate_signal()
    fs = 1000
    freq, pa = phase_alignment(ys, fs)
    pa_result = np.around(freq[np.where(pa > 0.9)])
    expected_return = np.array([50.0, 80.0])
    assert np.all(np.equal(pa_result, expected_return)), (
        f"Wrong return: The expected return is {expected_return}, " + f"but output is {pa_result}"
    )


if __name__ == "__main__":
    test_phase_alignment()

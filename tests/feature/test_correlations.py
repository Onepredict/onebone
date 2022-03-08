import numpy as np

from onebone.feature import correlations


def generate_signal():
    n = 3000
    t = 1 / 1000

    x = np.linspace(0.0, n * t, n, endpoint=False)
    signal = 3 * np.sin(50. * 2. * np.pi * x) + \
             2 * np.sin(80. * 2. * np.pi * x)
    ys = np.array([signal + np.random.uniform(low=-1, high=1, size=(n,)) for i in range(1000)])

    return ys


def check_phase_alignment():
    generate_signal()
    ys = generate_signal()
    freq, pa = correlations.phase_alignment(ys, 1 / 1000)
    pa_result = np.around(freq[np.where(pa > 0.9)])
    expected_return = np.array([50., 80.])
    assert np.all(np.equal(pa_result, expected_return)), (
            f"Wrong return: The expected return is {expected_return}, " + f"but output is {pa_result}"
    )

"""Test for envelope.py.

- Author: Kangwhi Kim
- Contact: kangwhi.kim@onepredict.com
"""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from oplib.signal.envelope import envelope_analytic


def test_bad_args():
    x = np.array([1.0 + 0.0j])
    with pytest.raises(ValueError) as ex:
        envelope_analytic(x)
    assert ex.value.args[0] == "`x` must be real."


def test_envelope_theoretical():
    decimal = 14

    pi = np.pi
    t = np.arange(0, 2 * pi, pi / 256)
    a0 = np.sin(t)
    a1 = np.cos(t)
    a2 = np.sin(2 * t)
    a3 = np.cos(2 * t)
    a = np.vstack([a0, a1, a2, a3])

    e = envelope_analytic(a)

    # The absolute value should be one everywhere, for this input:
    assert_almost_equal(e, np.ones(a.shape), decimal)


if __name__ == "__main__":
    test_bad_args()
    test_envelope_theoretical()

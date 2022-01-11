"""Test time analysis.

- Author: Kyunghwan Kim
- Contact: kyunghwan.kim@onepredict.com
"""

import numpy as np
from numpy.testing import assert_almost_equal

from oplib.feature import crestfactor, rms


def test_rms():
    x = np.array([[4, 9, 2, 10, 0], [6, 9, 7, 12, 0]])
    assert_almost_equal(rms(x), 7.1484264)
    assert_almost_equal(rms(x, axis=0), np.array([5.0990195, 9.0, 5.1478150, 11.0453610, 0.0]))
    assert_almost_equal(rms(x, axis=1), np.array([6.3403469, 7.8740078]))


def test_crestfactor():
    x = np.array([[4, 9, 2, 10, 0], [6, 9, 7, 12, 0]])
    assert_almost_equal(crestfactor(x), 1.6786908)
    assert_almost_equal(
        crestfactor(x, axis=0), np.array([0.3922321, 0.0, 0.9712856, 0.1810714, 0.0])
    )
    assert_almost_equal(crestfactor(x, axis=1), np.array([1.5772005, 1.52400133]))


if __name__ == "__main__":
    test_rms()
    test_crestfactor()

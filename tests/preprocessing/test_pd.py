"""Test code for pd.py

- Author: Hyunjae Kim
- Contact: hyunjae.kim@onepredict.com
"""

from typing import Tuple

import numpy as np

from onebone.preprocessing import pd


def _generate_pspd(coord_point: Tuple[int, int], cum_value: int) -> Tuple[np.ndarray, np.ndarray]:
    prps = np.zeros([3600, 128])
    prpd = np.zeros([128, 128])
    return prps, prpd


def _check_ps2pd():
    coord = (3, 12)
    cum_value = 50
    prps, prpd = _generate_pspd(coord, cum_value)
    prpd_transformed = pd.ps2pd(prps)
    # Check the output
    assert (prpd_transformed == prpd).all()


def test_ps2pd():
    _check_ps2pd()


if __name__ == "__main__":
    test_ps2pd()

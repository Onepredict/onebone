"""Test code for pd.py.

- Author: Hyunjae Kim
- Contact: hyunjae.kim@onepredict.com
"""

from typing import Tuple

import numpy as np

from onebone.preprocessing import pd


def _generate_pspd(coord_point: Tuple[int, int], cum_value: int) -> Tuple[np.ndarray, np.ndarray]:
    time_step = 3600
    phase_resol = 128
    range_amp = (0, 256)
    resol_amp = 128

    prps = np.zeros([time_step, phase_resol])
    idx_pulse = np.random.permutation(time_step)[:cum_value]
    prps[idx_pulse, coord[0]] = coord[1]

    prpd = np.zeros([phase_resol, resol_amp])
    prpd[resol_amp-int(coord[1]/range_amp[1]*resol_amp), coord[0]] = cum_value
    return prps, prpd


def _check_ps2pd():
    coord = (60, 182)
    cum_value = 50
    prps, prpd = _generate_pspd(coord, cum_value)
    prpd_transformed = pd.ps2pd(prps)
    # Check the output
    assert (prpd_transformed == prpd).all()


def test_ps2pd():
    for _ in range(10):
        _check_ps2pd()


if __name__ == "__main__":
    test_ps2pd()

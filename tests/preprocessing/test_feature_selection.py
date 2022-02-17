"""Test Cross Correlation Feature Selection method.

- Author: Junha Jeon
- Contact: junha.jeon@onepredict.com
"""

import numpy as np

from onebone.preprocessing import fs_crosscorrelation


def _generate_x(fs: float):
    # Generate signal
    t = np.linspace(0, 1, int(fs))
    signal_list = []
    signal_list.append(1.0 * np.sin(2 * np.pi * 10.0 * t))
    signal_list.append(1.0 * np.sin(2 * np.pi * 37.0 * t))
    signal_list.append(1.0 * np.sin(2 * np.pi * 20.0 * t))
    signal_list.append(1.0 * np.sin(2 * np.pi * 30.0 * t))
    signal_list.append(5.0 * np.sin(2 * np.pi * 30.0 * t))
    return np.stack(signal_list, axis=1)


def _generate_refer(fs: float):
    # Generate signal
    t = np.linspace(0, 1, int(fs))
    return 1.0 * np.sin(2 * np.pi * 10.0 * t)


def test_fs_crosscorrelation():
    x = _generate_x(1000)
    refer = _generate_refer(1000)
    x_dimreduced = fs_crosscorrelation(x, refer, output_col_num=2)

    assert np.all(
        np.equal(x_dimreduced, x[:, [0, 4]])
    ), "Wrong return: Dimension reduced input is different from expected return"


if __name__ == "__main__":
    test_fs_crosscorrelation()

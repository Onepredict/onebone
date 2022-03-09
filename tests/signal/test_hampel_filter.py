"""Test_hampel_filter.

- Author: Sunjin.kim
- Contact: sunjin.kim@onepredict.com
"""

import numpy as np

from onebone.signal import hampel_filter

## hampel filter 함수 명령어를 참고함
## Series 만 사용해야하는 단점 존재 (np.array _ 1D 만 가능 )


def timeseries_data():
    t = np.linspace(0, 1, 100)
    y = np.sin(2 * np.pi * 10 * t)
    np.put(y, [9, 13, 24, 30, 45], 4)
    return y


def test_hampel_filter():
    first_feature = hampel_filter.hampel_filter(timeseries_data(), 3)[0]
    first_feature
    second_feature = hampel_filter.hampel_filter(timeseries_data(), 4)[0]
    second_feature


if __name__ == "__main__":
    test_hampel_filter()

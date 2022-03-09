"""hampel_filter.
- Author: Sunjin Kim
- Contact: sunjin.kim@onepredict.com
"""

import numpy as np


def hampel_filter(series, window_size, n_sigmas=3, autowindow=False):

    filtered_series = series.copy()
    k = 1.4826  # scale factor for Gaussian distribution

    if type(window_size) != int:
        raise TypeError("type(window_size) = int")
    index = []
    """
    Parameters
    ----------
    x : numpy.ndarray
        1d-timeseries data

    window_size : int
                  정수형만 가능하며, 자신의 데이터에 맞게 Window size 를 조절해줘야한다.

    Returns
    ----------
    filtered_series : numpy.ndarray
                      필터로 인해 Outlier 값이 제거된 값이 나오게 됩니다.
                      현재는 Window size 에 따라 제거 되거나, 제거되지 않는 결과들이있으며,
                      이는 AutoWindow=True 로 개선할 예정입니다.
    index : list
            Outlier 로 판단되어 제거가 된 index 들을 list 로 반환해줍니다.

    References
    ----------
    [1] Pearson, Ronald K., et al. "Generalized hampel filters."
    EURASIP Journal on Advances in Signal Processing 2016.1 (2016): 1-18.
    DOI: 10.1186/s13634-016-0383-6
    [2] https://github.com/MichaelisTrofficus/hampel_filter

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> fs = 100.0
    >>> t = np.linspace(0, 1, int(fs))
    >>> y = np.sin(2 * np.pi *10.0 * t)
    >>> np.put(y,[9, 13, 24, 30, 45, 51,78],4)
    >>> first_hampel_filter_array = hampel_filter.hampel_filter(y,2)[0]
    >>> second_hampel_filter_array = hampel_filter.hampel_filter(y,3)[0]

    >>> ax1 = plt.subplot(2,2,1)
    >>> plt.plot(t,y,label='origin_data',color='b', alpha=0.5)
    >>> plt.legend(loc = 'upper right',fontsize=7)
    >>> ax2 = plt.subplot(2,2,2,sharey=ax1)
    >>> plt.plot(t,first_hampel_filter_array, label='window_size : 2',color='r', alpha=0.5)
    >>> plt.legend(loc = 'upper right',fontsize=7)
    >>> ax3 = plt.subplot(2,2,3,sharey=ax1)
    >>> plt.plot(t,second_hampel_filter_array, label='window_size : 3',color='g', alpha=0.5)
    >>> plt.legend(loc = 'upper right',fontsize=7)
    >>> ax4 = plt.subplot(2,2,4,sharey=ax1)
    >>> plt.plot(t,y,label='origin_data',color='b', alpha=0.5)
    >>> plt.plot(t,first_hampel_filter_array, label='window_size : 2',color='r', alpha=0.5)
    >>> plt.plot(t,second_hampel_filter_array, label='window_size : 3',color='g', alpha=0.5)
    >>> plt.legend(loc = 'upper right',fontsize=7)
    """
    if autowindow is False:
        for i in range((window_size), (len(filtered_series) - window_size)):
            window_median = np.median(filtered_series[(i - window_size) : (i + window_size)])
            S_k = k * np.median(
                np.abs(filtered_series[(i - window_size) : (i + window_size)] - window_median)
            )
            """
            MAD scale estimate, defined as : S_k = 1.4826 * median_j∈-K,K]{|x_(k-j) - m_k|}.
            """
            if np.abs(filtered_series[i] - window_median) > n_sigmas * S_k:
                filtered_series[i] = window_median
                index.append(i)
        return filtered_series, index

    elif autowindow is True:
        # window size를 지정 및 가장 유사한 graph로 후보군을 정리해주는 코드 구현 하고싶음
        raise AttributeError("We are preparing to provide a function")

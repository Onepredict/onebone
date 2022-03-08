"""hampel_filter.

- Author: Sunjin Kim
- Contact: sunjin.kim@onepredict.com
"""

import numpy as np 
## hampel filter 함수 명령어를 참고함 
## Series 만 사용해야하는 단점 존재 (np.array _ 1D 만 가능 )

def hampel_filter(array, window_size, n_sigmas=3, Autowindow = False):
    
    filtered_series = array.copy()
    k = 1.4826 # scale factor for Gaussian distribution

    if type(window_size) != int :
        raise TypeError("type(window_size) = int")
    index = []
    """

    Parameters
    ----------
    x : numpy.ndarray
        1d-timeseries data
    window_size : int,  
                  d


    Returns
    ----------
    hampel_filter : numpy.ndarray
    >>> fs = 1000.0
    >>> t = np.linspace(0, 1, int(fs))
    >>> x = 10.0 * np.sin(2 * np.pi * 20.0 * t)
    >>> snr_array = snr_feature(x, fs)
    
    Examples
    --------



    """
    
    if Autowindow == False:
        for i in range((window_size),(len(array) - window_size)):
            window_median = np.median(array[(i - window_size):(i + window_size)])
            S_k = k * np.median(np.abs(array[(i - window_size):(i + window_size)] - window_median))
            '''
            MAD scale estimate, defined as : S_k = 1.4826 * median_j∈-K,K]{|x_(k-j) - m_k|}.
            '''
            if (np.abs(array[i] - window_median) > n_sigmas * S_k):
                filtered_series[i] = window_median
                index.append(i)
        return filtered_series, index


    elif Autowindow == True:
        # window size를 지정 및 가장 유사한 graph로 후보군을 정리해주는 코드 구현 하고싶음 
        raise  AttributeError('We are preparing to provide a function')

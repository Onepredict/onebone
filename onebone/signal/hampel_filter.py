import numpy as np 
def hampel(input_series, window_size, n_sigmas=3, Autowindow = False):
    
    filtered_series = input_series.copy()
    k = 1.4826 # scale factor for Gaussian distribution
    
    indices = []
    if Autowindow == False:
    # possibly use np.nanmedian 
        for i in range((window_size),(len(input_series) - window_size)):
            window_median = np.median(input_series[(i - window_size):(i + window_size)])
            S_k = k * np.median(np.abs(input_series[(i - window_size):(i + window_size)] - window_median))
            '''
            MAD scale estimate, defined as : S_k = 1.4826 * median_j∈-K,K]{|x_(k-j) - m_k|}.
            '''
            if (np.abs(input_series[i] - window_median) > n_sigmas * S_k):
                filtered_series[i] = window_median
                indices.append(i)
    return filtered_series, indices
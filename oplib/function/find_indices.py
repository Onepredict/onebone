import numpy as np


def find_jump_indices(arr: np.ndarray, gap: float) -> np.ndarray:
    """1-D Array에서 이전과 다음 샘플값 차이가 설정값 이상인 index를 산출

    Parameters
    ----------
    arr: numpy.ndarray
        1차원 data
    gap: float
        간극 설정값

    Returns
    -------
    indices: numpy.ndarray
        조건에 해당하는 Input data의 인덱스값 배열
    """
    diff_arr = np.diff(arr)
    indices = np.where(diff_arr >= gap)[0] + 1
    return indices

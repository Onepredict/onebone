"""Test Moving Average.

- Author: Kibum Park
- Contact: kibum.park@onepredict.com
"""

from typing import Callable, Tuple

import numpy as np
import pytest

from oplib.signal import moving_average


def _generate_sin_signal() -> np.ndarray:
    # Generate signal
    signal = np.arange(1, 11)

    return signal


def check_1d_signal_default(
    function_: Callable, input_args: Tuple[int], expected_return: np.ndarray
):
    signal = _generate_sin_signal()
    ma = function_(signal, *input_args)

    assert np.all(np.isclose(ma, expected_return)), (
        f"Wrong return: The expected return is {expected_return}, " + f"but output is {ma}"
    )


def check_1d_signal_with_pad(
    function_: Callable, input_args: Tuple[int, bool], expected_return: np.ndarray
):
    signal = _generate_sin_signal()
    ma = function_(signal, *input_args)

    assert np.all(np.isclose(ma, expected_return)), (
        f"Wrong return: The expected return is {expected_return}, " + f"but output is {ma}"
    )


def check_1d_signal_with_weights(
    function_: Callable, input_args: Tuple[int, bool, np.ndarray], expected_return: np.ndarray
):
    signal = _generate_sin_signal()
    ma = function_(signal, *input_args)

    assert np.all(np.isclose(ma, expected_return)), (
        f"Wrong return: The expected return is {expected_return}, " + f"but output is {ma}"
    )


def check_2d_signal_wth_all(
    function_: Callable, input_args: Tuple[int, bool, np.ndarray], expected_return: np.ndarray
):
    signal = _generate_sin_signal()
    signal_2d = np.stack([signal] * 2)

    ma = function_(signal_2d, *input_args)

    assert np.all(np.isclose(ma, expected_return)), (
        f"Wrong return: The expected return is {expected_return}, " + f"but output is {ma}"
    )


def check_array_shape(function_: Callable, input_args: Tuple[int]):
    origin_signal = _generate_sin_signal()

    # case 1: signal shape [2, 3, data_length] -> ArrayShapeError
    changed_signal = np.stack([origin_signal] * 3, axis=0)
    changed_signal = np.stack([changed_signal] * 2, axis=0)
    with pytest.raises(ValueError) as ex:
        function_(changed_signal, *input_args)
    assert str(ex.value) == "Dimension of signal must be less than 3."


def test_moving_average():
    check_1d_signal_default(
        moving_average, (3,), np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    )
    check_1d_signal_with_pad(
        moving_average, (3, True), np.array([1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    )
    check_1d_signal_with_weights(
        moving_average,
        (3, False, np.array([1, 1, 2])),
        np.array([9, 13, 17, 21, 25, 29, 33, 37]) / 4,
    )
    check_2d_signal_wth_all(
        moving_average,
        (3, True, np.array([1, 1, 2])),
        np.stack([np.array([4, 20 / 3, 9, 13, 17, 21, 25, 29, 33, 37]) / 4] * 2),
    )
    check_array_shape(moving_average, (3,))


if __name__ == "__main__":
    test_moving_average()

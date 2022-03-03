"""Test for timeit.py.

- Author: Kangwhi Kim
- Contact: kangwhi.kim@onepredict.com
"""

import logging
import time
from pathlib import Path

from numpy.testing import assert_almost_equal

from onebone.utils import Timer

LOG_FILEDIR = "./test_1a2e3w4f9d13kd2jw8.log"
log_file_path = Path(LOG_FILEDIR)


def generate_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(LOG_FILEDIR)
    logger.addHandler(file_handler)
    return logger


def check_timer():
    if log_file_path.exists():
        raise Exception(
            "A log file with the name you specified already exists. Change it to a different file name."
        )

    time.sleep(2)
    logger = generate_logger()

    @Timer(logger)
    def timer_test():
        start = time.time()
        time.sleep(1)
        duration = time.time() - start
        return duration

    expected_return = timer_test()
    time.sleep(2)
    out = log_file_path.read_text()
    log_file_path.unlink()  # Delete the testing log file.
    time.sleep(2)
    out = float(out[out.find("is") + 3 : out.find(" sec")])

    assert_almost_equal(out, expected_return, decimal=1)


def test_timeit():
    check_timer()


if __name__ == "__main__":
    test_timeit()

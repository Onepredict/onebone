"""Timer function.

- Author: Kangwhi Kim
- Contact: kangwhi.kim@onepredict.com
"""

import logging
import time
from functools import wraps


class Timer:
    """
    Check the elapsed time of the function.

    .. note::
        Use it as a function decorator.

    Parameters
    ----------
    logger : logging.Logger, default=None
        A logger.

    Returns
    -------
    wrapper : function
        Wrapper function. When `logger` is not `None`,
        the debug level message is delivered to the logger within the wrapper function.
        But, when `logger` is `None`,
        the message is delivered to the `print` function.

    Examples
    --------
    >>> import logging
    >>> import time
    >>> from onebone.utils import Timer

        Create a logger.
    >>> logger = logging.getLogger(__name__)
    >>> logger.setLevel(logging.DEBUG)
    >>> stream_handler = logging.StreamHandler()
    >>> logger.addHandler(stream_handler)

        Add the `Timer` Decorator to the function.
    >>> @Timer(logger)
    >>> def timer_test():
            start = time.time()
            time.sleep(1)
            duration = time.time() - start
            return duration

        Run the function.
    >>> timer_test()
    """

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start
            if self.logger is None:
                print(f"The elapsed time[{func.__name__}] is {duration} sec")
            else:
                self.logger.debug(f"The elapsed time[{func.__name__}] is {duration} sec")
            return result

        return wrapper

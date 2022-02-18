"""Timer function.

- Author: Kangwhi Kim
- Contact: kangwhi.kim@onepredict.com
"""

from functools import wraps
from time import time


class Timer:
    def __init__(self, logger=None):
        """
        Check the elapsed time of the function.

        Parameters
        ----------

        Returns
        -------

        Examples
        --------
        >>>
        """
        self.logger = logger

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time()
            result = func(*args, **kwargs)
            duration = time() - start
            if self.logger is None:
                print(f"The elapsed time[{func.__name__}] is {duration} sec")
            else:
                self.logger.debug(f"The elapsed time[{func.__name__}] is {duration} sec")
            return result

        return wrapper

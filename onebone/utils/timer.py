from functools import wraps
from time import time


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f"Elapsed time[{func.__name__}]: {end-start} sec")
        print(f"Return[{func.__name__}]: {result}")
        return result

    return wrapper

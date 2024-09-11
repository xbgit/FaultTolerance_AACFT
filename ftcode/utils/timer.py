import time


def timer(fn):
    def wrapper(*args, **kwargs):
        time1 = time.time()
        result = fn(*args, **kwargs)
        time2 = time.time()
        print(fn.__name__ + ' : ' + str(time2 - time1))
        return result
    return wrapper

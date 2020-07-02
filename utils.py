import timeit

# add logging functionality
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        func_val = func(*args, **kwargs)
        elapsed = timeit.default_timer() - start_time
        print('Function "{name}" took {time} seconds to complete.'.format(name=func.__name__, time=elapsed))
        return func_val
    return wrapper


if __name__ == "__main__":
    pass
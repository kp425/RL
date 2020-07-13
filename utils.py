import tensorflow as tf
import timeit

def normalize(x, eps = 1e-8):
    mean = tf.math.reduce_mean(x)
    std = tf.math.reduce_std(x)
    return (x - mean)/(std + eps)


# A decorator to measure execution times of functions
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
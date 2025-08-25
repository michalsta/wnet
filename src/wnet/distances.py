import numpy as np


def wrap_distance_function(dist_func):
    def wrapped_dist(p, y):
        print(p, p.index, p.positions)
        i = p.index
        x = p.positions[:, i : i + 1]
        return dist_func(x[: np.newaxis], y)
    return wrapped_dist


    def wrapped(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        if x.shape != y.shape:
            raise ValueError("Input arrays must have the same shape.")
        return dist_func(x, y)
    return wrapped


class TruncatedL1Distance:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, x, y):
        return np.minimum(np.abs(x - y), self.threshold)
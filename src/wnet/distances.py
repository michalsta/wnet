import numpy as np


def wrap_distance_function(dist_func):
    def wrapped_dist(p, y):
        i = p.index
        x = p.positions[:, i : i + 1]
        return dist_func(x[: np.newaxis], y)

    return wrapped_dist


class Distance:
    def __call__(self, p, y):
        i = p.index
        x = p.positions[:, i : i + 1]
        return self.dist_func(x[: np.newaxis], y)

    def dist_func(self, x, y):
        raise NotImplementedError("Subclasses should implement this method.")


class L1Distance(Distance):
    def dist_func(self, x, y):
        return np.linalg.norm(x - y, axis=0)

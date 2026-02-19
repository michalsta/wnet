"""Tests for the Python Distance classes in wnet.distances."""

import numpy as np
from wnet.distances import L1Distance, Distance, wrap_distance_function


class FakePoint:
    """Mimics the point object passed to Distance.__call__."""

    def __init__(self, positions, index):
        self.positions = positions
        self.index = index


def test_l1distance_direct():
    """L1Distance.dist_func computes correct L1 norms."""
    dist = L1Distance()
    x = np.array([[1.0], [2.0]])  # shape (2, 1)
    y = np.array([[4.0, 7.0], [6.0, 2.0]])  # shape (2, 2)
    result = dist.dist_func(x, y)
    # |1-4|+|2-6| = 7,  |1-7|+|2-2| = 6
    np.testing.assert_array_equal(result, [7.0, 6.0])


def test_l1distance_call():
    """L1Distance.__call__ correctly extracts the point and computes distances."""
    positions = np.array([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]])
    p = FakePoint(positions, index=1)
    y = np.array([[0.0], [0.0]])

    dist = L1Distance()
    result = dist(p, y)
    # Point at index 1 is [20, 2]. L1 distance to [0, 0] = 22.
    np.testing.assert_array_equal(result, [22.0])


def test_wrap_distance_function():
    """wrap_distance_function correctly wraps a plain function."""

    def my_l1(x, y):
        return np.linalg.norm(x - y, ord=1, axis=0)

    wrapped = wrap_distance_function(my_l1)
    positions = np.array([[5.0, 10.0], [3.0, 7.0]])
    p = FakePoint(positions, index=0)
    y = np.array([[0.0], [0.0]])

    result = wrapped(p, y)
    # Point at index 0 is [5, 3]. L1 to [0, 0] = 8.
    np.testing.assert_array_equal(result, [8.0])


def test_l1distance_broadcasting():
    """L1Distance.__call__ broadcasts correctly against multiple targets."""
    positions = np.array([[0.0, 10.0], [0.0, 10.0]])
    p = FakePoint(positions, index=0)
    y = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])

    dist = L1Distance()
    result = dist(p, y)
    # Point [0,0] vs [1,1],[2,2],[3,3] → L1 = 2, 4, 6
    np.testing.assert_array_equal(result, [2.0, 4.0, 6.0])


if __name__ == "__main__":
    test_l1distance_direct()
    print("test_l1distance_direct: OK")
    test_l1distance_call()
    print("test_l1distance_call: OK")
    test_wrap_distance_function()
    print("test_wrap_distance_function: OK")
    test_l1distance_broadcasting()
    print("test_l1distance_broadcasting: OK")

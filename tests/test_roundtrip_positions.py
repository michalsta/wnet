"""Test that VectorDistribution round-trips positions correctly through C++."""

import numpy as np
from wnet.wnet_cpp import (
    CVectorDistribution1,
    CVectorDistribution2,
    CVectorDistribution3,
)


def test_roundtrip_1d():
    positions = np.array([[10.0, 20.0, 30.0]])
    intensities = np.array([1, 2, 3], dtype=np.int64)
    dist = CVectorDistribution1(positions, intensities)
    got = np.array(dist.py_get_positions())
    np.testing.assert_array_equal(got, positions)


def test_roundtrip_2d():
    positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    intensities = np.array([10, 20, 30], dtype=np.int64)
    dist = CVectorDistribution2(positions, intensities)
    got = np.array(dist.py_get_positions())
    np.testing.assert_array_equal(got, positions)


def test_roundtrip_3d():
    positions = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    intensities = np.array([7, 8], dtype=np.int64)
    dist = CVectorDistribution3(positions, intensities)
    got = np.array(dist.py_get_positions())
    np.testing.assert_array_equal(got, positions)


if __name__ == "__main__":
    test_roundtrip_1d()
    print("1D: OK")
    test_roundtrip_2d()
    print("2D: OK")
    test_roundtrip_3d()
    print("3D: OK")

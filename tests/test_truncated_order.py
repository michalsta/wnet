"""
Test that TruncatedWassersteinDistance produces the same result as manually
constructing a WassersteinNetwork with the correct operation order
(add_simple_trash BEFORE build).
This is important because the order of operations in WassersteinNetwork matters:
- add_simple_trash must be called before build, as it modifies the graph structure.
- If add_simple_trash is called after build, it will raise a RuntimeError.
"""

import math

import numpy as np

from wnet import Distribution, WassersteinNetwork
from wnet.distances import DistanceMetric
from wnet.wasserstein import TruncatedWassersteinDistance, _has_fractional_positions


def _correct_truncated_wasserstein(dist1, dist2, distance, max_distance):
    """Reference implementation with the correct order: add_simple_trash before build.

    Mirrors TruncatedWassersteinDistance's construction recipe (cost scaling +
    real threshold for fractional inputs, legacy integer mode otherwise) so the
    comparison isolates the operation-order question this file is about.
    """
    fractional = _has_fractional_positions(dist1, dist2) or float(
        max_distance
    ) != math.floor(max_distance)
    W = WassersteinNetwork(
        dist1, [dist2], distance, max_distance, round_max_distance=not fractional
    )
    if fractional:
        W.set_cost_scaling()
    W.add_simple_trash(max_distance)
    W.build()
    W.solve()
    return W.total_cost()


def test_truncated_wasserstein_order():
    """TruncatedWassersteinDistance should match the correct add_simple_trash-then-build order."""
    S1 = Distribution(np.array([[0, 10]]), np.array([5, 5]))
    S2 = Distribution(np.array([[1, 11]]), np.array([5, 5]))
    max_dist = 5

    expected = _correct_truncated_wasserstein(S1, S2, DistanceMetric.L2, max_dist)
    actual = TruncatedWassersteinDistance(S1, S2, DistanceMetric.L2, max_dist)

    assert actual == expected, (
        f"TruncatedWassersteinDistance returned {actual}, "
        f"but correct order gives {expected}"
    )


def test_truncated_wasserstein_with_trash_needed():
    """Case where some mass must go to trash (points far apart exceed max_distance)."""
    S1 = Distribution(np.array([[0, 100]]), np.array([3, 3]))
    S2 = Distribution(np.array([[1, 101]]), np.array([3, 3]))
    max_dist = 5

    expected = _correct_truncated_wasserstein(S1, S2, DistanceMetric.L2, max_dist)
    actual = TruncatedWassersteinDistance(S1, S2, DistanceMetric.L2, max_dist)

    assert actual == expected, (
        f"TruncatedWassersteinDistance returned {actual}, "
        f"but correct order gives {expected}"
    )


def test_truncated_wasserstein_many_unmatched():
    """Larger case with many points that cannot be matched within max_distance, forcing heavy trash usage."""
    rng = np.random.default_rng(42)
    # Two clusters far apart in S1, only one cluster matchable in S2
    positions1 = np.concatenate(
        [rng.uniform(0, 5, (1, 20)), rng.uniform(500, 505, (1, 20))], axis=1
    )
    intensities1 = rng.integers(1, 10, size=40)
    positions2 = np.concatenate(
        [rng.uniform(0, 5, (1, 20)), rng.uniform(500, 505, (1, 20))], axis=1
    )
    # Match total intensities
    intensities2 = rng.integers(1, 10, size=40)
    diff = int(np.sum(intensities1) - np.sum(intensities2))
    if diff > 0:
        intensities2[0] += diff
    elif diff < 0:
        intensities1[0] -= diff

    S1 = Distribution(positions1, intensities1)
    S2 = Distribution(positions2, intensities2)
    max_dist = 10  # Clusters 500 apart can't be cross-matched

    expected = _correct_truncated_wasserstein(S1, S2, DistanceMetric.L2, max_dist)
    actual = TruncatedWassersteinDistance(S1, S2, DistanceMetric.L2, max_dist)

    assert actual == expected, (
        f"TruncatedWassersteinDistance returned {actual}, "
        f"but correct order gives {expected}"
    )


def test_add_simple_trash_after_build_raises():
    """Calling add_simple_trash after build must raise RuntimeError."""
    S1 = Distribution(np.array([[0, 10]]), np.array([5, 5]))
    S2 = Distribution(np.array([[1, 11]]), np.array([5, 5]))

    W = WassersteinNetwork(S1, [S2], DistanceMetric.L2, 5)
    W.build()
    try:
        W.add_simple_trash(5)
        assert False, "Expected RuntimeError was not raised"
    except RuntimeError:
        pass


if __name__ == "__main__":
    test_truncated_wasserstein_order()
    print("test_truncated_wasserstein_order passed")
    test_truncated_wasserstein_with_trash_needed()
    print("test_truncated_wasserstein_with_trash_needed passed")
    test_truncated_wasserstein_many_unmatched()
    print("test_truncated_wasserstein_many_unmatched passed")
    test_add_simple_trash_after_build_raises()
    print("test_add_simple_trash_after_build_raises passed")
    print("All passed")

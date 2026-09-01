"""InfeasibleError: trash-less quantization imbalance must throw, never requantize."""

import numpy as np
import pytest

import wnet
from wnet import Distribution_1D, InfeasibleError, WassersteinDistance
from wnet.distances import DistanceMetric


def _fractional_equal_mass_pair(seed):
    """Two distributions with exactly equal real total mass, fractional peaks."""
    rng = np.random.default_rng(seed)
    n = 20
    i1 = rng.random(n) + 0.1
    i2 = rng.random(n) + 0.1
    i2 *= i1.sum() / i2.sum()  # exact equal totals as reals
    d1 = Distribution_1D(np.sort(rng.random(n) * 100), i1)
    d2 = Distribution_1D(np.sort(rng.random(n) * 100), i2)
    return d1, d2


def test_fractional_equal_mass_raises_infeasible_error():
    # Per-peak truncation almost always unbalances the integer totals; the
    # solve must fail loudly with InfeasibleError, not silently requantize.
    raised = 0
    for seed in range(10):
        d1, d2 = _fractional_equal_mass_pair(seed)
        try:
            WassersteinDistance(d1, d2, DistanceMetric.L1)
        except InfeasibleError as e:
            raised += 1
            msg = str(e)
            assert "trash" in msg
            assert "quantised" in msg or "integer" in msg
    assert raised > 0, "expected at least one quantization imbalance in 10 seeds"


def test_infeasible_error_is_runtime_error_subclass():
    # Pre-existing `except RuntimeError` handlers must keep working.
    assert issubclass(InfeasibleError, RuntimeError)
    d1, d2 = _fractional_equal_mass_pair(0)
    for seed in range(10):
        d1, d2 = _fractional_equal_mass_pair(seed)
        try:
            WassersteinDistance(d1, d2, DistanceMetric.L1)
        except RuntimeError:
            return
    pytest.skip("no imbalance in 10 seeds")


def test_integer_balanced_still_works():
    d1 = Distribution_1D(np.array([0.0, 10.0]), np.array([5.0, 5.0]))
    d2 = Distribution_1D(np.array([3.0, 10.0]), np.array([5.0, 5.0]))
    assert WassersteinDistance(d1, d2, DistanceMetric.L1) == 15.0


def test_trash_absorbs_imbalance():
    # The documented escape hatch: trash edges make fractional masses fine.
    from wnet import TruncatedWassersteinDistance

    d1, d2 = _fractional_equal_mass_pair(1)
    d = TruncatedWassersteinDistance(d1, d2, DistanceMetric.L1, max_distance=5.0)
    assert np.isfinite(d) and d >= 0

"""Tests for the RuntimeError raised by NetworkSimplex when no trash edges are present."""

import numpy as np
import pytest
from wnet import Distribution, WassersteinNetwork
from wnet.distances import DistanceMetric


def _small_wnet(method="network_simplex", add_trash=False):
    rng = np.random.default_rng(0)
    base = Distribution(
        rng.uniform(size=(2, 8)) * 10, (rng.uniform(size=(8,)) * 100).astype(int) + 1
    )
    target = [
        Distribution(
            rng.uniform(size=(2, 6)) * 10, (rng.uniform(size=(6,)) * 100).astype(int) + 1
        )
    ]
    W = WassersteinNetwork(base, target, distance=DistanceMetric.L2, max_distance=5, method=method)
    if add_trash:
        W.add_simple_trash(1000)
    W.build()
    return W


def test_network_simplex_no_trash_raises():
    W = _small_wnet("network_simplex")
    with pytest.raises(RuntimeError, match="NetworkSimplex"):
        W.solve()


def test_network_simplex_no_trash_raises_with_point():
    W = _small_wnet("network_simplex")
    with pytest.raises(RuntimeError, match="NetworkSimplex"):
        W.solve([0.5])


def test_network_simplex_with_trash_ok():
    W = _small_wnet("network_simplex", add_trash=True)
    W.solve()
    assert W.total_cost() >= 0


def test_cost_scaling_no_trash_ok():
    W = _small_wnet("cost_scaling")
    W.solve()
    assert W.total_cost() >= 0


def test_capacity_scaling_no_trash_ok():
    W = _small_wnet("capacity_scaling")
    W.solve()
    assert W.total_cost() >= 0


def test_cost_scaling_matches_network_simplex_with_trash():
    """Both solvers should agree on total cost when trash is present."""
    W_ns = _small_wnet("network_simplex", add_trash=True)
    W_cs = _small_wnet("cost_scaling", add_trash=True)
    W_ns.solve()
    W_cs.solve()
    assert W_ns.total_cost() == W_cs.total_cost()

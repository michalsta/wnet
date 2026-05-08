"""Tests for the RuntimeError raised by all solvers when no trash edges are present."""

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
            rng.uniform(size=(2, 6)) * 10,
            (rng.uniform(size=(6,)) * 100).astype(int) + 1,
        )
    ]
    W = WassersteinNetwork(
        base, target, distance=DistanceMetric.L2, max_distance=5, method=method
    )
    if add_trash:
        W.add_simple_trash(1000)
    W.build()
    return W


@pytest.mark.parametrize(
    "method", ["network_simplex", "cost_scaling", "capacity_scaling"]
)
def test_no_trash_raises(method):
    W = _small_wnet(method)
    with pytest.raises(RuntimeError, match="without trash edges"):
        W.solve()


@pytest.mark.parametrize(
    "method", ["network_simplex", "cost_scaling", "capacity_scaling"]
)
def test_no_trash_raises_with_point(method):
    W = _small_wnet(method)
    with pytest.raises(RuntimeError, match="without trash edges"):
        W.solve([0.5])


@pytest.mark.parametrize(
    "method", ["network_simplex", "cost_scaling", "capacity_scaling"]
)
def test_with_trash_ok(method):
    W = _small_wnet(method, add_trash=True)
    W.solve()
    assert W.total_cost() >= 0


def test_cost_scaling_matches_network_simplex_with_trash():
    """Both solvers should agree on total cost when trash is present."""
    W_ns = _small_wnet("network_simplex", add_trash=True)
    W_cs = _small_wnet("cost_scaling", add_trash=True)
    W_ns.solve()
    W_cs.solve()
    assert W_ns.total_cost() == W_cs.total_cost()

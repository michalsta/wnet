"""total_cost() accumulates in 128 bits and refuses to wrap.

The scaled total is a sum of int64 products (flow x cost per arc, trash price
x supply).  Those products overflow int64 long before the solver's own range
is exceeded, so the network now sums them in 128 bits, narrows once, and
raises OverflowError instead of returning a wrapped value.  The solve-time
budget guard on WassersteinNetwork.solve() normally prevents this; a subgraph
solved directly bypasses it, which is how these tests reach the overflow.
"""

import numpy as np
import pytest

from wnet import Distribution, WassersteinNetwork
from wnet.distances import DistanceMetric
from wnet.wnet_cpp import NetworkSimplex


def network(cost_scale, intensity_scale, trash="simple"):
    base = Distribution(np.array([[0.0, 1.0], [0.0, 0.0]]), np.array([1.0, 1.0]))
    target = Distribution(np.array([[0.5, 3.0], [0.0, 0.0]]), np.array([1.0, 1.0]))
    g = WassersteinNetwork(base, [target], DistanceMetric.LINF, solver=NetworkSimplex(),
                           max_distance=5.0, intensity_scale=intensity_scale, round_max_distance=False)
    g.set_cost_scaling(cost_scale)
    if trash == "simple":
        g.add_simple_trash(4.0)
    else:
        g.add_independent_asymmetric_trash(2.0, 2.0)
    g.build()
    return g


@pytest.mark.parametrize("trash", ["simple", "independent"])
def test_ordinary_totals_unchanged(trash):
    g = network(1000, 1000.0, trash)
    g.solve([1.0])
    raw = g.wnet.total_cost()
    sg = g.wnet.get_subgraph(0)
    assert raw == sg.total_cost()
    assert g.total_cost() == pytest.approx(raw / (g.scale_factor() * g.intensity_scale_factor()))
    # Repeated queries hit the per-solution cache and agree; a new solve refreshes it.
    assert g.wnet.total_cost() == raw
    g.solve([0.5])
    assert g.wnet.total_cost() != raw


@pytest.mark.parametrize("trash", ["simple", "independent"])
def test_overflowing_total_raises_instead_of_wrapping(trash):
    # Per-arc products of ~2^42 cost units and ~2^32 flow units are ~2^74:
    # far past int64, while every individual cost and supply fits easily.
    g = network(2**40, 2.0**32, trash)
    sg = g.wnet.get_subgraph(0)
    sg.set_point([1.0])          # bypasses the network's solve-time budget guard
    with pytest.raises(OverflowError, match="int64 range"):
        sg.total_cost()


def test_network_solve_guard_still_rejects_first():
    g = network(2**40, 2.0**32)
    with pytest.raises(OverflowError):
        g.solve([1.0])


# ---------------------------------------------------------------------------
# The NetworkSimplex total comes from strong duality (potentials, supplies and
# the finite-capacity arcs) instead of a scan over every arc.  It must equal
# the scan exactly on every solve, warm or cold, for every trash model.
# ---------------------------------------------------------------------------

from wnet.wnet_cpp import WarmMode, CostScaling  # noqa: E402


def random_network(solver, trash, seed, p=1, n=80):
    rng = np.random.default_rng(seed)
    def dist(k):
        return Distribution(rng.uniform(0, 10, size=(2, k)), rng.uniform(0.2, 2.0, size=k))
    g = WassersteinNetwork(dist(n), [dist(n // 2), dist(n // 3)], DistanceMetric.LINF, solver=solver,
                           p=p, max_distance=2.5, intensity_scale=1000.0, round_max_distance=False)
    g.set_cost_scaling(0)
    if trash == "simple":
        g.add_simple_trash(1.5 ** p)
    elif trash == "asymmetric":
        g.add_experimental_trash(1.0 ** p); g.add_theoretical_trash(1.5 ** p)
    else:
        g.add_independent_asymmetric_trash(0.8 ** p, 0.9 ** p)
    g.build()
    return g


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("trash", ["simple", "asymmetric", "independent"])
@pytest.mark.parametrize("warm", [WarmMode.NONE, WarmMode.DualRatio, WarmMode.Dual, WarmMode.Primal])
def test_duality_total_equals_scan(p, trash, warm):
    cfg = NetworkSimplex(); cfg.warm = warm
    g = random_network(cfg, trash, seed=17, p=p)
    ref = random_network(CostScaling(), trash, seed=17, p=p)
    rng = np.random.default_rng(1)
    for _ in range(12):
        point = list(rng.uniform(0.1, 1.5, size=2))
        g.solve(point); ref.solve(point)
        assert g.wnet.total_cost() == ref.wnet.total_cost()   # independent algorithm, same optimum
        for k in range(g.no_subgraphs()):
            sg = g.wnet.get_subgraph(k)
            bracket = sg.total_cost() - sg.total_cost_by_scan()
            if trash != "independent":
                assert bracket == 0
    g.refine_intensity_precision(10); ref.refine_intensity_precision(10)
    g.solve([0.7, 0.4]); ref.solve([0.7, 0.4])
    assert g.wnet.total_cost() == ref.wnet.total_cost()
